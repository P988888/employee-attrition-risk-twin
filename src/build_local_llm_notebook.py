from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "员工离职风险_本地模型分析.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r'''
# 员工离职风险：本地最小模型分析 Notebook

这个 Notebook 把两类模型分开：

- **预测模型**：使用已有的离职风险 Pipeline 计算风险分数；同时训练一个轻量逻辑回归，让你看到完整的训练、验证、阈值标定和测试过程。
- **本地大语言模型**：默认调用本机 Ollama 的 `qwen2.5:1.5b`（当前本机已安装的最小通用模型），启动后会先展示本地数据概览，然后可以直接用自然语言提问；模型只负责解释数据和提出可逆的沟通/核查建议，不替代预测模型，也不自动做任何人事决定。

## 运行前准备

在终端执行：

```bash
cd "/Users/pengjianghua/Desktop/线性回归预测"
ollama serve
# 另开一个终端窗口执行一次
ollama pull qwen2.5:1.5b
jupyter lab notebooks/
```

如果你本地已经安装了其他模型，只需要在第一个配置单元修改 `OLLAMA_MODEL`。

> 按顺序运行 Notebook 后，会自动打印数据概览并进入自由问答。输入自然语言问题即可，不需要手动拼接提示词；输入 `q`、`quit` 或 `退出` 结束。
'''),
    md(r'''
## 1. 配置与环境检查

先运行本单元，再按顺序运行后面的单元。这里不会自动调用云端服务，也不会上传数据。
'''),
    code(r'''
from pathlib import Path
import json
import re
import csv
import io
import time
import urllib.request
import urllib.error
from datetime import datetime

import numpy as np
import pandas as pd

RANDOM_STATE = 42
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"  # 改成你本地已经安装的模型名
TOP_N = 5
RUN_BATCH = False  # 默认不自动消耗 Top-N 调用；需要时在第 6 节改为 True
SHOW_INTERACTIVE_CHAT = True


def find_project_root():
    here = Path.cwd().resolve()
    candidates = [here, *here.parents]
    for p in candidates:
        if (p / "data").exists() and (p / "output").exists():
            return p
    raise FileNotFoundError("找不到项目根目录：需要同时存在 data/ 和 output/ 文件夹")


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
LLM_DIR = OUTPUT_DIR / "llm_analysis"
LLM_DIR.mkdir(parents=True, exist_ok=True)

print(f"项目根目录: {ROOT}")
print(f"Python 数据路径: {DATA_DIR}")
print(f"Ollama 地址: {OLLAMA_HOST}")
print(f"本地模型: {OLLAMA_MODEL}")
print(f"本次分析数量: Top {TOP_N}")
print("提示：即使 Ollama 尚未启动，前面的数据和训练单元仍然可以正常运行。")
'''),
    code(r'''
# 只检查本机 Ollama，不会发送员工数据。
def check_ollama(host=OLLAMA_HOST, model=OLLAMA_MODEL, timeout=2):
    tags_url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        names = [x.get("name", "") for x in payload.get("models", [])]
        exact = model in names
        print(f"Ollama 已连接，已安装模型数: {len(names)}")
        print("已安装模型:", names or "（暂时没有）")
        print(f"目标模型 {model}: {'已找到' if exact else '未找到'}")
        if not exact:
            print(f"请在终端执行：ollama pull {model}")
        return {"ok": True, "model_found": exact, "models": names}
    except Exception as exc:
        print("Ollama 当前不可用：", exc)
        print("需要调用本地模型时，请先执行：ollama serve")
        return {"ok": False, "model_found": False, "models": [], "error": str(exc)}


ollama_status = check_ollama()
'''),
    md(r'''
## 2. 加载数据和已经训练好的风险模型

项目当前的主模型保存在 `output/models/best_model.pkl`。它是一个完整 Pipeline，已经包含缺失值处理、标准化和 One-Hot 编码，因此预测时不能再次手工标准化。

训练口径：

- 真实数据 1470 条用于模型评估；
- 模拟数据只用于大屏体量，不参与主指标；
- 本单元会明确打印真实/模拟数量、模型名称、阈值和测试指标。
'''),
    code(r'''
DATA_FILE = DATA_DIR / "hr_merged.csv"
if not DATA_FILE.exists():
    DATA_FILE = DATA_DIR / "ibm_hr_attrition_raw.csv"

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
print(f"读取文件: {DATA_FILE.name}")
print(f"总数据形状: {df.shape}")
print("字段预览:", list(df.columns[:12]), "...")
if "_source" in df.columns:
    print("数据来源分布:")
    print(df["_source"].value_counts(dropna=False).to_string())
if "Attrition" in df.columns:
    print("离职标签分布:")
    print(df["Attrition"].value_counts(normalize=False).to_string())

MODEL_FILE = MODEL_DIR / "best_model.pkl"
META_FILE = MODEL_DIR / "model_meta.json"
if not MODEL_FILE.exists() or not META_FILE.exists():
    raise FileNotFoundError("缺少已有模型产物，请先运行 python3 src/step1_prepare_data.py 和 python3 src/step2_train_model.py")

import joblib

saved_pipe = joblib.load(MODEL_FILE)
model_meta = json.loads(META_FILE.read_text(encoding="utf-8"))
print("已有模型:", model_meta.get("best_model"))
print("风险阈值:", model_meta.get("best_threshold"))
print("测试集指标:", {k: model_meta.get("real_metrics", {}).get(k) for k in ["auc", "f1", "precision", "recall", "accuracy"]})
'''),
    code(r'''
# 与 src/step2_train_model.py 保持一致的派生特征。
def engineer(frame):
    d = frame.copy()
    eps = 1e-6
    d["IncomePerLevel"] = d["MonthlyIncome"] / (d["JobLevel"] * 5000.0 + eps)
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1.0)
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1.0)
    d["ManagerStability"] = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1.0)
    d["SatisfactionScore"] = (
        d["JobSatisfaction"] + d["EnvironmentSatisfaction"]
        + d["RelationshipSatisfaction"] + d["WorkLifeBalance"]
    ) / 4.0
    d["CareerStagnation"] = (
        d["YearsSinceLastPromotion"] / (d["YearsInCurrentRole"] + 1.0)
    ) * (1.0 - d["JobInvolvement"] / 4.0)
    d["CompanyHopRate"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1.0)
    return d


def risk_level(score):
    if score < 0.25:
        return "低风险"
    if score < 0.50:
        return "中风险"
    if score < 0.75:
        return "高风险"
    return "极高风险"


scored = engineer(df)
feature_cols = model_meta["features"]
missing = [c for c in feature_cols if c not in scored.columns]
if missing:
    raise KeyError(f"模型需要但数据缺少的字段: {missing}")

scored["model_risk_score"] = saved_pipe.predict_proba(scored[feature_cols])[:, 1]
scored["model_risk_level"] = scored["model_risk_score"].map(risk_level)
scored = scored.sort_values("model_risk_score", ascending=False).reset_index(drop=True)

print("模型评分完成")
print(scored["model_risk_level"].value_counts().reindex(["低风险", "中风险", "高风险", "极高风险"], fill_value=0).to_string())
print("最高风险记录（匿名编号、岗位、部门、分数）:")
preview_cols = [c for c in ["JobRole", "Department", "OverTime", "JobSatisfaction", "YearsAtCompany", "model_risk_score", "model_risk_level"] if c in scored.columns]
preview = scored.loc[:TOP_N-1, preview_cols].copy()
preview.insert(0, "record_id", [f"EMP-{i+1:04d}" for i in range(len(preview))])
print(preview.to_string(index=False))
'''),
    md(r'''
## 3. 轻量预测模型：完整显示训练过程

这里训练一个独立的逻辑回归教学模型。它不会覆盖 `best_model.pkl`，也不会改动看板数据。

训练、验证、测试职责：

- 训练集：拟合参数；
- 验证集：选择阈值；
- 测试集：只做最终一次评估。
'''),
    code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REAL = scored[scored.get("_source", pd.Series("real", index=scored.index)).eq("real")].copy()
if "_source" not in scored.columns:
    REAL = scored.copy()

TARGET = "Attrition"
NUMERIC_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]
DERIVED_COLS = [
    "IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
    "SatisfactionScore", "CareerStagnation", "CompanyHopRate",
]
TRAIN_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + DERIVED_COLS

X = REAL[TRAIN_FEATURES].copy()
y = REAL[TARGET].eq("Yes").astype(int)
X_trv, X_te, y_trv, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_trv, y_trv, test_size=0.25, stratify=y_trv, random_state=RANDOM_STATE
)
print(f"训练集: {len(X_tr)} 条，离职率 {y_tr.mean():.2%}")
print(f"验证集: {len(X_va)} 条，离职率 {y_va.mean():.2%}")
print(f"测试集: {len(X_te)} 条，离职率 {y_te.mean():.2%}")

try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC_COLS + DERIVED_COLS),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OHE),
    ]), CATEGORICAL_COLS),
])

light_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE)),
])
print("开始训练轻量逻辑回归 …")
light_pipe.fit(X_tr, y_tr)
print("训练完成。")

val_proba = light_pipe.predict_proba(X_va)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_va, val_proba)
f1_values = 2 * prec * rec / (prec + rec + 1e-12)
light_threshold = float(thresholds[np.argmax(f1_values[:-1])]) if len(thresholds) else 0.5
print(f"验证集阈值搜索完成：F1 最优阈值 = {light_threshold:.4f}")

def evaluate_split(name, X_part, y_part):
    proba = light_pipe.predict_proba(X_part)[:, 1]
    pred = (proba >= light_threshold).astype(int)
    result = {
        "split": name,
        "n": len(y_part),
        "auc": roc_auc_score(y_part, proba),
        "ap": average_precision_score(y_part, proba),
        "accuracy": accuracy_score(y_part, pred),
        "precision": precision_score(y_part, pred, zero_division=0),
        "recall": recall_score(y_part, pred, zero_division=0),
        "f1": f1_score(y_part, pred, zero_division=0),
        "confusion": confusion_matrix(y_part, pred).tolist(),
    }
    print(f"{name}: AUC={result['auc']:.4f} | AP={result['ap']:.4f} | Accuracy={result['accuracy']:.4f} | Precision={result['precision']:.4f} | Recall={result['recall']:.4f} | F1={result['f1']:.4f}")
    print("混淆矩阵 [[TN, FP], [FN, TP]]:", result["confusion"])
    return result

light_metrics = [
    evaluate_split("验证集", X_va, y_va),
    evaluate_split("测试集", X_te, y_te),
]
'''),
    md(r'''
## 4. 启动即查看数据，并准备自由提问

下面这个单元会先把当前本地数据整理成一个可读的概览并立即打印。之后你可以直接问自然语言问题，例如“哪个部门离职率最高？”或“EMP-0001 的风险信号是什么？”，不需要自己拼接提示词。

数据来自本地 `hr_merged.csv`，不是实时数据；模型回答只能依据这里提供的统计和风险评分。
'''),
    code(r'''
def value(row, col, default="未知"):
    v = row.get(col, default)
    if pd.isna(v):
        return default
    return v


def build_context(row, rank):
    score = float(row["model_risk_score"])
    return {
        "record_id": f"EMP-{rank:04d}",
        "risk_score": round(score, 4),
        "risk_level": row["model_risk_level"],
        "department": str(value(row, "Department")),
        "job_role": str(value(row, "JobRole")),
        "overtime": str(value(row, "OverTime")),
        "business_travel": str(value(row, "BusinessTravel")),
        "job_satisfaction_1_to_4": int(value(row, "JobSatisfaction", 0)),
        "environment_satisfaction_1_to_4": int(value(row, "EnvironmentSatisfaction", 0)),
        "work_life_balance_1_to_4": int(value(row, "WorkLifeBalance", 0)),
        "years_at_company": int(value(row, "YearsAtCompany", 0)),
        "years_in_current_role": int(value(row, "YearsInCurrentRole", 0)),
        "years_since_last_promotion": int(value(row, "YearsSinceLastPromotion", 0)),
        "years_with_current_manager": int(value(row, "YearsWithCurrManager", 0)),
        "distance_from_home": int(value(row, "DistanceFromHome", 0)),
        "monthly_income_band": "低" if float(value(row, "MonthlyIncome", 0)) < 4000 else "中" if float(value(row, "MonthlyIncome", 0)) < 9000 else "高",
    }


def rate_table(frame, group_col):
    grouped = frame.groupby(group_col, dropna=False)["Attrition"].agg(
        员工数="size", 离职数=lambda s: s.eq("Yes").sum())
    grouped["离职率"] = grouped["离职数"] / grouped["员工数"]
    return grouped.sort_values("离职率", ascending=False)


def make_data_snapshot():
    real = REAL.copy()
    attrition_count = int(real["Attrition"].eq("Yes").sum())
    attrition_rate = attrition_count / max(len(real), 1)
    overtime = rate_table(real, "OverTime")
    departments = rate_table(real, "Department")
    roles = rate_table(real, "JobRole")
    risk_counts = scored["model_risk_level"].value_counts().reindex(
        ["低风险", "中风险", "高风险", "极高风险"], fill_value=0)
    top_rows = []
    for i, (_, row) in enumerate(scored.head(TOP_N).iterrows(), start=1):
        top_rows.append({
            "record_id": f"EMP-{i:04d}",
            "department": str(value(row, "Department")),
            "job_role": str(value(row, "JobRole")),
            "risk_score": round(float(row["model_risk_score"]), 4),
            "risk_level": row["model_risk_level"],
        })
    return {
        "data_file": DATA_FILE.name,
        "total_records": int(len(df)),
        "real_records": int(len(real)),
        "simulated_records": int(len(df) - len(real)),
        "historical_attrition_count": attrition_count,
        "historical_attrition_rate": round(attrition_rate, 4),
        "overtime_attrition_rate": {str(k): round(float(v["离职率"]), 4) for k, v in overtime.iterrows()},
        "department_attrition_rate": {str(k): round(float(v["离职率"]), 4) for k, v in departments.iterrows()},
        "job_role_attrition_rate_top": {str(k): round(float(v["离职率"]), 4) for k, v in roles.head(6).iterrows()},
        "risk_level_distribution": {str(k): int(v) for k, v in risk_counts.items()},
        "top_risk_records": top_rows,
        "risk_model": model_meta.get("best_model"),
        "test_metrics": {k: model_meta.get("real_metrics", {}).get(k) for k in ["auc", "f1", "precision", "recall", "accuracy"]},
    }


def display_data_snapshot(snapshot):
    print("=" * 80)
    print("本地员工离职风险数据概览")
    print("=" * 80)
    print(f"数据文件：{snapshot['data_file']}")
    print(f"总记录：{snapshot['total_records']} 条 | 真实数据：{snapshot['real_records']} 条 | 模拟数据：{snapshot['simulated_records']} 条")
    print(f"真实数据历史离职：{snapshot['historical_attrition_count']} 人 | 历史离职率：{snapshot['historical_attrition_rate']:.2%}")
    print("\n风险等级分布：")
    print(pd.Series(snapshot["risk_level_distribution"], name="人数").to_string())
    print("\n加班与离职率：")
    print(pd.Series(snapshot["overtime_attrition_rate"], name="离职率").map(lambda x: f"{x:.2%}").to_string())
    print("\n部门离职率：")
    dept_view = pd.DataFrame.from_dict(snapshot["department_attrition_rate"], orient="index", columns=["离职率"])
    print(dept_view.assign(离职率=dept_view["离职率"].map(lambda x: f"{x:.2%}")).to_string())
    print("\n高离职风险岗位 Top 6：")
    role_view = pd.DataFrame.from_dict(snapshot["job_role_attrition_rate_top"], orient="index", columns=["离职率"])
    print(role_view.assign(离职率=role_view["离职率"].map(lambda x: f"{x:.2%}")).to_string())
    print("\n模型和最高风险匿名记录：")
    print(f"模型：{snapshot['risk_model']} | 测试 AUC：{snapshot['test_metrics'].get('auc')} | 测试 F1：{snapshot['test_metrics'].get('f1')}")
    print(pd.DataFrame(snapshot["top_risk_records"]).to_string(index=False))


data_snapshot = make_data_snapshot()
display_data_snapshot(data_snapshot)
scored["record_id"] = [f"EMP-{i:04d}" for i in range(1, len(scored) + 1)]
context_by_id = {rid: build_context(row, i + 1) for i, (rid, (_, row)) in enumerate(zip(scored["record_id"], scored.iterrows()))}
all_contexts = list(context_by_id.values())
contexts = all_contexts[:TOP_N]
example_context = contexts[0] if contexts else None
print(f"\n已为 {len(all_contexts)} 条记录生成匿名编号 EMP-0001 到 EMP-{len(all_contexts):04d}。")
print("查询单个员工请用 lookup_employee('EMP-0123')，或直接问模型。")
'''),
    md(r'''
## 5. 调用本地 Ollama 模型并支持自由提问

`ask("你的问题")` 会自动把与问题相关的统计数据和匿名员工信息交给本地模型。本单元先定义提示词拼接、事实筛选、`ask()` 和交互问答循环。
'''),
    code(r'''
def build_prompt(context):
    return f"""你是企业人力分析助手，只做风险解释和可逆的关注建议，不做自动人事决定。
请根据下面的匿名员工风险数据进行谨慎分析。

硬性要求：
1. 只能返回一个 JSON 对象，不要 Markdown 代码围栏，不要额外文字。
2. 必须包含字段：risk_summary、evidence、recommended_actions、priority、uncertainty、human_review_required。
3. evidence 和 recommended_actions 必须是字符串数组。
4. priority 只能是 low、medium、high 之一。
5. human_review_required 必须始终为 true。
6. 不得建议辞退、降薪、处罚、排斥或任何不可逆的人事动作。
7. 不要把模型分数当作事实；明确说明这是筛查信号，需要人工核实。

匿名数据：
{json.dumps(context, ensure_ascii=False, indent=2)}"""


def snapshot_as_facts(snapshot):
    lines = [
        f"- 数据文件：{snapshot['data_file']}（本地历史数据，不是实时数据）",
        f"- 总记录 {snapshot['total_records']} 条：真实数据 {snapshot['real_records']} 条，模拟数据 {snapshot['simulated_records']} 条",
        f"- 真实数据离职人数 {snapshot['historical_attrition_count']} 人，历史离职率 {snapshot['historical_attrition_rate']:.2%}",
    ]
    for level, count in snapshot["risk_level_distribution"].items():
        lines.append(f"- 模型风险等级 {level}：{count} 人")
    for label, name in [("Yes", "加班"), ("No", "不加班")]:
        if label in snapshot["overtime_attrition_rate"]:
            lines.append(f"- {name}员工历史离职率：{snapshot['overtime_attrition_rate'][label]:.2%}")
    for dept, rate in snapshot["department_attrition_rate"].items():
        lines.append(f"- 部门 {dept} 的历史离职率：{rate:.2%}")
    for role, rate in snapshot["job_role_attrition_rate_top"].items():
        lines.append(f"- 岗位 {role} 的历史离职率：{rate:.2%}")
    for item in snapshot["top_risk_records"]:
        lines.append(
            f"- 匿名员工 {item['record_id']}：{item['department']} / {item['job_role']}，"
            f"风险分数 {item['risk_score']}，风险等级 {item['risk_level']}")
    metrics = snapshot["test_metrics"]
    lines.append(
        f"- 风险模型 {snapshot['risk_model']} 的测试指标："
        f"AUC {metrics.get('auc')}，F1 {metrics.get('f1')}，"
        f"Precision {metrics.get('precision')}，Recall {metrics.get('recall')}")
    return "\n".join(lines)


def context_as_facts(context):
    return "\n".join([
        f"- 匿名员工 {context['record_id']}：部门 {context['department']}，岗位 {context['job_role']}，"
        f"风险分数 {context['risk_score']}（{context['risk_level']}）",
        f"  加班情况 {context['overtime']}，出差频率 {context['business_travel']}",
        f"  工作满意度 {context['job_satisfaction_1_to_4']}/4，环境满意度 {context['environment_satisfaction_1_to_4']}/4，"
        f"工作生活平衡 {context['work_life_balance_1_to_4']}/4",
        f"  司龄 {context['years_at_company']} 年，当前岗位 {context['years_in_current_role']} 年，"
        f"距上次晋升 {context['years_since_last_promotion']} 年，与现任经理共事 {context['years_with_current_manager']} 年",
        f"  通勤距离 {context['distance_from_home']}，月收入档位 {context['monthly_income_band']}",
    ])


def retrieve_facts(question):
    # 小模型不擅长自己在长文本里找数据，先用确定性规则挑出与问题相关的统计数据。
    q = str(question).strip()
    snapshot = data_snapshot
    facts = []
    if re.search(r"部门|Sales|Research|Human Resources|研发|销售|人力", q, flags=re.I):
        for dept, rate in snapshot["department_attrition_rate"].items():
            facts.append(f"部门 {dept} 的历史离职率是 {rate:.2%}")
    if re.search(r"岗位|职位|JobRole", q, flags=re.I):
        for role, rate in snapshot["job_role_attrition_rate_top"].items():
            facts.append(f"岗位 {role} 的历史离职率是 {rate:.2%}")
    if re.search(r"加班|超时|OverTime", q, flags=re.I):
        for label, name in [("Yes", "加班"), ("No", "不加班")]:
            if label in snapshot["overtime_attrition_rate"]:
                facts.append(f"{name}员工的历史离职率是 {snapshot['overtime_attrition_rate'][label]:.2%}")
        facts.append(f"全部真实员工的历史离职率是 {snapshot['historical_attrition_rate']:.2%}")
    if re.search(r"风险等级|风险分布|多少.{0,4}风险|高风险人数", q):
        for level, count in snapshot["risk_level_distribution"].items():
            facts.append(f"模型风险等级 {level} 有 {count} 人")
    if re.search(r"AUC|F1|精确率|召回率|指标|Precision|Recall|模型效果", q, flags=re.I):
        metrics = snapshot["test_metrics"]
        facts.append(
            f"风险模型 {snapshot['risk_model']} 的测试集 AUC 是 {metrics.get('auc')}，F1 是 {metrics.get('f1')}，"
            f"Precision 是 {metrics.get('precision')}，Recall 是 {metrics.get('recall')}")
    if re.search(r"最高风险|风险最高|Top|哪些员工|匿名记录", q, flags=re.I):
        for item in snapshot["top_risk_records"]:
            facts.append(
                f"匿名员工 {item['record_id']} 属于 {item['department']} 的 {item['job_role']}，"
                f"风险分数 {item['risk_score']}（{item['risk_level']}）")
    if re.search(r"离职率|离职人数|总体|多少人离职|历史", q):
        facts.append(
            f"真实数据共 {snapshot['real_records']} 条，其中离职 {snapshot['historical_attrition_count']} 人，"
            f"历史离职率 {snapshot['historical_attrition_rate']:.2%}")
    return facts


def build_chat_prompt(question):
    question = str(question).strip()
    matched_ids = re.findall(r"EMP-\d{4}", question.upper())
    selected_contexts = [context_by_id[i] for i in matched_ids if i in context_by_id]
    retrieved = retrieve_facts(question)

    parts = []
    if retrieved:
        parts.append("与问题直接相关的统计数据（可以直接引用并计算）：")
        parts.extend(f"- {line}" for line in retrieved)
    if selected_contexts:
        parts.append("")
        parts.append("问题中提到的匿名员工详细信息：")
        parts.extend(context_as_facts(c) for c in selected_contexts)
    parts.append("")
    parts.append("其余可用的数据概览：")
    parts.append(snapshot_as_facts(data_snapshot))
    facts = "\n".join(parts)
    return f"""你是企业人力数据分析助手，请用中文回答问题。

{facts}

用户问题：{question}

回答要求：
1. 优先使用上面“与问题直接相关的统计数据”，需要比较或排序时自己算出结果并给出结论。
2. 上面列出的数据都是真实存在的，不要说“数据不足”或“没有该字段”。
3. 只有问题涉及上面完全没有的字段时，才说明“当前数据没有该字段”。
4. 这是本地历史数据，不是实时数据。
5. 风险分数只是筛查信号，不是事实，不能单独证明员工会离职。
6. 不能建议辞退、降薪、处罚、排斥或其他不可逆的人事动作。
7. 涉及员工时只使用匿名编号，并建议人工沟通和核查。
8. 用中文分点作答，先给结论再给依据，不要输出 JSON，不要复述数据列表。
9. 只有上面明确给出的人数才可以使用；离职人数 237 是全体真实员工的总数，不能套用到某个部门或岗位上。"""


def ask(question, show_prompt=False):
    question = str(question).strip()
    if not question:
        print("请输入一个问题。")
        return None
    prompt = build_chat_prompt(question)
    if show_prompt:
        print("发送给本地模型的动态提示词：\n", prompt)
    print(f"\n正在询问本地模型 {OLLAMA_MODEL} …")
    started = time.time()
    try:
        response = ollama_generate(prompt, json_mode=False)
        answer = response.get("response", "").strip()
        print(f"回答完成，用时 {time.time() - started:.1f} 秒：\n")
        print(answer or "模型没有返回文字。")
        return answer
    except Exception as exc:
        print("本地模型调用失败：", exc)
        print(f"请确认已执行 ollama serve，并且已安装：ollama pull {OLLAMA_MODEL}")
        return None


def interactive_chat():
    display_data_snapshot(data_snapshot)
    print("\n自由问答已启动：")
    print("- 直接输入问题，交给本地模型回答")
    print("- 输入 EMP-0123 直接查该员工的原始数据（不经过模型）")
    print("- 输入 summary 重新显示数据概览，输入 q 退出")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n问答结束。")
            break
        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"} or question in {"退出", "结束"}:
            print("问答结束。")
            break
        if question.lower() in {"summary", "data summary"} or question == "数据概览":
            display_data_snapshot(data_snapshot)
            continue
        if re.fullmatch(r"(EMP-)?\d{1,4}", question, flags=re.I):
            lookup_employee(question)
            continue
        ask(question)
'''),
    md(r'''
## 5.1 查询单个员工（42 人或 3000 人任选）

`lookup_employee("EMP-0123")` 按匿名编号从 3000 条数据精确取值；真实 42 人档案用 `lookup_profile("E00012")`。两者都不经过模型，没有抄写误差。

`lookup_by_condition(department="Sales", overtime="Yes", top=5)` 可以按部门、加班情况、最低风险分数先筛人再查看。
'''),
    code(r'''
def lookup_employee(record_id):
    """按匿名编号直接查表，结果来自本地数据，不经过语言模型。"""
    rid = str(record_id).strip().upper()
    if rid.isdigit():
        rid = f"EMP-{int(rid):04d}"
    if not rid.startswith("EMP-"):
        rid = f"EMP-{rid}"
    rows = scored.loc[scored["record_id"].eq(rid)]
    if rows.empty:
        print(f"没有找到 {rid}。可用范围：EMP-0001 到 EMP-{len(scored):04d}")
        return None
    row = rows.iloc[0]
    fields = {
        "匿名编号": rid,
        "部门": value(row, "Department"),
        "岗位": value(row, "JobRole"),
        "风险分数": round(float(row["model_risk_score"]), 4),
        "风险等级": row["model_risk_level"],
        "是否离职(历史标签)": value(row, "Attrition"),
        "加班": value(row, "OverTime"),
        "出差频率": value(row, "BusinessTravel"),
        "工作满意度(1-4)": value(row, "JobSatisfaction"),
        "环境满意度(1-4)": value(row, "EnvironmentSatisfaction"),
        "工作生活平衡(1-4)": value(row, "WorkLifeBalance"),
        "司龄": value(row, "YearsAtCompany"),
        "当前岗位年限": value(row, "YearsInCurrentRole"),
        "距上次晋升年限": value(row, "YearsSinceLastPromotion"),
        "与经理共事年限": value(row, "YearsWithCurrManager"),
        "通勤距离": value(row, "DistanceFromHome"),
        "月收入": value(row, "MonthlyIncome"),
    }
    print(pd.Series(fields, name=rid).to_string())
    print("\n风险分数是模型筛查信号，不是事实，需要人工核实。")
    return fields


def lookup_by_condition(department=None, min_risk=None, overtime=None, top=10):
    """按条件筛选并返回表格，例如 lookup_by_condition(department='Sales', top=5)。"""
    frame = scored
    if department:
        frame = frame[frame["Department"].astype(str).str.contains(department, case=False, na=False)]
    if overtime:
        frame = frame[frame["OverTime"].astype(str).str.lower().eq(str(overtime).lower())]
    if min_risk is not None:
        frame = frame[frame["model_risk_score"].ge(float(min_risk))]
    frame = frame.sort_values("model_risk_score", ascending=False).head(int(top))
    view = frame[["record_id", "Department", "JobRole", "OverTime", "JobSatisfaction", "YearsAtCompany", "model_risk_score", "model_risk_level"]]
    print(f"共筛选出 {len(view)} 条（最多显示 {top} 条）：")
    print(view.to_string(index=False))
    return view


def lookup_profile(employee_id):
    """按 E00001 这样的真实工号查 42 人档案，结果不经过模型。"""
    profiles = load_employee_profiles()
    if profiles is None:
        return None
    eid = str(employee_id).strip().upper()
    if eid.isdigit():
        eid = f"E{int(eid):05d}"
    rows = profiles.loc[profiles["employee_id"].astype(str).str.upper().eq(eid)]
    if rows.empty:
        print(f"没有找到 {eid}。可用范围：E00001 到 E{len(profiles):05d}")
        return None
    row = rows.iloc[0]
    print(row.to_string())
    print("\n以上为原始档案，未经模型转述。")
    return row.to_dict()


PROFILE_FILE = DATA_DIR / "employee_profiles.csv"


def load_employee_profiles(path=PROFILE_FILE):
    """读取 42 人完整档案（含姓名、工位、绩效、风险分等 51 个字段）。"""
    if not Path(path).exists():
        print(f"没有找到 {path}，跳过真实 42 人档案。")
        return None
    profiles = pd.read_csv(path, encoding="utf-8-sig")
    print(f"已读取 {path.name}：{len(profiles)} 人 × {len(profiles.columns)} 个字段")
    return profiles


def profiles_as_lines(profiles, columns=None):
    """把档案压成每员工一行，字段紧凑排列，便于模型逐条照抄。"""
    columns = columns or [
        "employee_id", "name", "floor", "department", "desk_id", "Age", "Gender", "JobRole",
        "JobLevel", "MonthlyIncome", "OverTime", "Attrition", "JobSatisfaction",
        "EnvironmentSatisfaction", "WorkLifeBalance", "YearsAtCompany",
        "risk_score", "risk_level", "overall_score", "potential",
    ]
    lines = []
    for _, row in profiles[columns].iterrows():
        lines.append(
            f"{row['employee_id']} | {row['name']} | {row['floor']}楼 | {row['department']} | "
            f"{row['desk_id']} | {row['Age']}岁 | {row['Gender']} | {row['JobRole']} | "
            f"L{row['JobLevel']} | 月薪{row['MonthlyIncome']} | 加班{row['OverTime']} | "
            f"离职{row['Attrition']} | 满意度{row['JobSatisfaction']} | 环境{row['EnvironmentSatisfaction']} | "
            f"平衡{row['WorkLifeBalance']} | 司龄{row['YearsAtCompany']} | "
            f"风险{round(float(row['risk_score']), 3)} | {row['risk_level']} | "
            f"总分{round(float(row['overall_score']), 1)} | {row['potential']}"
        )
    return lines


def dump_all_employees(profiles, chunk_size=14, show_prompt=False):
    """让本地模型一次性读取全部员工并逐条输出；结果自动与原始数据逐条比对。"""
    if profiles is None or profiles.empty:
        print("没有可用的员工档案。")
        return None
    columns = [
        "employee_id", "name", "floor", "department", "desk_id", "Age", "Gender", "JobRole",
        "JobLevel", "MonthlyIncome", "OverTime", "Attrition", "JobSatisfaction",
        "EnvironmentSatisfaction", "WorkLifeBalance", "YearsAtCompany",
        "risk_score", "risk_level", "overall_score", "potential",
    ]
    field_note = "工号 | 姓名 | 楼层 | 部门 | 工位 | 年龄 | 性别 | 岗位 | 职级 | 月薪 | 加班 | 离职 | 工作满意度 | 环境满意度 | 工作生活平衡 | 司龄 | 风险分 | 风险等级 | 总分 | 潜力"
    total = len(profiles)
    if total <= chunk_size:
        chunks = [profiles]
        offset = 0
    else:
        chunks = [profiles.iloc[i:i + chunk_size] for i in range(0, total, chunk_size)]
        offset = 0

    all_rows = []
    started = time.time()
    for idx, chunk in enumerate(chunks, start=1):
        data_text = "\n".join(profiles_as_lines(chunk, columns))
        prompt = f"""下面是员工完整数据，每条一行，字段依次为：
{field_note}

{data_text}

请把上面每名员工原样整理输出，一条都不能少，按工号升序。每行严格按下面六个位置填写：
第1个位置 = 该员工的真实工号（形如 E00001，不是“工号”这两个字）
第2个位置 = 姓名
第3个位置 = 部门
第4个位置 = 岗位
第5个位置 = 风险分
第6个位置 = 风险等级
六项之间用 " - " 连接。不要在行首加序号，不要输出表头，不要加解释或总结，只输出这些数据行。
正确示例（仅示意格式）：E00009 - 黄梓涵 - 研发部 - 高级工程师 - 0.079 - 低风险"""
        if show_prompt:
            print(prompt)
        print(f"\n第 {idx}/{len(chunks)} 批：{len(chunk)} 人，正在调用 {OLLAMA_MODEL} …")
        try:
            response = ollama_generate(prompt, json_mode=False, max_tokens=4096)
        except Exception as exc:
            print("本地模型调用失败：", exc)
            print(f"请确认已执行 ollama serve，并且已安装：ollama pull {OLLAMA_MODEL}")
            return None
        raw = response.get("response", "")
        # 小模型偶尔把第一行拆成两段，这里逐行抓取，只保留六段完整的数据行
        parsed = re.findall(
            r"(E\d{5})\s*[-－—]\s*([^-\n]+?)\s*[-－—]\s*([^-\n]+?)\s*[-－—]\s*([^-\n]+?)\s*[-－—]\s*([\d.]+)\s*[-－—]\s*(\S+)\s*$",
            raw, flags=re.M)
        done_reason = response.get("done_reason", "")
        print(f"模型返回 {len(parsed)} 条，输出 {response.get('eval_count', '?')} tokens，用时 {time.time() - started:.1f} 秒")
        if done_reason == "length":
            print("注意：输出达到 token 上限被截断。请把 DUMP_CHUNK_SIZE 调小（例如 14）后重跑。")
        all_rows.extend(parsed)

    print(f"\n全部 {len(chunks)} 批完成，共解析 {len(all_rows)} 条，总用时 {time.time() - started:.1f} 秒。")

    # 逐条比对：模型抄袭是否正确
    truth = profiles.set_index("employee_id")
    parsed_ids = [r[0] for r in all_rows]
    missing = [i for i in truth.index if i not in parsed_ids]
    duplicates = sorted({i for i in parsed_ids if parsed_ids.count(i) > 1})
    errors = []
    for eid, name, dept, role, score, level in all_rows:
        if eid not in truth.index:
            errors.append({"工号": eid, "问题": "原始数据里没有该工号"})
            continue
        t = truth.loc[eid]
        if name.strip() != str(t["name"]):
            errors.append({"工号": eid, "问题": f"姓名 {name.strip()} ≠ {t['name']}"})
        if dept.strip() != str(t["department"]):
            errors.append({"工号": eid, "问题": f"部门 {dept.strip()} ≠ {t['department']}"})
        if abs(float(score) - round(float(t["risk_score"]), 3)) > 0.002:
            errors.append({"工号": eid, "问题": f"风险分 {score} ≠ {round(float(t['risk_score']), 3)}"})
        if level.strip() != str(t["risk_level"]):
            errors.append({"工号": eid, "问题": f"风险等级 {level.strip()} ≠ {t['risk_level']}"})

    print(f"缺失 {len(missing)} 人，重复 {len(duplicates)} 人，字段错误 {len(errors)} 处。")
    if missing:
        print("缺失工号：", missing)
    if duplicates:
        print("重复工号：", duplicates)
    for item in errors:
        print(f"  {item['工号']}: {item['问题']}")
    accuracy = (len(all_rows) - len(errors)) / max(len(truth), 1)
    print(f"\n模型照抄准确率约 {accuracy:.1%}。风险分和风险等级请以 lookup_employee() 的查表结果为准。")
    return {"parsed": all_rows, "missing": missing, "duplicates": duplicates, "errors": errors}
'''),
    md(r'''
## 5.2 让模型一次性读出全部 42 人，并自动比对

`data/employee_profiles.csv` 是真实的 42 人完整档案（51 个字段）。压成每员工一行后只有约 3500 tokens，小模型能完整读完。

把 `RUN_FULL_DUMP` 改成 `True` 后运行，模型会逐条吐出全部 42 人，并自动打印缺失、重复和字段错误。

另外，`hr_merged.csv` 的 3000 条模拟数据不适合整体塞给模型：紧凑排列也有约 25 万字，远超小模型可靠处理的范围。
'''),
    code(r'''
RUN_FULL_DUMP = False     # 改成 True 后运行本单元
DUMP_CHUNK_SIZE = 14      # 分批人数；设为 42 就是一次性把全部 42 人塞给模型
employee_profiles = load_employee_profiles()
if RUN_FULL_DUMP and employee_profiles is not None:
    full_dump_result = dump_all_employees(employee_profiles, chunk_size=DUMP_CHUNK_SIZE)
else:
    full_dump_result = None
    print("已跳过全量输出。把 RUN_FULL_DUMP 改成 True 后运行本单元，可让模型逐条输出全部 42 人并自动比对。")
'''),
    md(r'''
## 5.3 结论：模型能不能读完所有员工

实测 `qwen2.5:1.5b`（就是本机现在这个最小模型），同一份 42 人数据两种方式：

- **每批 14 人、分 3 批**：42 人全部输出，无遗漏、无重复、无字段错误，准确率 100%，总用时约 19 秒。这是默认的 `DUMP_CHUNK_SIZE=14`。
- **一次性塞入全部 42 人**（`DUMP_CHUNK_SIZE=42`）：结果不稳定，实测出现过两种失败——只吐出最后 16 人、前面的全部丢掉；或吐全 42 行但把 4 个人的风险等级/部门抄错。同一份数据多次运行结果不同。

输入本身并不超长（42 人约 2050 tokens），失败发生在模型生成长列表时的回溯上。所以结论是：小模型读到长列表末尾会分心，分批发才算可靠。

真正要拿准确数据时，用 `lookup_profile("E00012")`（42 人档案）或 `lookup_employee("EMP-0123")`（3000 条记录）直接查表，完全绕开模型。

3000 条模拟数据不适合整体塞给模型，改用 `retrieve_facts()` 先筛选、再让模型解释。
'''),
    md(r'''
## 5.4 调用本地 Ollama 模型并显示原始过程

这一部分只访问 `127.0.0.1`。如果 Ollama 没启动，会打印清晰的启动命令，不会影响前面的训练结果。
'''),
    code(r'''
def ollama_generate(prompt, host=OLLAMA_HOST, model=OLLAMA_MODEL, timeout=300, json_mode=False, max_tokens=512):
    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }
    if json_mode:
        request_body["format"] = "json"
    body = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_json_response(text):
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0)), "模型输出包含额外文字，已提取 JSON 对象"
            except json.JSONDecodeError as exc:
                return None, f"JSON 解析失败: {exc}"
        return None, "模型没有返回可解析的 JSON 对象"


def normalize_analysis(parsed):
    if not isinstance(parsed, dict):
        return None
    out = {
        "risk_summary": str(parsed.get("risk_summary", "")),
        "evidence": parsed.get("evidence", []),
        "recommended_actions": parsed.get("recommended_actions", []),
        "priority": str(parsed.get("priority", "medium")).lower(),
        "uncertainty": str(parsed.get("uncertainty", "需要人工核实")),
        "human_review_required": True,
    }
    if not isinstance(out["evidence"], list):
        out["evidence"] = [str(out["evidence"])]
    if not isinstance(out["recommended_actions"], list):
        out["recommended_actions"] = [str(out["recommended_actions"])]
    out["evidence"] = [str(x) for x in out["evidence"]]
    out["recommended_actions"] = [str(x) for x in out["recommended_actions"]]
    if out["priority"] not in {"low", "medium", "high"}:
        out["priority"] = "medium"
    return out


def analyze_one(context, show_prompt=False):
    record_id = context["record_id"]
    print("=" * 80)
    print(f"开始分析 {record_id} | 本地模型: {OLLAMA_MODEL}")
    if show_prompt:
        print("提示词:\n", build_prompt(context))
    started = time.time()
    try:
        response = ollama_generate(build_prompt(context), json_mode=True)
        raw = response.get("response", "")
        print(f"HTTP 请求完成，用时 {time.time() - started:.1f} 秒")
        print("模型原始回复:\n", raw)
        parsed, parse_note = parse_json_response(raw)
        normalized = normalize_analysis(parsed)
        if parse_note:
            print("解析提示:", parse_note)
        print("结构化结果:")
        print(json.dumps(normalized, ensure_ascii=False, indent=2) if normalized else "解析失败，保留原始回复")
        return {
            "record_id": record_id,
            "model": OLLAMA_MODEL,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "context": context,
            "raw_response": raw,
            "analysis": normalized,
            "parse_note": parse_note,
            "error": None,
        }
    except Exception as exc:
        print("本地模型调用失败:", exc)
        print(f"请确认 Ollama 已启动，并执行过：ollama pull {OLLAMA_MODEL}")
        return {
            "record_id": record_id,
            "model": OLLAMA_MODEL,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "context": context,
            "raw_response": "",
            "analysis": None,
            "parse_note": None,
            "error": str(exc),
        }


# 不再自动分析固定的第一条记录；需要观察结构化风险解释时，再手动运行：
# single_result = analyze_one(example_context, show_prompt=True)
'''),
    md(r'''
## 6. 可选：顺序分析 Top-N 并保存结果

每名记录单独请求一次，不并发，便于本地小模型运行和观察日志。默认跳过，避免一打开 Notebook 就产生多次模型调用；需要批量结果时，把 `RUN_BATCH` 改为 `True` 后运行本单元。
'''),
    code(r'''
batch_results = []
if RUN_BATCH:
    for context in contexts:
        batch_results.append(analyze_one(context))
else:
    print("已跳过批量调用：RUN_BATCH=False")

jsonl_path = LLM_DIR / "employee_analysis.jsonl"
summary_path = LLM_DIR / "employee_analysis_summary.csv"

if batch_results:
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in batch_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_rows = []
    for item in batch_results:
        analysis = item.get("analysis") or {}
        ctx = item["context"]
        summary_rows.append({
            "record_id": item["record_id"],
            "model": item["model"],
            "risk_score": ctx["risk_score"],
            "risk_level": ctx["risk_level"],
            "department": ctx["department"],
            "job_role": ctx["job_role"],
            "priority": analysis.get("priority", ""),
            "risk_summary": analysis.get("risk_summary", ""),
            "evidence": " | ".join(analysis.get("evidence", [])),
            "recommended_actions": " | ".join(analysis.get("recommended_actions", [])),
            "uncertainty": analysis.get("uncertainty", ""),
            "human_review_required": True,
            "error": item.get("error", ""),
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"JSONL 已保存: {jsonl_path}")
    print(f"汇总 CSV 已保存: {summary_path}")
else:
    print("没有批量结果，不写入文件。")
'''),
    md(r'''
## 7. 启动自由问答并查看边界

运行下面最后一个代码单元后，会再次显示数据概览并进入输入框。你可以连续提问，不需要重新构造提示词。

- 数值预测模型负责给出风险概率和风险等级。
- 本地大模型负责根据当前数据回答问题、解释结构化信号和提出可逆的关注建议。
- 大模型输出必须经过人工审核，涉及员工时只能使用匿名编号。
- 不能根据单次模型分数直接辞退、降薪、处罚或给员工贴标签。
- 本地数据是历史数据，不是实时数据；模型不能凭空读取今天的新数据。

推荐提问示例：

- `哪个部门的历史离职率最高？`
- `加班和离职率有什么关系？`
- `EMP-0001 的风险信号是什么？`
- `高风险岗位有哪些？`
- `模型的测试 F1 和 AUC 怎么理解？`

也可以在输入框直接输入 `EMP-0123`，不经过模型，直接打印该员工的原始数据。
'''),
    code(r'''
if SHOW_INTERACTIVE_CHAT:
    interactive_chat()
'''),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata.language_info = {"name": "python", "version": "3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"已生成: {OUT}")
print(f"cell 数量: {len(cells)}")
