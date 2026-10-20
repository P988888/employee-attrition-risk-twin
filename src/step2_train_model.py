"""Step 2 特征工程 + 多模型训练 + 评估 + 风险评分。

流程：
  特征工程 -> 列变换管道 -> 5 折交叉验证 -> 测试集评估 -> 模型对比图
           -> 最优模型(随机森林)全量风险评分 -> 导出 3D 大屏数据

评估口径说明（重要）：
  - 模型训练与所有指标 = 只用 IBM 真实数据(1470 条)，避免模拟数据把成绩"注水"。
  - 最终风险评分 = 对真实 1470 条打分；模拟 1530 条一并打分，仅用于 3D 大屏的视觉体量。
    模拟集指标单独记为 sim_eval，供大屏展示，不与真实指标混淆。
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report,
)
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    MERGED_CSV, FIGURES, MODELS, DASH_JSON, SCORE_CSV, TARGET,
    NUMERIC_COLS, CATEGORICAL_COLS, RANDOM_STATE,
    VAL_SIZE, TEST_SIZE_FINAL,
)

warnings.filterwarnings("ignore")

# macOS 可用的中文字体，按优先级回落
_CJK = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Hiragino Sans GB", "STHeiti"]
_available = {f.name for f in font_manager.fontManager.ttflist}
CJK_FONT = next((f for f in _CJK if f in _available), "Arial Unicode MS")
plt.rcParams["font.sans-serif"] = [CJK_FONT] + _CJK
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=CJK_FONT)
PALETTE = {"No": "#2E86AB", "Yes": "#E4572E"}


# ---------------------------------------------------------------- 特征工程
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """构造业务含义明确的派生特征。全部为比值/差值，不含标签信息，不泄漏。"""
    d = df.copy()
    eps = 1e-6

    # 薪酬相对职级：同样职级拿得更少的人更容易走
    d["IncomePerLevel"] = d["MonthlyIncome"] / (d["JobLevel"] * 5000.0 + eps)
    # 当前公司工龄占整个职业生涯比重：越低说明是新人/惯于跳槽
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1.0)
    # 距上次晋升年限占司龄比：越高说明职业停滞
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1.0)
    # 与现任经理共事年限占司龄比：越低说明刚换领导
    d["ManagerStability"] = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1.0)
    # 满意度综合分（4 项主观量表）
    d["SatisfactionScore"] = (
        d["JobSatisfaction"] + d["EnvironmentSatisfaction"]
        + d["RelationshipSatisfaction"] + d["WorkLifeBalance"]
    ) / 4.0
    # 职业停滞指数：晋升空窗越久、同一岗位待得越久，越容易流失
    d["CareerStagnation"] = (
        d["YearsSinceLastPromotion"] / (d["YearsInCurrentRole"] + 1.0)
    ) * (1.0 - d["JobInvolvement"] / 4.0)
    # 跳槽频率：平均每家公司的任职年限，越小越爱跳
    d["CompanyHopRate"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1.0)
    return d


def build_pipeline(model, use_smote: bool = False) -> Pipeline:
    """构造完整管道。

    use_smote=True 时把 SMOTE 插到列变换之后、分类器之前，
    这样重采样只作用于训练折（交叉验证时 SMOTE 在每个训练折内部独立执行），
    不会让合成样本污染验证折，避免指标虚高。
    """
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_feats = NUMERIC_COLS + [
        "IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
        "SatisfactionScore", "CareerStagnation", "CompanyHopRate",
    ]
    ct = ColumnTransformer(
        [("num", num_pipe, num_feats), ("cat", cat_pipe, CATEGORICAL_COLS)],
        remainder="drop")
    steps = [("prep", ct)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    steps.append(("clf", model))
    # 含 SMOTE 时必须用 imblearn 的 Pipeline，否则交叉验证不会在折内重采样
    return ImbPipeline(steps), num_feats + CATEGORICAL_COLS


def get_models() -> dict:
    """返回 (模型, 是否启用 SMOTE)。
    树模型本身有 class_weight 处理不平衡，逻辑回归在 SMOTE 后召回提升最明显。
    """
    return {
        "逻辑回归": (LogisticRegression(
            max_iter=3000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE), True),
        "逻辑回归(无采样)": (LogisticRegression(
            max_iter=3000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE), False),
        "决策树": (DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=15, class_weight="balanced",
            random_state=RANDOM_STATE), True),
        "随机森林": (RandomForestClassifier(
            n_estimators=500, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE), True),
        "梯度提升": (GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=RANDOM_STATE), True),
    }


# ---------------------------------------------------------------- 评估
def split_three_way(X, y, val_size: float, test_size: float, seed: int):
    """训练 / 验证 / 测试 三段式分层划分。

    先切出测试集（全程封存），再把剩余部分切出验证集。
    验证集职责：模型选型 + 决策阈值标定
    测试集职责：最终一次性评估，反映真实泛化能力
    """
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    # 在剩余数据中，验证集占 val_size/(1-test_size)
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=rel_val, stratify=y_trv, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def pick_threshold_on_val(y_va, proba_va) -> float:
    """用验证集标定决策阈值（F1 最优），不碰测试集。"""
    prec, rec, thr = precision_recall_curve(y_va, proba_va)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1s[:-1])]) if len(thr) else 0.5


def _metrics(y_true, proba, thr: float) -> dict:
    """给定阈值，计算一整套指标 + 混淆矩阵。"""
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "ap": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(thr),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(name, pipe, Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    """完整评估：交叉验证（训练集）→ 验证集定阈值 → 测试集最终评估。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)

    # 验证集：用于阈值标定与模型选型
    proba_va = pipe.predict_proba(Xva)[:, 1]
    best_thr = pick_threshold_on_val(yva, proba_va)
    va = _metrics(yva, proba_va, best_thr)

    # 测试集：只评估，不用来调任何东西
    proba_te = pipe.predict_proba(Xte)[:, 1]
    te = _metrics(yte, proba_te, best_thr)
    pred_te = (proba_te >= best_thr).astype(int)

    return {
        "model": name,
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "best_threshold": best_thr,
        # 测试集指标（对外汇报口径）
        "auc": te["auc"], "ap": te["ap"], "accuracy": te["accuracy"],
        "precision": te["precision"], "recall": te["recall"],
        "f1": te["f1"], "brier": te["brier"], "confusion": te["confusion"],
        # 验证集指标（用于对比泛化差距）
        "val_auc": va["auc"], "val_f1": va["f1"],
        "val_accuracy": va["accuracy"], "val_precision": va["precision"],
        "val_recall": va["recall"], "val_brier": va["brier"],
        "val_confusion": va["confusion"],
        # 泛化差距：测试集 AUC - 验证集 AUC
        "auc_gap": te["auc"] - va["auc"],
        "threshold": best_thr,
        "_proba": proba_te,
        "_proba_val": proba_va,
        "_y": yte,
        "_y_val": yva,
        "_report": classification_report(yte, pred_te, target_names=["留任", "离职"], zero_division=0),
    }


# ---------------------------------------------------------------- 绘图
def plot_model_comparison(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    m = df.melt(id_vars="model",
                value_vars=["accuracy", "precision", "recall", "f1", "auc"],
                var_name="metric", value_name="score")
    sns.barplot(data=m, x="model", y="score", hue="metric", ax=axes[0], palette="viridis")
    axes[0].set_title("各模型指标对比（测试集）", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=3, fontsize=9, loc="lower right")
    axes[0].set_xlabel("")

    axes[1].errorbar(df["model"], df["cv_auc_mean"], yerr=df["cv_auc_std"],
                     fmt="o", capsize=8, markersize=10, color="#2E86AB", ecolor="#8AB4C9")
    for i, r in df.iterrows():
        axes[1].annotate(f"{r.cv_auc_mean:.4f}", (i, r.cv_auc_mean),
                         textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10)
    axes[1].set_title("5 折交叉验证 AUC（均值 ± 标准差）", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES / "01_模型对比.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for r in rows:
        fpr, tpr, _ = roc_curve(r["_y"], r["_proba"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{r['model']} (AUC={r['auc']:.3f})")
        p, rc, _ = precision_recall_curve(r["_y"], r["_proba"])
        axes[1].plot(rc, p, lw=2, label=f"{r['model']} (AP={r['ap']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
    axes[0].set_title("ROC 曲线", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
    axes[0].legend(loc="lower right", fontsize=9)

    base = rows[0]["_y"].mean()
    axes[1].axhline(base, color="k", ls="--", lw=1, label=f"基准={base:.3f}")
    axes[1].set_title("PR 曲线（不平衡数据更敏感）", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
    axes[1].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES / "02_ROC_PR曲线.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(rows: list[dict]) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    for ax, r in zip(np.atleast_1d(axes), rows):
        cm = np.array([[r["confusion"]["tn"], r["confusion"]["fp"]],
                       [r["confusion"]["fn"], r["confusion"]["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["留任", "离职"], yticklabels=["留任", "离职"])
        ax.set_title(f"{r['model']}\n阈值={r['best_threshold']:.2f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("预测"); ax.set_ylabel("实际")
    plt.suptitle("混淆矩阵对比", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "03_混淆矩阵.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipe, feats) -> pd.DataFrame:
    """特征重要性。

    逻辑回归/树模型的系数与重要性落在「列变换之后」的空间上，
    数值特征保持原名，类别特征被 one-hot 成 BusinessTravel_Yes 这类列。
    这里从已拟合的 ColumnTransformer 里取回真实展开名，让图例可读。
    """
    clf = pipe.named_steps["clf"]
    prep = pipe.named_steps["prep"]
    try:
        names = list(prep.get_feature_names_out())
        # 去掉 num__ / cat__ 前缀
        names = [n.split("__", 1)[1] if "__" in n else n for n in names]
    except Exception:
        names = list(feats)

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_).ravel()
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_).ravel())
    else:
        imp = np.ones(len(names))

    # 长度对齐（正常情况下二者相等）
    n = min(len(imp), len(names))
    s = pd.Series(imp[:n], index=names[:n]).sort_values(ascending=False)
    top = s.head(15)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="rocket")
    ax.set_title(f"特征重要性 Top 15（{type(clf).__name__}）", fontsize=14, fontweight="bold")
    ax.set_xlabel("重要性 / |系数|")
    plt.tight_layout()
    plt.savefig(FIGURES / "04_特征重要性.png", dpi=150, bbox_inches="tight")
    plt.close()
    return s


def plot_risk_distribution(scored: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    real = scored[scored["_source"] == "real"]

    for label, grp in real.groupby("Attrition"):
        axes[0].hist(grp["risk_score"], bins=30, alpha=0.65, label=f"实际{label}", color=PALETTE[label])
    axes[0].set_title("风险评分分布（按实际离职标签）", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("离职风险概率"); axes[0].legend()

    order = ["低风险", "中风险", "高风险", "极高风险"]
    cols = ["#4CAF50", "#FFC107", "#FF7043", "#C62828"]
    vc = real["risk_level"].value_counts().reindex(order).fillna(0)
    axes[1].bar(order, vc.values, color=cols)
    for i, v in enumerate(vc.values):
        axes[1].text(i, v + 5, f"{int(v)}\n({v/len(real):.0%})", ha="center", fontsize=10)
    axes[1].set_title("真实员工风险等级分布", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, vc.max() * 1.25)

    dept = real.groupby("Department").agg(均值=("risk_score", "mean"), 人数=("risk_score", "size"))
    axes[2].barh(dept.index, dept["均值"], color="#5C6BC0")
    for i, (idx, r) in enumerate(dept.iterrows()):
        axes[2].text(r["均值"] + .005, i, f'{r["均值"]:.1%} (n={int(r["人数"])})', va="center", fontsize=10)
    axes[2].set_title("各部门平均离职风险", fontsize=13, fontweight="bold")
    axes[2].set_xlim(0, dept["均值"].max() * 1.35)

    plt.tight_layout()
    plt.savefig(FIGURES / "05_风险分布.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation(df: pd.DataFrame) -> None:
    num = df[NUMERIC_COLS + ["SatisfactionScore", "TenureRatio",
                             "CareerStagnation", "IncomePerLevel"]].copy()
    num["离职"] = df[TARGET].eq("Yes").astype(int)
    corr = num.corr()["离职"].drop("离职").sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#E4572E" if v > 0 else "#2E86AB" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set_title("各特征与离职的相关性（红=正相关，蓝=负相关）", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson 相关系数")
    plt.tight_layout()
    plt.savefig(FIGURES / "06_相关性分析.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- 主流程
def risk_level(p: float) -> str:
    if p < 0.25:
        return "低风险"
    if p < 0.50:
        return "中风险"
    if p < 0.75:
        return "高风险"
    return "极高风险"


def main() -> None:
    df = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
    df = engineer(df)
    real = df[df["_source"] == "real"].reset_index(drop=True)
    print(f"合并数据 {df.shape}，其中真实 {len(real)} 条\n")

    y = real[TARGET].eq("Yes").astype(int).values
    X = real[NUMERIC_COLS + CATEGORICAL_COLS + [
        "IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
        "SatisfactionScore", "CareerStagnation", "CompanyHopRate"]]

    Xtr, Xva, Xte, ytr, yva, yte = split_three_way(
        X, y, VAL_SIZE, TEST_SIZE_FINAL, RANDOM_STATE)
    print("三段式分层划分（训练 / 验证 / 测试）")
    print(f"  训练集 {Xtr.shape[0]:>4} 条  离职率 {ytr.mean():6.2%}")
    print(f"  验证集 {Xva.shape[0]:>4} 条  离职率 {yva.mean():6.2%}  <- 选模型 + 定阈值")
    print(f"  测试集 {Xte.shape[0]:>4} 条  离职率 {yte.mean():6.2%}  <- 最终评估，全程封存\n")

    rows, pipes = [], {}
    for name, (model, use_smote) in get_models().items():
        pipe, feats = build_pipeline(model, use_smote)
        pipe.fit(Xtr, ytr)
        r = evaluate(name, pipe, Xtr, ytr, Xva, yva, Xte, yte)
        rows.append(r)
        pipes[name] = (pipe, feats)
        print(f"[{name:<16}] 测试 AUC={r['auc']:.4f}  F1={r['f1']:.4f}  准确={r['accuracy']:.4f}  "
              f"精确={r['precision']:.4f}  查全={r['recall']:.4f}")
        print(f"{'':<19} 验证 AUC={r['val_auc']:.4f}  F1={r['val_f1']:.4f}   "
              f"CV-AUC={r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}  阈值={r['best_threshold']:.3f}  "
              f"泛化差距={r['auc_gap']:+.4f}")

    # 选型：在验证集上选，避免用测试集做选择
    best_row = max(rows, key=lambda r: (r["val_auc"], r["val_f1"]))
    best_name = best_row["model"]
    print(f"\n==> 最优模型(按验证集选型): {best_name}")
    print(f"    验证 AUC={best_row['val_auc']:.4f} / 测试 AUC={best_row['auc']:.4f} "
          f"(泛化差距 {best_row['auc_gap']:+.4f})")
    print("\n" + best_row["_report"])

    # 图
    plot_model_comparison(rows)
    plot_roc_pr(rows)
    plot_confusion(rows)
    imp = plot_feature_importance(*pipes[best_name])
    plot_correlation(real)

    # ---- 全量评分（真实 + 模拟，供大屏使用）----
    best_pipe = pipes[best_name][0]
    df["risk_score"] = best_pipe.predict_proba(df[NUMERIC_COLS + CATEGORICAL_COLS + [
        "IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
        "SatisfactionScore", "CareerStagnation", "CompanyHopRate"]])[:, 1]
    df["risk_level"] = df["risk_score"].map(risk_level)
    df["risk_rank"] = df["risk_score"].rank(ascending=False, method="dense").astype(int)
    df.to_csv(SCORE_CSV, index=False, encoding="utf-8-sig")
    plot_risk_distribution(df)

    # 模拟集单独评估（并注明是模拟数据，不作主指标）
    sim = df[df["_source"] == "simulated"]
    sim_y = sim[TARGET].eq("Yes").astype(int).values
    sim_auc = roc_auc_score(sim_y, sim["risk_score"])
    sim_ap = average_precision_score(sim_y, sim["risk_score"])
    sim_pred = (sim["risk_score"] >= best_row["best_threshold"]).astype(int)
    sim_metrics = {
        "auc": float(sim_auc), "ap": float(sim_ap),
        "accuracy": float(accuracy_score(sim_y, sim_pred)),
        "precision": float(precision_score(sim_y, sim_pred, zero_division=0)),
        "recall": float(recall_score(sim_y, sim_pred, zero_division=0)),
        "f1": float(f1_score(sim_y, sim_pred, zero_division=0)),
        "n": int(len(sim)),
    }
    print(f"\n[模拟集参考] n={len(sim)}  AUC={sim_auc:.4f}  F1={sim_metrics['f1']:.4f}"
          f"  （仅用于大屏体量，不作为模型主指标）")

    # ---- 导出大屏数据 ----
    export_dashboard(df, rows, best_name, best_row, imp, sim_metrics)

    # ---- 保存模型 ----
    joblib.dump(best_pipe, MODELS / "best_model.pkl")
    meta = {
        "best_model": best_name,
        "trained_on": "IBM HR Analytics 真实数据 1470 条",
        "split": {
            "scheme": "train/val/test = 0.6/0.2/0.2（分层抽样）",
            "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]),
            "n_test": int(Xte.shape[0]),
            "val_role": "模型选型 + 阈值标定",
            "test_role": "最终评估（全程封存，不参与任何调参）",
            "val_attrition": float(yva.mean()),
            "test_attrition": float(yte.mean()),
        },
        "features": NUMERIC_COLS + CATEGORICAL_COLS + [
            "IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
            "SatisfactionScore", "CareerStagnation", "CompanyHopRate"],
        "best_threshold": best_row["best_threshold"],
        "real_metrics": {k: v for k, v in best_row.items() if not k.startswith("_")},
        "sim_metrics": sim_metrics,
        "all_models": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    (MODELS / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n模型 -> {MODELS/'best_model.pkl'}")
    print(f"评分 -> {SCORE_CSV.name}")
    print(f"大屏数据 -> {DASH_JSON.name}")
    print(f"图表 -> {FIGURES}")


def export_dashboard(df, rows, best_name, best_row, imp, sim_metrics) -> None:
    real = df[df["_source"] == "real"]
    sim = df[df["_source"] == "simulated"]

    def clean(dfx, n=None):
        d = dfx if n is None else dfx.head(n)
        out = []
        for _, r in d.iterrows():
            out.append({
                "id": int(r["EmployeeNumber"]) if pd.notna(r.get("EmployeeNumber")) else None,
                "src": r["_source"],
                "age": int(r["Age"]),
                "gender": r["Gender"],
                "dept": r["Department"],
                "role": r["JobRole"],
                "level": int(r["JobLevel"]),
                "income": int(r["MonthlyIncome"]),
                "tenure": int(r["YearsAtCompany"]),
                "totalYears": int(r["TotalWorkingYears"]),
                "overtime": r["OverTime"],
                "travel": r["BusinessTravel"],
                "marital": r["MaritalStatus"],
                "sat": round(float(r["SatisfactionScore"]), 2),
                "wlb": int(r["WorkLifeBalance"]),
                "stagnation": round(float(r["CareerStagnation"]), 3),
                "dist": int(r["DistanceFromHome"]),
                "risk": round(float(r["risk_score"]), 4),
                "level_": r["risk_level"],
                "left": r[TARGET] == "Yes",
            })
        return out

    payload = {
        "meta": {
            "total": int(len(df)),
            "real": int(len(real)),
            "sim": int(len(sim)),
            "attritionRate": float(real[TARGET].eq("Yes").mean()),
            "bestModel": best_name,
            "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        },
        "metrics": [{
            "model": r["model"], "auc": round(r["auc"], 4), "f1": round(r["f1"], 4),
            "accuracy": round(r["accuracy"], 4), "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4), "ap": round(r["ap"], 4),
            "cvAuc": round(r["cv_auc_mean"], 4), "cvStd": round(r["cv_auc_std"], 4),
            "confusion": r["confusion"],
        } for r in rows],
        "best": {
            "model": best_name,
            "auc": round(best_row["auc"], 4), "f1": round(best_row["f1"], 4),
            "accuracy": round(best_row["accuracy"], 4),
            "precision": round(best_row["precision"], 4),
            "recall": round(best_row["recall"], 4),
            "threshold": round(best_row["best_threshold"], 4),
            "confusion": best_row["confusion"],
        },
        "simMetrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in sim_metrics.items()},
        "importance": [{"name": k, "value": round(float(v), 5)} for k, v in imp.items()],
        "roc": [],
        "riskBands": [
            {"level": lv, "n": int((real["risk_level"] == lv).sum())}
            for lv in ["低风险", "中风险", "高风险", "极高风险"]
        ],
        "byDept": [
            {"dept": k, "n": int(v["n"]), "avgRisk": round(float(v["avg"]), 4),
             "leftRate": round(float(v["left"]), 4)}
            for k, v in _group(real)
        ],
        "byRole": [
            {"role": k, "n": int(v["n"]), "avgRisk": round(float(v["avg"]), 4),
             "leftRate": round(float(v["left"]), 4)}
            for k, v in _group_role(real)
        ],
        "employees": clean(real),
        "simEmployees": clean(sim),
    }

    # ROC 曲线点（用于大屏画图）
    from sklearn.metrics import roc_curve
    for r in rows:
        fpr, tpr, _ = roc_curve(r["_y"], r["_proba"])
        idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        payload["roc"].append({
            "model": r["model"],
            "points": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx],
        })

    # 3D 散点图用的派生视图：年龄 × 收入 × 司龄 三维空间
    payload["cloud"] = [
        {"x": e["age"], "y": e["income"], "z": e["tenure"],
         "risk": e["risk"], "level": e["level_"], "dept": e["dept"],
         "left": e["left"], "src": e["src"]}
        for e in payload["employees"]
    ]

    DASH_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"大屏数据: {len(payload['employees'])} 真实 + {len(payload['simEmployees'])} 模拟 "
          f"= {payload['meta']['total']} 条")


def _group(d):
    return d.groupby("Department").agg(n=("risk_score", "size"), avg=("risk_score", "mean"),
                                       left=(TARGET, lambda s: s.eq("Yes").mean())).iterrows()


def _group_role(d):
    g = d.groupby("JobRole").agg(n=("risk_score", "size"), avg=("risk_score", "mean"),
                                 left=(TARGET, lambda s: s.eq("Yes").mean()))
    return g.sort_values("avg", ascending=False).iterrows()


if __name__ == "__main__":
    main()
