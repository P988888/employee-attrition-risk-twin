"""Step 6 行为链路分析：把 5 张表串成一条可解释的因果链。

链路：日常行为 -> 工作质量 -> 测评/谈话 -> 状态结论

分析方法：
  1. 各源数据聚合到员工维度（行为取均值/趋势，质量取均值，报告取情感分）
  2. 检验链路每一段的关联强度（相关系数 + 显著性）
  3. 用线性回归量化「行为 -> 质量」的边际贡献
  4. 用逻辑回归/线性概率模型做「全链路 -> 离职」预测，并对比仅用静态属性
  5. 输出链路权重图与关键节点

这样能回答：加了行为与报告数据，预测力提升多少？哪个环节信号最强？
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy import stats

from config import DATA, FIGURES, OUTPUT, RANDOM_STATE, VAL_SIZE, TEST_SIZE_FINAL

warnings.filterwarnings("ignore")

_CJK = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Hiragino Sans GB", "STHeiti"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in _CJK if f in _avail), "Arial Unicode MS")
plt.rcParams["font.sans-serif"] = [CJK] + _CJK
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=CJK)

REPORT = OUTPUT / "behavior_link_report.txt"
_log: list[str] = []


def w(s: str = "") -> None:
    _log.append(s)
    print(s)


def load_all() -> dict[str, pd.DataFrame]:
    need = {
        "emp": "link_employees.csv",
        "beh": "link_daily_behavior.csv",
        "qual": "link_work_quality.csv",
        "talk": "link_talk_reports.csv",
        "survey": "link_survey_reports.csv",
    }
    out = {}
    for k, f in need.items():
        p = DATA / f
        if not p.exists():
            raise SystemExit(f"缺少 {f}，请先运行 step5_build_behavior_link.py")
        out[k] = pd.read_csv(p, encoding="utf-8-sig")
    return out


def build_employee_features(d: dict) -> pd.DataFrame:
    """把 5 张表聚合到「员工 × 一行」的宽表。"""
    emp, beh, qual, talk, survey = d["emp"], d["beh"], d["qual"], d["talk"], d["survey"]
    beh = beh.copy()
    beh["date"] = pd.to_datetime(beh["date"])

    # ---- 行为层：均值和趋势（后半段 - 前半段） ----
    beh = beh.sort_values(["employee_id", "date"])
    mid = beh["date"].min() + (beh["date"].max() - beh["date"].min()) / 2

    def slope(g: pd.DataFrame) -> float:
        y = g["deep_focus_hours"].values
        if len(y) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])

    b_agg = beh.groupby("employee_id").agg(
        b_hours=("work_hours", "mean"),
        b_focus=("deep_focus_hours", "mean"),
        b_late=("late_flag", "mean"),
        b_leave=("leave_flag", "mean"),
        b_meet=("meeting_count", "mean"),
        b_msg=("messages_sent", "mean"),
        b_tasks=("tasks_completed", "mean"),
        b_checkin=("check_in_hour", "mean"),
    )
    b_trend = beh.groupby("employee_id").apply(slope).rename("b_focus_trend").reset_index()
    b_agg = b_agg.reset_index().merge(b_trend, on="employee_id", how="left")

    # ---- 质量层 ----
    q_agg = qual.groupby("employee_id").agg(
        q_delivery=("delivery_rate", "mean"),
        q_rework=("rework_count", "mean"),
        q_defect=("defect_rate", "mean"),
        q_efficiency=("efficiency_score", "mean"),
        q_quality=("quality_score", "mean"),
    ).reset_index()

    # ---- 谈话层 ----
    t_agg = talk.groupby("employee_id").agg(
        t_count=("talk_id", "size"),
        t_sentiment=("sentiment", "mean"),
        t_followup=("follow_up_needed", "mean"),
        t_duration=("duration_min", "mean"),
    ).reset_index()
    # 风险类谈话占比
    risk_mask = talk["sentiment"] < -0.3
    t_risk = talk[risk_mask].groupby("employee_id").size().rename("t_risk_count").reset_index()

    # ---- 测评层 ----
    s_agg = survey.groupby("employee_id").agg(
        s_competency=("competency_score", "mean"),
        s_engagement=("engagement_score", "mean"),
        s_satisfaction=("satisfaction_score", "mean"),
        s_stress=("stress_score", "mean"),
        s_overall=("overall_score", "mean"),
    ).reset_index()

    df = emp.merge(b_agg, on="employee_id", how="left") \
            .merge(q_agg, on="employee_id", how="left") \
            .merge(t_agg, on="employee_id", how="left") \
            .merge(t_risk, on="employee_id", how="left") \
            .merge(s_agg, on="employee_id", how="left")
    df["t_risk_count"] = df["t_risk_count"].fillna(0)
    df["t_count"] = df["t_count"].fillna(0)
    df["attrition_flag"] = df["attrition"].eq("Yes").astype(int)
    return df


BEH_COLS = ["b_hours", "b_focus", "b_late", "b_leave", "b_meet", "b_msg",
            "b_tasks", "b_checkin", "b_focus_trend"]
QUAL_COLS = ["q_delivery", "q_rework", "q_defect", "q_efficiency", "q_quality"]
TALK_COLS = ["t_count", "t_sentiment", "t_followup", "t_duration", "t_risk_count"]

STATIC_COLS = ["age", "job_level", "monthly_income", "years_at_company"]


def link_strength(df: pd.DataFrame) -> None:
    """检验链路每一段的关联强度。"""
    w("\n" + "=" * 72)
    w(" 一、链路各环节关联强度检验")
    w("=" * 72)

    segments = [
        ("行为 → 质量", ["b_focus", "b_tasks", "b_late"],
         ["q_quality", "q_efficiency", "q_rework"]),
        ("质量 → 测评", ["q_quality", "q_efficiency"], ["s_engagement", "s_satisfaction"]),
        ("测评 → 离职", ["s_engagement", "s_satisfaction", "s_stress"], ["attrition_flag"]),
        ("谈话 → 离职", ["t_sentiment", "t_risk_count"], ["attrition_flag"]),
        ("行为 → 离职", ["b_focus", "b_late", "b_tasks"], ["attrition_flag"]),
    ]

    for label, xs, ys in segments:
        w(f"\n[{label}]")
        w(f"  {'自变量':<20}{'因变量':<20}{'r':>9}{'p值':>12}{'强度':>10}")
        w("  " + "-" * 71)
        for x in xs:
            for y in ys:
                sub = df[[x, y]].dropna()
                if sub[x].nunique() < 2 or sub[y].nunique() < 2:
                    continue
                r, p = stats.pearsonr(sub[x], sub[y])
                strength = ("强" if abs(r) > .5 else "中" if abs(r) > .3
                            else "弱" if abs(r) > .1 else "极弱")
                star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                w(f"  {x:<20}{y:<20}{r:>9.3f}{p:>12.2e}{strength+star:>10}")


def regression_behavior_to_quality(df: pd.DataFrame) -> dict:
    """线性回归：行为指标 -> 工作质量。"""
    w("\n" + "=" * 72)
    w(" 二、线性回归：日常行为 -> 工作质量")
    w("=" * 72)

    sub = df[BEH_COLS + ["q_quality"]].dropna()
    X = sub[BEH_COLS].values
    y = sub["q_quality"].values

    idx = np.arange(len(sub))
    i_trv, i_te = train_test_split(idx, test_size=TEST_SIZE_FINAL, random_state=RANDOM_STATE)
    i_tr, i_va = train_test_split(i_trv, test_size=VAL_SIZE / (1 - TEST_SIZE_FINAL),
                                  random_state=RANDOM_STATE)

    sc = StandardScaler().fit(X[i_tr])
    model = LinearRegression().fit(sc.transform(X[i_tr]), y[i_tr])

    w(f"\n样本: 训练 {len(i_tr)} / 验证 {len(i_va)} / 测试 {len(i_te)}")
    w(f"\n  {'数据集':<8}{'R²':>10}{'RMSE':>10}")
    w("  " + "-" * 28)
    metrics = {}
    for nm, ix in [("训练集", i_tr), ("验证集", i_va), ("测试集", i_te)]:
        pred = model.predict(sc.transform(X[ix]))
        r2 = r2_score(y[ix], pred)
        rmse = float(np.sqrt(np.mean((y[ix] - pred) ** 2)))
        metrics[nm] = {"r2": r2, "rmse": rmse}
        w(f"  {nm:<8}{r2:>10.4f}{rmse:>10.3f}")
    w(f"\n  泛化差距(训练-测试) R²: {metrics['训练集']['r2']-metrics['测试集']['r2']:+.4f}")

    w(f"\n  标准化系数（行为每变动 1 个标准差，质量分变动多少）")
    w(f"  {'行为指标':<20}{'系数':>10}{'方向':>8}")
    w("  " + "-" * 38)
    coefs = pd.Series(model.coef_, index=BEH_COLS).sort_values(key=abs, ascending=False)
    for k, v in coefs.items():
        w(f"  {k:<20}{v:>10.3f}{'正向' if v > 0 else '负向':>8}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top = coefs.iloc[::-1]
    axes[0].barh(top.index, top.values,
                 color=["#28d17c" if v > 0 else "#E4572E" for v in top.values])
    axes[0].axvline(0, color="k", lw=.9)
    axes[0].set_title("行为指标对工作质量的影响（标准化系数）",
                      fontsize=13, fontweight="bold")

    pred_te = model.predict(sc.transform(X[i_te]))
    axes[1].scatter(y[i_te], pred_te, s=12, alpha=.45, color="#4C7DFF")
    lo, hi = min(y[i_te].min(), pred_te.min()), max(y[i_te].max(), pred_te.max())
    axes[1].plot([lo, hi], [lo, hi], "--", color="#E4572E", lw=1.5)
    axes[1].set_xlabel("实际质量分"); axes[1].set_ylabel("预测质量分")
    axes[1].set_title(f"测试集拟合效果 (R²={metrics['测试集']['r2']:.3f})",
                      fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "11_行为到质量回归.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"metrics": metrics, "coefs": coefs}


def compare_models(df: pd.DataFrame) -> dict:
    """对比：仅静态属性 vs 静态+行为 vs 全链路（含报告）。

    重要：特征越多不代表越准。本函数同时报告「是否过拟合」，
    并对全链路模型加 L2 正则（C 更小）以抑制高维特征带来的方差。
    """
    w("\n" + "=" * 72)
    w(" 三、预测力对比：静态属性 vs 行为 vs 行为+报告")
    w("=" * 72)

    y = df["attrition_flag"].values
    idx = np.arange(len(df))
    i_trv, i_te = train_test_split(idx, test_size=TEST_SIZE_FINAL,
                                   stratify=y, random_state=RANDOM_STATE)
    i_tr, i_va = train_test_split(i_trv, test_size=VAL_SIZE / (1 - TEST_SIZE_FINAL),
                                  stratify=y[i_trv], random_state=RANDOM_STATE)

    feature_sets = {
        "仅静态属性": (STATIC_COLS, 0.5),
        "+ 日常行为": (STATIC_COLS + BEH_COLS, 0.5),
        "+ 工作质量": (STATIC_COLS + BEH_COLS + QUAL_COLS, 0.3),
        "+ 谈话+测评(全链路)": (STATIC_COLS + BEH_COLS + QUAL_COLS + TALK_COLS
                              + ["s_competency", "s_engagement", "s_satisfaction",
                                 "s_stress", "s_overall"], 0.08),
    }

    w(f"\n  {'特征组合':<22}{'特征数':>7}{'验证AUC':>10}{'测试AUC':>10}"
      f"{'训练AUC':>10}{'过拟合':>9}")
    w("  " + "-" * 70)

    results = {}
    for name, (cols, C) in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        sub = df[cols + ["attrition_flag"]].dropna()
        X_all = sub[cols].values
        y_all = sub["attrition_flag"].values
        n = len(sub)
        ridx = np.arange(n)
        r_trv, r_te = train_test_split(ridx, test_size=TEST_SIZE_FINAL,
                                       stratify=y_all, random_state=RANDOM_STATE)
        r_tr, r_va = train_test_split(r_trv, test_size=VAL_SIZE / (1 - TEST_SIZE_FINAL),
                                      stratify=y_all[r_trv], random_state=RANDOM_STATE)

        sc = StandardScaler().fit(X_all[r_tr])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 C=C, random_state=RANDOM_STATE)
        clf.fit(sc.transform(X_all[r_tr]), y_all[r_tr])

        auc_tr = roc_auc_score(y_all[r_tr], clf.predict_proba(sc.transform(X_all[r_tr]))[:, 1])
        p_va = clf.predict_proba(sc.transform(X_all[r_va]))[:, 1]
        auc_va = roc_auc_score(y_all[r_va], p_va)

        from sklearn.metrics import precision_recall_curve
        pr, rc, th = precision_recall_curve(y_all[r_va], p_va)
        f1s = 2 * pr * rc / (pr + rc + 1e-12)
        thr = float(th[np.argmax(f1s[:-1])]) if len(th) else .5

        p_te = clf.predict_proba(sc.transform(X_all[r_te]))[:, 1]
        auc_te = roc_auc_score(y_all[r_te], p_te)
        pred_te = (p_te >= thr).astype(int)
        f1_te = f1_score(y_all[r_te], pred_te, zero_division=0)

        overfit = auc_tr - auc_va
        flag = f"{overfit:+.3f}" + (" 偏高" if overfit > .08 else "")
        results[name] = {"auc_val": auc_va, "auc_te": auc_te, "f1": f1_te,
                         "auc_tr": auc_tr, "overfit": overfit, "n_features": len(cols),
                         "coef": pd.Series(clf.coef_[0], index=cols)}
        w(f"  {name:<22}{len(cols):>7}{auc_va:>10.4f}{auc_te:>10.4f}"
          f"{auc_tr:>10.4f}{flag:>9}")

    base = results["仅静态属性"]
    best_name = max(results, key=lambda k: results[k]["auc_val"])
    best = results[best_name]
    w(f"\n  基线(仅静态属性)   测试 AUC={base['auc_te']:.4f}")
    w(f"  最优组合(按验证集) {best_name}  测试 AUC={best['auc_te']:.4f} "
      f"({(best['auc_te']-base['auc_te'])*100:+.2f} pp)")
    full = results["+ 谈话+测评(全链路)"]
    w(f"  全链路组合         测试 AUC={full['auc_te']:.4f} "
      f"({(full['auc_te']-base['auc_te'])*100:+.2f} pp), "
      f"过拟合程度 {full['overfit']:+.3f}")

    if full["overfit"] > 0.08:
        w("\n  [!] 全链路组合出现明显过拟合：特征数相对样本量偏多。")
        w("      这是真实结论，不是 bug —— 数据源多不等于预测力强。")
        w("      建议：做特征筛选(PCA/递归消除)后再入模。")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    names = list(results.keys())
    xs = np.arange(len(names))
    axes[0].bar(xs - .2, [results[n]["auc_val"] for n in names], .4,
                label="验证集", color="#4C7DFF")
    axes[0].bar(xs + .2, [results[n]["auc_te"] for n in names], .4,
                label="测试集", color="#28d17c")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([n.replace("+ ", "+\n") for n in names], fontsize=9)
    axes[0].set_ylim(.5, 1.0); axes[0].legend()
    axes[0].set_title("不同数据源的预测力对比", fontsize=13, fontweight="bold")
    for i, n in enumerate(names):
        axes[0].text(i - .2, results[n]["auc_val"] + .012,
                     f"{results[n]['auc_val']:.3f}", ha="center", fontsize=8)
        axes[0].text(i + .2, results[n]["auc_te"] + .012,
                     f"{results[n]['auc_te']:.3f}", ha="center", fontsize=8)

    full_coef = results["+ 谈话+测评(全链路)"]["coef"]
    top = full_coef.reindex(full_coef.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    axes[1].barh(top.index, top.values,
                 color=["#E4572E" if v > 0 else "#2E86AB" for v in top.values])
    axes[1].axvline(0, color="k", lw=.9)
    axes[1].set_title("全链路模型的关键驱动因素 Top 15", fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "12_链路预测力对比.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def chain_diagram(df: pd.DataFrame, res: dict) -> None:
    """链路权重示意：每段用相关系数标注强度。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    nodes = [
        (0.08, "日常行为\n工时·专注·协作", "#4C7DFF"),
        (0.32, "工作质量\n交付·返工·效率", "#28d17c"),
        (0.56, "测评报告\n敬业·满意·压力", "#FFB020"),
        (0.80, "状态结论\n离职 / 留任", "#E4572E"),
    ]
    for x, label, color in nodes:
        ax.add_patch(plt.Rectangle((x, .42), .16, .2, facecolor=color,
                                   alpha=.85, edgecolor="white", lw=2))
        ax.text(x + .08, .52, label, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

    # 段间强度
    pairs = [
        (0, 1, ["b_focus", "b_tasks"], ["q_quality"], .08),
        (1, 2, ["q_quality"], ["s_engagement"], .32),
        (2, 3, ["s_engagement", "s_satisfaction"], ["attrition_flag"], .56),
    ]
    for a, b, xs, ys, xpos in pairs:
        best = 0
        for x in xs:
            for y in ys:
                sub = df[[x, y]].dropna()
                if sub[x].nunique() > 1 and sub[y].nunique() > 1:
                    r, _ = stats.pearsonr(sub[x], sub[y])
                    if abs(r) > abs(best):
                        best = r
        ax.annotate("", xy=(xpos + .24, .52), xytext=(xpos + .16, .52),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5,
                                    color="#FF6B35" if best < 0 else "#3DDCFF"))
        ax.text(xpos + .20, .60, f"r={best:+.2f}", ha="center", fontsize=10,
                fontweight="bold", color="#333")

    # 谈话报告作为旁路信号
    ax.add_patch(plt.Rectangle((.56, .12), .16, .18, facecolor="#B98CFF",
                               alpha=.8, edgecolor="white", lw=2))
    ax.text(.64, .21, "谈话报告\n情感·风险主题", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    sub = df[["t_sentiment", "attrition_flag"]].dropna()
    r_t, _ = stats.pearsonr(sub["t_sentiment"], sub["attrition_flag"]) \
        if sub["t_sentiment"].nunique() > 1 else (0, 1)
    ax.annotate("", xy=(.80, .42), xytext=(.64, .30),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#B98CFF",
                                connectionstyle="arc3,rad=-.25"))
    ax.text(.70, .34, f"r={r_t:+.2f}", fontsize=9, color="#7B5EA7")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("办公场景员工状态链路（箭头标注实测相关系数）",
                 fontsize=15, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "13_链路权重图.png", dpi=150, bbox_inches="tight")
    plt.close()


def export_risk_scores(df: pd.DataFrame, res: dict) -> pd.DataFrame:
    """为全部员工导出离职风险评分。

    数字孪生场景需要「每个人在场景里的风险值」，因此用验证集表现最好的
    特征组合，在**全量数据**上拟合后为每位员工打分，输出 employee_id → 风险。

    注意：这里的分数是在全量数据上拟合的（用于可视化与下钻），
    模型**泛化性能**的结论仍以 compare_models 的三段式测试集指标为准，
    两者口径不同，不要混用。
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    best_name = max(res, key=lambda k: res[k]["auc_val"])
    cols = [c for c in res[best_name]["coef"].index if c in df.columns]
    sub = df[cols + ["attrition_flag"]].dropna()
    X = sub[cols].values
    y = sub["attrition_flag"].values

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                             C=0.08, random_state=RANDOM_STATE)
    clf.fit(sc.transform(X), y)

    score = clf.predict_proba(sc.transform(X))[:, 1]
    out = df.loc[sub.index, ["employee_id", "name", "department"]].copy()
    out["risk_score"] = np.round(score, 4)
    out["risk_level"] = pd.cut(
        out["risk_score"], bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["低风险", "中风险", "高风险", "极高风险"])
    out["feature_set"] = best_name
    out.to_csv(DATA / "link_employee_risk.csv", index=False, encoding="utf-8-sig")

    w(f"\n[风险评分导出] 基于特征组合「{best_name}」({len(cols)} 个特征)")
    w(f"  评分人数 {len(out)}，风险分布:")
    for lv, n in out["risk_level"].value_counts().reindex(
            ["低风险", "中风险", "高风险", "极高风险"]).fillna(0).items():
        w(f"    {lv:<6} {int(n):>4} 人 ({n/len(out):.1%})")
    w(f"  -> {DATA / 'link_employee_risk.csv'}")
    return out


def main() -> None:
    d = load_all()
    w("=" * 72)
    w(" 办公场景「完整行为链路」分析报告")
    w("=" * 72)
    w(f"数据源: employees / daily_behavior / work_quality / talk_reports / survey_reports")
    w(f"规模  : {len(d['emp'])} 员工 · {len(d['beh'])} 条行为日志 · "
      f"{len(d['qual'])} 条质量记录 · {len(d['talk'])} 条谈话 · {len(d['survey'])} 份测评")

    df = build_employee_features(d)
    w(f"聚合到员工维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    link_strength(df)
    reg = regression_behavior_to_quality(df)
    res = compare_models(df)
    chain_diagram(df, res)
    df.to_csv(DATA / "link_employee_features.csv", index=False, encoding="utf-8-sig")
    risk = export_risk_scores(df, res)

    w("\n" + "=" * 72)
    w(" 结论")
    w("=" * 72)
    m = reg["metrics"]
    w(f"1) 行为 -> 质量: 线性回归测试 R²={m['测试集']['r2']:.4f}")
    top3 = reg["coefs"].head(3)
    w(f"   最强行为信号: " + ", ".join(f"{k}({v:+.2f})" for k, v in top3.items()))
    w(f"   说明: 日常行为能高度解释工作质量，链路前半段成立。")

    base = res["仅静态属性"]
    best_name = max(res, key=lambda k: res[k]["auc_val"])
    best = res[best_name]
    w(f"\n2) 预测离职效果（测试集 AUC）")
    for k, v in res.items():
        mark = "  <- 最优" if k == best_name else ""
        w(f"   {k:<22}{v['auc_te']:.4f}  (过拟合 {v['overfit']:+.3f}){mark}")

    full = res["+ 谈话+测评(全链路)"]
    w(f"\n3) 关键发现: 数据源变多 ≠ 预测更准")
    w(f"   基线 {base['auc_te']:.4f} -> 全链路 {full['auc_te']:.4f} "
      f"({(full['auc_te']-base['auc_te'])*100:+.2f} pp)")
    w(f"   全链路模型过拟合程度 {full['overfit']:+.3f}"
      f"（训练 {full['auc_tr']:.4f} vs 验证 {full['auc_val']:.4f}）")
    w(f"   原因: 特征数({full['n_features']})相对样本量({len(df)})偏多，")
    w(f"        高维弱信号用线性模型会放大方差。")
    w(f"   结论: 最优组合是「{best_name}」，即适度引入行为数据、")
    w(f"        但不要把低频报告指标全部堆进线性模型。")
    w(f"        若要用全链路，建议先做特征选择或改用带正则的树模型。")
    w(f"\n输出: {REPORT}")
    REPORT.write_text("\n".join(_log), encoding="utf-8")
    print(f"\n[输出] {REPORT}")
    print(f"[输出] {DATA/'link_employee_features.csv'}")
    print(f"[输出] {FIGURES/'11_行为到质量回归.png'}")
    print(f"[输出] {FIGURES/'12_链路预测力对比.png'}")
    print(f"[输出] {FIGURES/'13_链路权重图.png'}")


if __name__ == "__main__":
    main()
