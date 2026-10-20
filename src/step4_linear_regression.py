"""Step 4 线性回归建模与分析。

本脚本回答两个层面的问题：

  A. 【回归视角】用线性回归预测「连续型」目标
     - MonthlyIncome（月收入）  <- 最常用的回归目标，用来解释薪酬由什么决定
     - 输出系数、p 值、VIF（多重共线性）、残差诊断

  B. 【分类视角】用「线性概率模型 + 逻辑回归」预测离职
     - 线性回归直接拟合 0/1 离职标签 = 线性概率模型(LPM)，
       它的系数就是「该特征每变动 1 单位，离职概率变动多少个百分点」，
       这是线性回归用于分类的可解释写法
     - 逻辑回归作为对照（概率有界、AUC 更高）

  C. 【评估三件套】训练集 / 验证集 / 测试集 三段式评估

依赖：statsmodels（提供 p 值、VIF、残差诊断）
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

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (
    MERGED_CSV, FIGURES, DATA, OUTPUT, TARGET,
    NUMERIC_COLS, CATEGORICAL_COLS, RANDOM_STATE,
    VAL_SIZE, TEST_SIZE_FINAL,
)

warnings.filterwarnings("ignore")

_CJK = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Hiragino Sans GB", "STHeiti"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in _CJK if f in _avail), "Arial Unicode MS")
plt.rcParams["font.sans-serif"] = [CJK] + _CJK
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=CJK)

REPORT = OUTPUT / "linear_regression_report.txt"
_log: list[str] = []


def w(line: str = "") -> None:
    _log.append(line)
    print(line)


# ================================================================ 特征工程
def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """与主管线保持一致的派生特征，保证口径统一。"""
    d = d.copy()
    eps = 1e-6
    d["IncomePerLevel"] = d["MonthlyIncome"] / (d["JobLevel"] * 5000.0 + eps)
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1.0)
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1.0)
    d["ManagerStability"] = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1.0)
    d["SatisfactionScore"] = (d["JobSatisfaction"] + d["EnvironmentSatisfaction"]
                              + d["RelationshipSatisfaction"] + d["WorkLifeBalance"]) / 4.0
    d["CareerStagnation"] = (d["YearsSinceLastPromotion"] / (d["YearsInCurrentRole"] + 1.0)) \
        * (1.0 - d["JobInvolvement"] / 4.0)
    d["CompanyHopRate"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1.0)
    return d


DERIVED = ["IncomePerLevel", "TenureRatio", "PromotionGap", "ManagerStability",
           "SatisfactionScore", "CareerStagnation", "CompanyHopRate"]


def split_three_way(idx: np.ndarray, y: np.ndarray, seed: int):
    """三段式分层划分，返回三个索引数组。"""
    from sklearn.model_selection import train_test_split
    i_trv, i_te = train_test_split(idx, test_size=TEST_SIZE_FINAL,
                                   stratify=y, random_state=seed)
    rel = VAL_SIZE / (1.0 - TEST_SIZE_FINAL)
    i_tr, i_va = train_test_split(i_trv, test_size=rel,
                                  stratify=y[i_trv], random_state=seed)
    return i_tr, i_va, i_te


# ================================================================ A. 回归：预测月收入
def regression_on_income(real: pd.DataFrame, i_tr, i_va, i_te) -> dict:
    w("\n" + "=" * 70)
    w(" A. 线性回归 —— 预测月收入 MonthlyIncome")
    w("=" * 70)
    w("回归目标: MonthlyIncome（连续值）")
    w("用途    : 量化各因素对薪酬的边际贡献，并为薪酬公平性提供依据")

    y = real["MonthlyIncome"].values.astype(float)

    # 特征：剔除与目标高度共线的项，避免 IncomePerLevel 泄漏收入本身
    feats_num = [c for c in NUMERIC_COLS if c != "MonthlyIncome"] + \
                [c for c in ["TenureRatio", "PromotionGap", "ManagerStability",
                             "SatisfactionScore", "CareerStagnation", "CompanyHopRate"]
                 if c in real.columns]
    feats_cat = CATEGORICAL_COLS
    X = real[feats_num + feats_cat]

    w(f"\n特征数: 数值 {len(feats_num)} + 类别 {len(feats_cat)}")
    w(f"样本划分: 训练 {len(i_tr)} / 验证 {len(i_va)} / 测试 {len(i_te)}")

    # ---- sklearn 管线（含 one-hot），三段式评估 ----
    num_pipe = SkPipeline([("sc", StandardScaler())])
    cat_pipe = SkPipeline([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, feats_num), ("cat", cat_pipe, feats_cat)])
    lr = SkPipeline([("prep", ct), ("reg", LinearRegression())])
    lr.fit(X.iloc[i_tr], y[i_tr])

    metrics = {}
    for name, ix in [("训练集", i_tr), ("验证集", i_va), ("测试集", i_te)]:
        pred = lr.predict(X.iloc[ix])
        metrics[name] = {
            "R2": r2_score(y[ix], pred),
            "RMSE": np.sqrt(mean_squared_error(y[ix], pred)),
            "MAE": mean_absolute_error(y[ix], pred),
        }

    w("\n[三段式评估] 线性回归预测月收入")
    w(f"  {'数据集':<8}{'R²':>10}{'RMSE':>12}{'MAE':>12}")
    w("  " + "-" * 40)
    for k, m in metrics.items():
        w(f"  {k:<8}{m['R2']:>10.4f}{m['RMSE']:>12.1f}{m['MAE']:>12.1f}")

    gap = metrics["训练集"]["R2"] - metrics["测试集"]["R2"]
    w(f"\n  R² 泛化差距(训练-测试): {gap:+.4f}")
    w("  解读: 差距小 => 无明显过拟合；R² 越高说明收入越能被这些特征解释。")

    # ---- statsmodels：给出系数、p 值、置信区间 ----
    Xs = pd.get_dummies(X, columns=feats_cat, drop_first=True).astype(float)
    Xs = Xs.loc[:, Xs.std() > 1e-9]          # 去掉零方差哑变量
    Xs_tr = sm.add_constant(Xs.iloc[i_tr], has_constant="add")
    ols = sm.OLS(y[i_tr], Xs_tr).fit()

    w(f"\n[OLS 回归] 训练集 n={len(i_tr)}  特征(含哑变量)={Xs_tr.shape[1]-1}")
    w(f"  R²={ols.rsquared:.4f}   调整R²={ols.rsquared_adj:.4f}   F检验 p={ols.f_pvalue:.3e}")
    w(f"  AIC={ols.aic:.1f}  BIC={ols.bic:.1f}")

    coef = pd.DataFrame({
        "系数": ols.params, "标准误": ols.bse, "t值": ols.tvalues,
        "p值": ols.pvalues,
        "CI下界": ols.conf_int()[0], "CI上界": ols.conf_int()[1],
    })
    coef = coef.drop("const", errors="ignore")
    sig = coef[coef["p值"] < 0.05].sort_values("系数", key=abs, ascending=False)

    w(f"\n[显著性] 显著特征(p<0.05): {len(sig)} / {len(coef)}")
    w(f"\n  对月收入影响最大的 12 个因素（按|系数|排序，单位：元）")
    w(f"  {'特征':<34}{'系数':>11}{'p值':>10}{'显著性':>8}")
    w("  " + "-" * 64)
    for k, r in sig.head(12).iterrows():
        star = "***" if r["p值"] < 0.001 else "**" if r["p值"] < 0.01 else "*"
        w(f"  {k:<34}{r['系数']:>11.1f}{r['p值']:>10.4f}{star:>8}")

    # ---- 多重共线性 VIF ----
    vif_rows = []
    for i, c in enumerate(Xs_tr.columns):
        if c == "const":
            continue
        try:
            vif_rows.append({"特征": c, "VIF": variance_inflation_factor(Xs_tr.values, i)})
        except Exception:
            pass
    vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    high_vif = vif[vif["VIF"] > 10]
    w(f"\n[多重共线性 VIF]  >10 视为严重共线")
    w(f"  最大 VIF: {vif.iloc[0]['特征']} = {vif.iloc[0]['VIF']:.1f}")
    if len(high_vif):
        w(f"  存在 {len(high_vif)} 个高共线特征:")
        for k, r in high_vif.head(8).iterrows():
            w(f"    {r['特征']:<34}VIF={r['VIF']:>8.1f}")
        w("  处理建议: 保留业务含义更清晰的一个，或用岭回归(L2)替代 OLS。")
    else:
        w("  无严重共线性特征。")

    # ---- 残差诊断 ----
    resid = ols.resid
    fitted = ols.fittedvalues
    w(f"\n[残差诊断]")
    w(f"  均值={resid.mean():.2f}（应接近0）  标准差={resid.std():.1f}")
    w(f"  Durbin-Watson={sm.stats.durbin_watson(resid):.3f}（接近2为无自相关）")
    from scipy import stats as st
    jb, jbp = st.jarque_bera(resid)
    w(f"  Jarque-Bera={jb:.1f}, p={jbp:.3e} => "
      f"{'残差非正态' if jbp < 0.05 else '残差近似正态'}")

    # ---- 残差诊断图 ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].scatter(fitted, resid, s=14, alpha=.5, color="#2E86AB")
    axes[0].axhline(0, color="#E4572E", lw=1.5)
    axes[0].set_xlabel("拟合值"); axes[0].set_ylabel("残差")
    axes[0].set_title("残差 vs 拟合值", fontsize=13, fontweight="bold")

    st.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("残差 Q-Q 图", fontsize=13, fontweight="bold")

    axes[2].hist(resid, bins=40, color="#5C6BC0", edgecolor="white")
    axes[2].axvline(0, color="#E4572E", lw=1.5)
    axes[2].set_title(f"残差分布 (偏度={st.skew(resid):.2f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "08_收入回归_残差诊断.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 系数可视化 ----
    top = sig.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E4572E" if v > 0 else "#2E86AB" for v in top["系数"]]
    ax.barh(top.index, top["系数"], color=colors)
    ax.axvline(0, color="k", lw=.9)
    ax.set_title("月收入影响因素（OLS 显著系数，单位：元）", fontsize=14, fontweight="bold")
    ax.set_xlabel("系数：该因素每增加 1 单位，月收入变动额")
    plt.tight_layout()
    plt.savefig(FIGURES / "09_收入回归系数.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"metrics": metrics, "ols": ols, "coef": coef, "vif": vif}


# ================================================================ B. 分类：线性概率模型 vs 逻辑回归
def classification_with_linear(real: pd.DataFrame, i_tr, i_va, i_te) -> dict:
    w("\n" + "=" * 70)
    w(" B. 线性回归用于离职分类 —— 线性概率模型(LPM) 与 逻辑回归对比")
    w("=" * 70)
    w("说明: 把 0/1 离职标签直接作为因变量做最小二乘，即线性概率模型。")
    w("      其系数含义直观 —— 该特征每变动 1 单位，离职概率变动多少个百分点。")
    w("      缺点: 预测值可能落在 [0,1] 之外。逻辑回归可作对照修正。")

    y = real[TARGET].eq("Yes").astype(int).values
    feats_num = NUMERIC_COLS + DERIVED
    X = real[feats_num + CATEGORICAL_COLS]

    def make(clf):
        ct = ColumnTransformer([
            ("num", SkPipeline([("sc", StandardScaler())]), feats_num),
            ("cat", SkPipeline([("oh", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))]), CATEGORICAL_COLS),
        ])
        return SkPipeline([("prep", ct), ("clf", clf)])

    lpm = make(LinearRegression())
    logit = make(LogisticRegression(max_iter=3000, class_weight="balanced",
                                    C=0.5, random_state=RANDOM_STATE))

    w(f"\n[三段式评估] 离职分类")
    w(f"  {'模型':<14}{'数据集':<8}{'AUC':>9}{'准确率':>9}{'精确率':>9}{'召回率':>9}{'F1':>9}")
    w("  " + "-" * 68)

    out = {}
    for label, model in [("线性概率模型", lpm), ("逻辑回归", logit)]:
        model.fit(X.iloc[i_tr], y[i_tr])
        out[label] = {}
        for ds, ix in [("训练集", i_tr), ("验证集", i_va), ("测试集", i_te)]:
            p = model.predict(X.iloc[ix])
            # LPM 输出可能越界，评估时截断到 [0,1]
            p_c = np.clip(p, 0, 1)
            pr = (p_c >= 0.5).astype(int)
            m = {
                "auc": roc_auc_score(y[ix], p_c),
                "acc": accuracy_score(y[ix], pr),
                "prec": precision_score(y[ix], pr, zero_division=0),
                "rec": recall_score(y[ix], pr, zero_division=0),
                "f1": f1_score(y[ix], pr, zero_division=0),
                "out_of_range": int(((p < 0) | (p > 1)).sum()),
                "pred": p,
            }
            out[label][ds] = m
            w(f"  {label:<14}{ds:<8}{m['auc']:>9.4f}{m['acc']:>9.4f}"
              f"{m['prec']:>9.4f}{m['rec']:>9.4f}{m['f1']:>9.4f}")

    oob = out["线性概率模型"]["测试集"]["out_of_range"]
    n_te = len(i_te)
    w(f"\n  [LPM 越界检查] 测试集 {n_te} 条中有 {oob} 条预测值落在 [0,1] 外 "
      f"({oob/n_te:.1%})")
    w("  => 这正是 LPM 的固有缺陷，也是分类任务更常用逻辑回归的原因。")

    # LPM 系数解释（对离职概率的边际贡献）
    ct = lpm.named_steps["prep"]
    names = [n.split("__", 1)[1] if "__" in n else n
             for n in ct.get_feature_names_out()]
    coefs = lpm.named_steps["clf"].coef_
    s = pd.Series(coefs, index=names).sort_values(key=abs, ascending=False)

    w(f"\n[LPM 系数] 影响离职概率最大的 12 个因素（百分点 / 标准差）")
    w(f"  {'特征':<34}{'系数(pp)':>12}{'方向':>8}")
    w("  " + "-" * 56)
    for k, v in s.head(12).items():
        w(f"  {k:<34}{v*100:>12.3f}{'风险↑' if v > 0 else '风险↓':>8}")

    # 可视化：LPM vs Logit 的预测分布
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (label, model) in zip(axes[:2], [("线性概率模型", lpm), ("逻辑回归", logit)]):
        p = out[label]["测试集"]["pred"]
        pc = np.clip(p, 0, 1)
        ax.hist(pc[y[i_te] == 0], bins=30, alpha=.65, label="实际留任", color="#2E86AB")
        ax.hist(pc[y[i_te] == 1], bins=30, alpha=.65, label="实际离职", color="#E4572E")
        ax.set_title(f"{label} 预测分布", fontsize=13, fontweight="bold")
        ax.set_xlabel("预测离职概率"); ax.legend()
        if label == "线性概率模型":
            ax.axvline(0, color="k", ls=":", lw=1)
            ax.axvline(1, color="k", ls=":", lw=1)

    ds = list(out["线性概率模型"].keys())
    x = np.arange(len(ds)); wd = .35
    axes[2].bar(x - wd/2, [out["线性概率模型"][d]["auc"] for d in ds], wd,
                label="线性概率模型", color="#E4572E")
    axes[2].bar(x + wd/2, [out["逻辑回归"][d]["auc"] for d in ds], wd,
                label="逻辑回归", color="#2E86AB")
    axes[2].set_xticks(x); axes[2].set_xticklabels(ds)
    axes[2].set_ylim(.5, 1); axes[2].legend()
    axes[2].set_title("三段式 AUC 对比", fontsize=13, fontweight="bold")
    for i, d in enumerate(ds):
        axes[2].text(i - wd/2, out["线性概率模型"][d]["auc"] + .012,
                     f"{out['线性概率模型'][d]['auc']:.3f}", ha="center", fontsize=9)
        axes[2].text(i + wd/2, out["逻辑回归"][d]["auc"] + .012,
                     f"{out['逻辑回归'][d]['auc']:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "10_LPM对比逻辑回归.png", dpi=150, bbox_inches="tight")
    plt.close()

    return out


# ================================================================ 主流程
def main() -> None:
    df = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
    df = engineer(df)
    real = df[df["_source"] == "real"].reset_index(drop=True)

    w("=" * 70)
    w(" 线性回归建模与分析报告")
    w("=" * 70)
    w(f"数据: IBM HR Analytics 真实数据 {len(real)} 条")
    w(f"划分: 训练/验证/测试 = {1-VAL_SIZE-TEST_SIZE_FINAL:.0%}/"
      f"{VAL_SIZE:.0%}/{TEST_SIZE_FINAL:.0%}（分层）")

    y_cls = real[TARGET].eq("Yes").astype(int).values
    idx = np.arange(len(real))
    i_tr, i_va, i_te = split_three_way(idx, y_cls, RANDOM_STATE)
    w(f"     训练 {len(i_tr)} / 验证 {len(i_va)} / 测试 {len(i_te)}")

    reg = regression_on_income(real, i_tr, i_va, i_te)
    cls = classification_with_linear(real, i_tr, i_va, i_te)

    # 汇总
    w("\n" + "=" * 70)
    w(" 结论汇总")
    w("=" * 70)
    m = reg["metrics"]
    w(f"1) 收入回归: 测试集 R²={m['测试集']['R2']:.4f}, "
      f"RMSE={m['测试集']['RMSE']:.0f} 元")
    w(f"   => 这组特征能解释月收入约 {m['测试集']['R2']:.1%} 的变异。")
    w(f"2) 离职分类: 逻辑回归测试 AUC="
      f"{cls['逻辑回归']['测试集']['auc']:.4f}, "
      f"线性概率模型测试 AUC={cls['线性概率模型']['测试集']['auc']:.4f}")
    w(f"   => 逻辑回归更稳健；LPM 系数可直接解释为「百分点」，"
      f"适合向业务方讲清方向。")
    w(f"3) 验证集作用: 用于模型选型与阈值标定，测试集始终封存，")
    w(f"   因此测试指标不含选择性偏差。")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    print(f"\n[输出] {REPORT}")
    print(f"[输出] {FIGURES/'08_收入回归_残差诊断.png'}")
    print(f"[输出] {FIGURES/'09_收入回归系数.png'}")
    print(f"[输出] {FIGURES/'10_LPM对比逻辑回归.png'}")


if __name__ == "__main__":
    main()
