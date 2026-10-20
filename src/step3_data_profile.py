"""Step 3 数据整理与探索性分析（EDA）。

输出一份可直接贴进报告的「数据体检」结果：
  1. 数据规模、缺失、重复、异常
  2. 目标变量分布与不平衡程度
  3. 数值特征描述统计 + 偏度
  4. 类别特征分布与离职率交叉
  5. 数值特征与离职的相关性排序
  6. 缺失/异常清单

同时把清洗后的规范化数据导出，供 Jupyter notebook 直接读取，
避免 notebook 里重复实现清洗逻辑。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from config import RAW_CSV, MERGED_CSV, DATA, FIGURES, TARGET, NUMERIC_COLS, CATEGORICAL_COLS

_CJK = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Hiragino Sans GB", "STHeiti"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in _CJK if f in _avail), "Arial Unicode MS")
plt.rcParams["font.sans-serif"] = [CJK] + _CJK
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=CJK)

OUT_TXT = DATA / "data_profile.txt"


def section(title: str, lines: list[str]) -> list[str]:
    block = ["", "=" * 64, f" {title}", "=" * 64]
    return block + lines


def main() -> None:
    # 优先读合并数据；没有就先跑 step1
    path = MERGED_CSV if MERGED_CSV.exists() else RAW_CSV
    df = pd.read_csv(path, encoding="utf-8-sig")
    real = df[df["_source"] == "real"] if "_source" in df.columns else df
    report: list[str] = []
    report.append(f"数据来源: {path.name}")
    report.append(f"读入形状: {df.shape}    真实子集: {real.shape}")

    # ---------- 1. 质量体检 ----------
    n_dup = int(df.duplicated().sum())
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    miss = df.isna().sum()
    miss = miss[miss > 0]
    report += section("一、数据质量体检", [
        f"总行数        : {len(df)}",
        f"总列数        : {df.shape[1]}",
        f"完全重复行    : {n_dup}",
        f"含缺失值列数  : {len(miss)}  ({'无缺失' if len(miss) == 0 else dict(miss)})",
        f"单值常量列    : {const_cols if const_cols else '无'}",
        f"数值列        : {len(NUMERIC_COLS)} 个",
        f"类别列        : {len(CATEGORICAL_COLS)} 个",
    ])

    # ---------- 2. 目标变量 ----------
    vc = real[TARGET].value_counts()
    rate = real[TARGET].eq("Yes").mean()
    report += section("二、目标变量分布（离职 Attrition）", [
        f"留任 No  : {vc.get('No', 0):>5} 条  ({vc.get('No', 0)/len(real):.2%})",
        f"离职 Yes : {vc.get('Yes', 0):>5} 条  ({rate:.2%})",
        f"不平衡比 : 1 : {vc.get('No', 1)/max(vc.get('Yes', 1), 1):.2f}",
        "结论     : 属中度不平衡，建模需用 class_weight 或重采样，",
        "           且评估不能只看准确率（全预测留任也有 83.9% 准确率）。",
    ])

    # ---------- 3. 数值特征描述 ----------
    desc = real[NUMERIC_COLS].describe().T
    desc["偏度"] = real[NUMERIC_COLS].skew()
    desc["缺失"] = real[NUMERIC_COLS].isna().sum()
    report += section("三、数值特征描述统计", [
        f"{'特征':<24}{'均值':>10}{'标准差':>10}{'最小':>8}{'中位':>9}{'最大':>9}{'偏度':>9}",
        "-" * 80,
    ])
    for c, r in desc.iterrows():
        report.append(
            f"{c:<24}{r['mean']:>10.2f}{r['std']:>10.2f}{r['min']:>8.0f}"
            f"{r['50%']:>9.1f}{r['max']:>9.0f}{r['偏度']:>9.2f}")

    hi_skew = desc[desc["偏度"].abs() > 1].index.tolist()
    report.append("")
    report.append(f"高偏度(|偏度|>1)特征: {hi_skew if hi_skew else '无'}")
    report.append("说明: 偏度高不代表要删，树模型不受影响；逻辑回归靠标准化缓解。")

    # ---------- 4. 类别特征 × 离职率 ----------
    report += section("四、类别特征与离职率交叉", [])
    key_cat = [c for c in CATEGORICAL_COLS if real[c].nunique() <= 10]
    for c in key_cat:
        g = real.groupby(c, observed=True)[TARGET].agg(
            人数="size", 离职数=lambda s: s.eq("Yes").sum())
        g["离职率"] = g["离职数"] / g["人数"]
        g = g.sort_values("离职率", ascending=False)
        report.append(f"\n[{c}]")
        report.append(f"  {'取值':<22}{'人数':>7}{'离职数':>8}{'离职率':>9}")
        for k, r in g.iterrows():
            flag = "  <<< 高" if r["离职率"] > rate * 1.5 else ""
            report.append(f"  {str(k):<22}{int(r['人数']):>7}{int(r['离职数']):>8}"
                          f"{r['离职率']:>9.2%}{flag}")

    # ---------- 5. 相关性 ----------
    num_df = real[NUMERIC_COLS].copy()
    num_df["离职"] = real[TARGET].eq("Yes").astype(int)
    corr = num_df.corr()["离职"].drop("离职").sort_values(key=abs, ascending=False)
    report += section("五、数值特征与离职的相关性（按 |r| 排序）", [
        f"  {'特征':<26}{'Pearson r':>12}{'方向':>8}", "-" * 48,
    ])
    for k, v in corr.items():
        report.append(f"  {k:<26}{v:>12.4f}{'正相关' if v > 0 else '负相关':>8}")

    # ---------- 导出 ----------
    OUT_TXT.write_text("\n".join(report), encoding="utf-8")

    # 清洗后的分析用数据（真实 + 模拟，带 source 标记）
    clean = df.drop(columns=["EmployeeCount", "Over18", "StandardHours",
                             "EmployeeNumber"], errors="ignore")
    clean.to_csv(DATA / "clean_analysis.csv", index=False, encoding="utf-8-sig")

    # EDA 配图
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].bar(["留任", "离职"], [vc.get("No", 0), vc.get("Yes", 0)],
                color=["#2E86AB", "#E4572E"])
    for i, v in enumerate([vc.get("No", 0), vc.get("Yes", 0)]):
        axes[0].text(i, v + 12, f"{v}\n{v/len(real):.1%}", ha="center", fontsize=10)
    axes[0].set_title("目标变量分布（不平衡）", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, vc.max() * 1.2)

    top = corr.head(12)[::-1]
    axes[1].barh(top.index, top.values,
                 color=["#E4572E" if v > 0 else "#2E86AB" for v in top.values])
    axes[1].axvline(0, color="k", lw=.8)
    axes[1].set_title("与离职相关性 Top 12", fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=8)

    ot = real.groupby("OverTime")[TARGET].apply(lambda s: s.eq("Yes").mean())
    axes[2].bar(["不加班", "加班"], [ot.get("No", 0), ot.get("Yes", 0)],
                color=["#4CAF50", "#E4572E"])
    for i, v in enumerate([ot.get("No", 0), ot.get("Yes", 0)]):
        axes[2].text(i, v + .008, f"{v:.1%}", ha="center", fontsize=11)
    axes[2].axhline(rate, ls="--", color="k", lw=1, label=f"总体 {rate:.1%}")
    axes[2].set_title("加班与离职率", fontsize=13, fontweight="bold")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "07_数据体检.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n".join(report))
    print(f"\n[输出] {OUT_TXT}")
    print(f"[输出] {DATA/'clean_analysis.csv'}")
    print(f"[输出] {FIGURES/'07_数据体检.png'}")


if __name__ == "__main__":
    main()
