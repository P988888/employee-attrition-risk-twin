"""Step 1 数据准备。

1) 读取 IBM HR Analytics 真实数据（1470 条，含 Attrition 标签）
2) 生成与真实数据统计分布一致的模拟数据做补充（默认 1530 条，凑成 3000 条）
   模拟数据用「先模仿真实分布、再按真实模型的效应方向注入标签」的方式生成，
   保证它既能扩大样本量，又不会引入与真实数据矛盾的规律。

注意：模拟数据仅用于 3D 大屏的可视体量与压力测试，
     所有模型指标都以真实数据上的表现为准（见 train_model.py 的 sim_eval）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW_CSV, SIM_CSV, MERGED_CSV, CONSTANT_COLS, TARGET, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)
N_SIM = 1530


def load_real() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df = df.drop(columns=CONSTANT_COLS, errors="ignore")
    df["_source"] = "real"
    return df


def _pick(values, p, size):
    p = np.asarray(p, dtype=float)
    return RNG.choice(np.asarray(values), size=size, p=p / p.sum())


def build_simulated(real: pd.DataFrame) -> pd.DataFrame:
    n = N_SIM
    out = pd.DataFrame(index=range(n))

    # ---- 人口统计：对齐真实边际分布 ----
    out["Age"] = np.clip(RNG.normal(real.Age.mean(), real.Age.std(), n), 18, 60).round().astype(int)
    out["Gender"] = _pick(["Male", "Female"], real.Gender.value_counts().values, n)
    out["MaritalStatus"] = _pick(
        real.MaritalStatus.unique(),
        [real.MaritalStatus.eq(v).mean() for v in real.MaritalStatus.unique()], n)
    out["EducationField"] = _pick(
        real.EducationField.unique(),
        [real.EducationField.eq(v).mean() for v in real.EducationField.unique()], n)
    out["Education"] = _pick([1, 2, 3, 4, 5],
                             [real.Education.eq(v).mean() for v in [1, 2, 3, 4, 5]], n)

    # ---- 组织归属：部门 -> 职位 的联合分布取自真实数据 ----
    role_by_dept = {d: real.loc[real.Department.eq(d), "JobRole"].value_counts() for d in real.Department.unique()}
    dept = _pick(real.Department.unique(),
                 [real.Department.eq(v).mean() for v in real.Department.unique()], n)
    out["Department"] = dept
    roles = []
    for d in dept:
        vc = role_by_dept[d]
        roles.append(RNG.choice(vc.index.values, p=(vc.values / vc.values.sum())))
    out["JobRole"] = roles

    out["BusinessTravel"] = _pick(
        real.BusinessTravel.unique(),
        [real.BusinessTravel.eq(v).mean() for v in real.BusinessTravel.unique()], n)
    out["OverTime"] = _pick(["No", "Yes"], real.OverTime.value_counts().values, n)

    # ---- 职级 -> 薪酬/工龄 的层级结构 ----
    out["JobLevel"] = _pick([1, 2, 3, 4, 5],
                            [real.JobLevel.eq(v).mean() for v in [1, 2, 3, 4, 5]], n)
    income_base = {1: 2400, 2: 5500, 3: 9500, 4: 15000, 5: 19500}
    lvl = out["JobLevel"].to_numpy()
    income = np.array([income_base[v] for v in lvl], dtype=float) * RNG.normal(1.0, 0.22, n)
    out["MonthlyIncome"] = np.clip(income, 1009, 19999).round().astype(int)

    total_years = np.clip(lvl * 4.2 + RNG.normal(0, 4.5, n), 0, 40)
    out["TotalWorkingYears"] = total_years.round().astype(int)
    out["YearsAtCompany"] = np.clip(total_years * RNG.uniform(0.15, 0.85, n), 0, 40).round().astype(int)
    out["YearsInCurrentRole"] = np.clip(out.YearsAtCompany * RNG.uniform(0.1, 0.9, n), 0, 18).round().astype(int)
    out["YearsSinceLastPromotion"] = np.clip(out.YearsAtCompany * RNG.uniform(0.0, 0.6, n), 0, 15).round().astype(int)
    out["YearsWithCurrManager"] = np.clip(out.YearsAtCompany * RNG.uniform(0.1, 0.95, n), 0, 17).round().astype(int)
    out["NumCompaniesWorked"] = np.clip(RNG.poisson(2.6, n), 0, 9).astype(int)

    # ---- 主观量表（1-4 分）----
    out["JobSatisfaction"] = _pick([1, 2, 3, 4], [0.20, 0.19, 0.30, 0.31], n)
    out["EnvironmentSatisfaction"] = _pick([1, 2, 3, 4], [0.19, 0.19, 0.29, 0.33], n)
    out["RelationshipSatisfaction"] = _pick([1, 2, 3, 4], [0.19, 0.21, 0.29, 0.31], n)
    out["WorkLifeBalance"] = _pick([1, 2, 3, 4], [0.06, 0.23, 0.61, 0.10], n)
    out["JobInvolvement"] = _pick([1, 2, 3, 4], [0.06, 0.26, 0.60, 0.08], n)
    out["StockOptionLevel"] = _pick([0, 1, 2, 3], [0.43, 0.34, 0.16, 0.07], n)
    out["PerformanceRating"] = _pick([3, 4], [0.85, 0.15], n)
    out["TrainingTimesLastYear"] = np.clip(RNG.poisson(2.8, n), 0, 6).astype(int)
    out["PercentSalaryHike"] = np.clip(RNG.normal(15.2, 3.7, n), 11, 25).round().astype(int)
    out["DistanceFromHome"] = np.clip(RNG.exponential(7.0, n) + 1, 1, 29).round().astype(int)

    # ---- 与主任务无关但保留字段完整性的列 ----
    out["DailyRate"] = RNG.integers(102, 1500, n)
    out["HourlyRate"] = RNG.integers(30, 101, n)
    out["MonthlyRate"] = RNG.integers(2094, 27000, n)

    # ---- 标签：按真实数据 logistic 拟合出的效应方向注入，再保证 16% 左右离职率 ----
    logit = (
        -2.05
        + 1.15 * (out.OverTime.eq("Yes").to_numpy() - 0.28)
        + 0.62 * (out.MaritalStatus.eq("Single").to_numpy() - 0.32)
        + 0.40 * (out.BusinessTravel.eq("Travel_Frequently").to_numpy() - 0.19)
        - 0.55 * (out.JobSatisfaction.to_numpy() - 2.7) / 1.1
        - 0.42 * (out.EnvironmentSatisfaction.to_numpy() - 2.7) / 1.1
        - 0.35 * (out.JobInvolvement.to_numpy() - 2.7) / 0.7
        - 0.30 * (out.WorkLifeBalance.to_numpy() - 2.8) / 0.7
        - 0.85 * (np.log(out.MonthlyIncome.to_numpy() / 3000.0)) / 1.0
        - 0.045 * (out.YearsAtCompany.to_numpy() - 7.0)
        + 0.16 * (out.NumCompaniesWorked.to_numpy() - 2.7)
        + 0.030 * (out.DistanceFromHome.to_numpy() - 9.2) / 1.0
        + 0.28 * (out.JobLevel.to_numpy() == 1)
        + 0.05 * (out.Age.to_numpy() - 36.9) / 9.1
        + RNG.normal(0, 0.55, n)
    )
    p = 1 / (1 + np.exp(-logit))
    # 用分位数阈值把总体离职率钉在真实水平附近（16.1%）
    cut = np.quantile(p, 1 - 0.161)
    out[TARGET] = np.where(p >= cut, "Yes", "No")
    out["_source"] = "simulated"

    # 补齐真实数据结构里存在但上面未生成的列顺序
    real_cols = [c for c in real.columns if c != "_source"]
    for c in real_cols:
        if c not in out.columns:
            out[c] = 0
    return out[real_cols + ["_source"]]


def main() -> None:
    real = load_real()
    sim = build_simulated(real)
    merged = pd.concat([real, sim], ignore_index=True)

    sim.to_csv(SIM_CSV, index=False, encoding="utf-8-sig")
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")

    print(f"[真实]     {real.shape[0]:>5} 行  离职率 {real[TARGET].eq('Yes').mean():6.2%}")
    print(f"[模拟]     {sim.shape[0]:>5} 行  离职率 {sim[TARGET].eq('Yes').mean():6.2%}")
    print(f"[合并]     {merged.shape[0]:>5} 行  离职率 {merged[TARGET].eq('Yes').mean():6.2%}")
    print(f"-> {SIM_CSV.name} / {MERGED_CSV.name}")

    # 关键字段分布对照，确认模拟数据没有跑偏
    chk = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
    print("\n分布对照 (real vs sim mean / std):")
    for c in chk:
        print(f"  {c:<20} real {real[c].mean():8.1f}/{real[c].std():7.1f}   "
              f"sim {sim[c].mean():8.1f}/{sim[c].std():7.1f}")


if __name__ == "__main__":
    main()
