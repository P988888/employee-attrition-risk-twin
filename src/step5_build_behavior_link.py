"""Step 5 办公场景「完整行为链路」多源数据构建。

把一次员工状态观测拆成 5 张互相可关联的表，模拟一个真实办公系统：

  1. employees        员工主数据（静态属性）
  2. daily_behavior   日常行为日志（打卡、工时、专注、协作、产出）
  3. work_quality     工作质量与效率指标（交付、返工、缺陷、达成率）
  4. talk_reports     谈话报告（1:1 面谈、离职面谈、HR 访谈摘要）
  5. survey_reports   测评报告（满意度、敬业度、压力、能力评估）

关联键统一为 employee_id，时间键为 date / 周次，可做任意维度下钻。

设计原则：
  - 行为指标是「日频」信号，报告是「低频」信号，二者时间粒度不同
  - 链路方向：行为 -> 质量 -> 测评/谈话 -> 状态结论(离职/留任)
  - 所有表共享同一批 employee_id，且保持与 IBM 主数据一致的部门/职位分布
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA, MERGED_CSV, TARGET, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)

# 观测窗口
N_DAYS = 90          # 90 天行为日志
N_WEEKS = 13         # 约 13 周

# 四个部门（市场部 / 研发部 / 财务部 / 运营部）—— 与 step7 平面布局一致
DEPARTMENTS = [
    ("marketing", "市场部", 0.25),
    ("rnd", "研发部", 0.40),
    ("finance", "财务部", 0.15),
    ("ops", "运营部", 0.20),
]
DEPT_KEYS = [d[0] for d in DEPARTMENTS]
DEPT_CN = {d[0]: d[1] for d in DEPARTMENTS}
DEPT_W = [d[2] for d in DEPARTMENTS]

# 各部门的典型职位（用于生成与部门匹配的岗位）
ROLES_BY_DEPT = {
    "marketing": ["市场营销经理", "市场专员", "品牌经理", "内容运营", "市场分析师"],
    "rnd": ["软件工程师", "高级工程师", "技术负责人", "系统架构师",
            "测试工程师", "产品经理"],
    "finance": ["会计", "财务分析师", "财务经理", "审计专员", "出纳"],
    "ops": ["运营经理", "运营专员", "供应链专员", "客户成功经理", "行政专员"],
}


# ---------------------------------------------------------------- 1. 员工主数据
def build_employees(n: int) -> pd.DataFrame:
    """员工主数据。

    部门按四部门权重分配（市场/研发/财务/运营），职位从该部门岗位池抽取；
    其余人口学与薪酬属性从 IBM 真实数据的分布中重采样，保持统计特征一致。
    工位坐标不在这里生成 —— 由 step7 按真实平面布局分配（见 office_desks.csv）。
    """
    real = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
    real = real[real["_source"] == "real"]
    base = real.sample(n=n, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)

    dept_keys = RNG.choice(DEPT_KEYS, size=n, p=DEPT_W)
    roles = [RNG.choice(ROLES_BY_DEPT[k]) for k in dept_keys]

    # 职级与薪酬挂钩：越核心的部门/岗位，职级分布略高
    levels = base["JobLevel"].values

    emp = pd.DataFrame({
        "employee_id": [f"E{i:05d}" for i in range(1, n + 1)],
        "name": [f"员工{i:04d}" for i in range(1, n + 1)],
        "department_key": dept_keys,
        "department": [DEPT_CN[k] for k in dept_keys],
        "job_role": roles,
        "job_level": levels,
        "gender": base["Gender"].values,
        "age": base["Age"].values,
        "marital": base["MaritalStatus"].values,
        "monthly_income": base["MonthlyIncome"].values,
        "years_at_company": base["YearsAtCompany"].values,
        "overtime_flag": base["OverTime"].values,
        "attrition": base[TARGET].values,
    })
    return emp


# ---------------------------------------------------------------- 2. 日常行为日志
def build_daily_behavior(emp: pd.DataFrame) -> pd.DataFrame:
    """每人每天一条行为记录：出勤、工时、专注度、协作、沟通量。

    关键：离职倾向高的员工在行为上会呈现
      - 加班变多但有效产出下降（疲惫/心不在焉）
      - 会议/沟通参与度下降（心理脱离）
      - 专注时长碎片化
    """
    n = len(emp)
    dates = pd.date_range(end="2026-10-15", periods=N_DAYS, freq="B")  # 工作日

    # 每位员工的隐状态（决定其行为基线）
    risk_proxy = (
        emp["overtime_flag"].eq("Yes").astype(float) * 0.6
        + (1 - emp["job_level"] / 5.0) * 0.3
        + (emp["monthly_income"] < 4000).astype(float) * 0.3
        + RNG.normal(0, 0.25, n)
    )
    risk_proxy = (risk_proxy - risk_proxy.min()) / (np.ptp(risk_proxy.values) + 1e-9)
    emp_risk = dict(zip(emp["employee_id"], risk_proxy))

    rows = []
    for eid, lvl, ot in zip(emp["employee_id"], emp["job_level"], emp["overtime_flag"]):
        r = emp_risk[eid]
        # 基线
        base_in = 9.0 + RNG.normal(0, .4)
        base_hours = 8.2 + r * 1.2 + (0.6 if ot == "Yes" else 0)
        base_focus = 5.2 - r * 1.6
        base_meet = 3.1 - r * 0.9
        base_msg = 42 - r * 14

        # 随时间演化：风险越高的员工，行为劣化趋势越明显
        trend = np.linspace(0, 1, len(dates)) * (r * 1.6)

        for i, d in enumerate(dates):
            # 迟到倾向随风险上升
            late = RNG.random() < (0.03 + r * 0.12)
            check_in = base_in + trend[i] * 0.35 + (0.5 if late else 0) + RNG.normal(0, .25)
            hours = np.clip(base_hours + trend[i] * 0.4 + RNG.normal(0, .5), 4, 15)
            focus = np.clip(base_focus - trend[i] * 0.5 + RNG.normal(0, .6), 0.2, 8.5)
            meetings = np.clip(base_meet - trend[i] * 0.4 + RNG.normal(0, .6), 0, 8)
            messages = np.clip(base_msg - trend[i] * 8 + RNG.normal(0, 8), 0, 120)
            tasks_done = np.clip(4.5 - r * 1.2 - trend[i] * 0.4 + RNG.normal(0, .8), 0, 12)

            rows.append((
                eid, d.date(), round(float(check_in), 2), late,
                round(float(hours), 2), round(float(focus), 2),
                round(float(meetings), 1), int(messages), int(tasks_done),
                bool(RNG.random() < 0.05 + r * 0.1),      # 是否请假
            ))

    cols = ["employee_id", "date", "check_in_hour", "late_flag", "work_hours",
            "deep_focus_hours", "meeting_count", "messages_sent", "tasks_completed",
            "leave_flag"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- 3. 工作质量
def build_work_quality(emp: pd.DataFrame, beh: pd.DataFrame) -> pd.DataFrame:
    """周频工作质量表，由行为数据聚合派生 —— 体现「行为 -> 质量」的链路。"""
    beh = beh.copy()
    beh["date"] = pd.to_datetime(beh["date"])
    beh["week"] = beh["date"].dt.isocalendar().week.astype(int)

    g = beh.groupby(["employee_id", "week"]).agg(
        avg_hours=("work_hours", "mean"),
        avg_focus=("deep_focus_hours", "mean"),
        total_tasks=("tasks_completed", "sum"),
        late_days=("late_flag", "sum"),
        leave_days=("leave_flag", "sum"),
    ).reset_index()

    # 质量指标由行为派生 + 噪音：专注高、任务多、迟到少 -> 质量高
    g["delivery_rate"] = np.clip(
        0.72 + (g["avg_focus"] - 4.5) * 0.06 + (g["late_days"] * -0.035)
        + RNG.normal(0, 0.04, len(g)), 0.3, 1.0)
    g["rework_count"] = np.clip(
        np.round(3.2 - (g["avg_focus"] - 4.5) * 0.7 + RNG.normal(0, .8, len(g))), 0, None)
    g["defect_rate"] = np.clip(
        0.055 - (g["avg_focus"] - 4.5) * 0.008 + RNG.normal(0, .012, len(g)), 0.0, 0.3)
    g["efficiency_score"] = np.clip(
        72 + (g["avg_focus"] - 4.5) * 6.5 + (g["avg_hours"] - 8.6) * 1.2
        + RNG.normal(0, 4, len(g)), 20, 100)
    g["quality_score"] = np.clip(
        g["delivery_rate"] * 55 + g["efficiency_score"] * 0.45
        - g["rework_count"] * 1.1 - g["defect_rate"] * 60
        + RNG.normal(0, 2.5, len(g)), 20, 100)

    return g[["employee_id", "week", "avg_hours", "avg_focus", "total_tasks",
              "late_days", "leave_days", "delivery_rate", "rework_count",
              "defect_rate", "efficiency_score", "quality_score"]]


# ---------------------------------------------------------------- 4. 谈话报告
TOPICS_POS = ["职业发展路径清晰，主动性提升", "承担了跨部门协作的关键角色",
              "对团队氛围评价积极", "近周期交付质量稳定"]
TOPICS_MID = ["希望调整工作内容，重复性任务偏多", "对当前薪酬涨幅有讨论",
              "反馈会议占用过多专注时间", "期望更明确的晋升时间表"]
TOPICS_RISK = ["明确表达了离职意向，已沟通留任方案", "抱怨长期加班且回报不匹配",
               "提到外部机会，处于观望状态", "对直接上级的管理方式有较强不满",
               "工作投入度明显下降，已连续缺席团队活动"]

TALK_TYPES = ["一对一沟通", "绩效面谈", "离职面谈", "HR访谈", "团队例会发言"]


def build_talk_reports(emp: pd.DataFrame, qual: pd.DataFrame) -> pd.DataFrame:
    """谈话报告：低频（每季度若干次），带主题与情绪倾向。"""
    # 用员工的质量表现决定其谈话内容倾向
    avg_q = qual.groupby("employee_id")["quality_score"].mean()
    rows = []
    rid = 1
    for eid in emp["employee_id"]:
        q = float(avg_q.get(eid, 70))
        # 质量越低，越可能产生风险类谈话
        p_risk = np.clip(0.10 + (72 - q) / 130.0, 0.03, 0.62)
        n_talk = RNG.integers(2, 6)
        for _ in range(n_talk):
            u = RNG.random()
            if u < p_risk:
                topic = RNG.choice(TOPICS_RISK)
                sentiment = RNG.uniform(-1.0, -0.35)
                ttype = RNG.choice(["离职面谈", "HR访谈", "一对一沟通"], p=[.2, .3, .5])
                follow = bool(RNG.random() < 0.75)
            elif u < p_risk + 0.35:
                topic = RNG.choice(TOPICS_MID)
                sentiment = RNG.uniform(-0.35, 0.25)
                ttype = RNG.choice(TALK_TYPES, p=[.4, .3, .0, .2, .1])
                follow = bool(RNG.random() < 0.5)
            else:
                topic = RNG.choice(TOPICS_POS)
                sentiment = RNG.uniform(0.25, 1.0)
                ttype = RNG.choice(TALK_TYPES, p=[.35, .3, .0, .15, .2])
                follow = bool(RNG.random() < 0.35)

            rows.append((
                f"T{rid:06d}", eid, ttype, topic, round(float(sentiment), 3),
                follow, int(RNG.integers(15, 61)),
            ))
            rid += 1

    df = pd.DataFrame(rows, columns=[
        "talk_id", "employee_id", "talk_type", "topic", "sentiment",
        "follow_up_needed", "duration_min"])
    # 随机分布到观测窗口内
    all_dates = pd.date_range(end="2026-10-15", periods=N_DAYS, freq="B")
    df["date"] = pd.to_datetime(RNG.choice(all_dates, len(df))).date
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 5. 测评报告
def build_survey_reports(emp: pd.DataFrame, qual: pd.DataFrame) -> pd.DataFrame:
    """测评报告：低频能力/心理测评，含分维度得分。"""
    avg_q = qual.groupby("employee_id")["quality_score"].mean()
    rows = []
    sid = 1
    for eid, lvl in zip(emp["employee_id"], emp["job_level"]):
        q = float(avg_q.get(eid, 70))
        stress = np.clip(62 - (q - 70) * 0.9 + RNG.normal(0, 9), 5, 98)
        engage = np.clip(70 + (q - 70) * 0.85 + RNG.normal(0, 8), 5, 100)
        satis = np.clip(68 + (q - 70) * 0.75 + RNG.normal(0, 9), 5, 100)
        comp = np.clip(60 + lvl * 6 + RNG.normal(0, 8), 20, 100)
        rows.append((
            f"S{sid:06d}", eid,
            round(float(comp), 1), round(float(engage), 1),
            round(float(satis), 1), round(float(stress), 1),
            round(float(np.clip(engage * .4 + satis * .35 + comp * .25
                                - stress * .2 + RNG.normal(0, 4), 10, 100)), 1),
        ))
        sid += 1

    df = pd.DataFrame(rows, columns=[
        "survey_id", "employee_id", "competency_score", "engagement_score",
        "satisfaction_score", "stress_score", "overall_score"])
    all_dates = pd.date_range(end="2026-10-15", periods=N_DAYS, freq="B")
    df["date"] = pd.to_datetime(RNG.choice(all_dates, len(df))).date
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    N = 1200   # 一个办公地点的人数规模
    print("构建办公场景「完整行为链路」多源数据")
    print("=" * 58)

    emp = build_employees(N)
    print(f"[1/5] employees        {emp.shape}   员工主数据")

    beh = build_daily_behavior(emp)
    print(f"[2/5] daily_behavior   {beh.shape}   {N_DAYS} 天 × {N} 人")

    qual = build_work_quality(emp, beh)
    print(f"[3/5] work_quality     {qual.shape}   {N_WEEKS} 周质量指标")

    talks = build_talk_reports(emp, qual)
    print(f"[4/5] talk_reports     {talks.shape}   谈话报告")

    surveys = build_survey_reports(emp, qual)
    print(f"[5/5] survey_reports   {surveys.shape}   测评报告")

    # 落盘
    emp.to_csv(DATA / "link_employees.csv", index=False, encoding="utf-8-sig")
    beh.to_csv(DATA / "link_daily_behavior.csv", index=False, encoding="utf-8-sig")
    qual.to_csv(DATA / "link_work_quality.csv", index=False, encoding="utf-8-sig")
    talks.to_csv(DATA / "link_talk_reports.csv", index=False, encoding="utf-8-sig")
    surveys.to_csv(DATA / "link_survey_reports.csv", index=False, encoding="utf-8-sig")

    print()
    print("关联键: employee_id 可把 5 张表串成完整链路")
    print(f"  行为日志覆盖员工 {beh['employee_id'].nunique()} 人 / "
          f"{beh['date'].nunique()} 天")
    print(f"  谈话记录 {len(talks)} 条，涉及 {talks['employee_id'].nunique()} 人")
    print(f"  测评报告 {len(surveys)} 条，覆盖 {surveys['employee_id'].nunique()} 人")
    print()
    print("链路方向: 日常行为 -> 工作质量 -> 测评/谈话 -> 状态结论")
    print("输出目录: data/link_*.csv")


if __name__ == "__main__":
    main()
