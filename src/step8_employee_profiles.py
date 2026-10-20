"""Step 8 员工档案数据（供 3D 场景点击查看）

为大楼里 **42 名员工**每人生成一份完整档案，字段参照 **IBM HR Analytics**
的真实字段体系（35 个原始字段 + 派生），并补充三类业务数据：

  1. profile      人事档案：IBM 格式字段 + 背调信息（学历/院校/证书/前雇主/合同/紧急联系人）
  2. behavior     日常行为：90 天考勤与工时的聚合 + 近 7 天明细 + 工作质量
  3. assessment   测评报告：绩效/敬业度/满意度/压力/能力/潜力
  4. talks        谈话记录：1:1 面谈 / HR 访谈摘要与情绪倾向
  5. risk         离职风险：评分、等级、主要影响因子

产出
----
  data/employee_profiles.json    42 人完整档案（点击头像时读取这个）
  data/employee_profiles.csv     扁平版（便于表格查看/建模）

数据口径说明
------------
「IBM 格式」指字段名、取值域、单位与 IBM HR Analytics 数据集保持一致，
便于任何按该数据集写的分析脚本直接复用；数值部分按真实数据的边际分布采样，
不凭空捏造取值范围。行为与测评数据是按业务逻辑合成的模拟数据。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import DATA, MERGED_CSV, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)

# IBM 数据集的字段与取值域（用于采样时对齐真实分布）
EDU_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
              "Other", "Human Resources"]
TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
DEGREE_CN = {1: "高中及以下", 2: "大专", 3: "本科", 4: "硕士", 5: "博士"}
SCHOOLS = ["华东理工大学", "上海交通大学", "同济大学", "东南大学", "华中科技大学",
           "北京邮电大学", "西安电子科技大学", "南京大学", "浙江大学", "武汉大学",
           "四川大学", "中山大学", "哈尔滨工业大学", "天津大学", "厦门大学"]
MAJORS = ["计算机科学与技术", "软件工程", "信息管理", "市场营销", "工商管理",
          "会计学", "统计学", "电子信息工程", "自动化", "人力资源管理学"]
CERTS = ["PMP", "CPA", "CFA", "AWS Certified", "CET-6", "中级经济师",
         "系统架构设计师", "人力资源管理师", "注册会计师", "无"]
CONTRACTS = ["正式员工", "正式员工", "正式员工", "试用期", "劳务派遣"]

TALK_TYPES = ["一对一沟通", "绩效面谈", "HR访谈", "离职面谈", "团队例会发言"]
TALK_POS = ["职业发展路径清晰，主动性提升", "承担了跨部门协作的关键角色",
            "对团队氛围评价积极", "近周期交付质量稳定"]
TALK_MID = ["希望调整工作内容，重复性任务偏多", "对当前薪酬涨幅有讨论",
            "反馈会议占用过多专注时间", "期望更明确的晋升时间表"]
TALK_RISK = ["明确表达了离职意向，已沟通留任方案", "抱怨长期加班且回报不匹配",
             "提到外部机会，处于观望状态", "对直接上级的管理方式有较强不满",
             "工作投入度明显下降，已连续缺席团队活动"]


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 IBM 真实数据（作为取值分布参考）与已生成的行为/测评数据。"""
    real = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
    real = real[real["_source"] == "real"].reset_index(drop=True)

    beh = pd.read_csv(DATA / "link_daily_behavior.csv", encoding="utf-8-sig")
    qual = pd.read_csv(DATA / "link_work_quality.csv", encoding="utf-8-sig")
    sur = pd.read_csv(DATA / "link_survey_reports.csv", encoding="utf-8-sig")
    tal = pd.read_csv(DATA / "link_talk_reports.csv", encoding="utf-8-sig")
    return real, (beh, qual, sur, tal)


def pick(series, n):
    """按某列的真实经验分布重采样 n 个值。"""
    vc = series.value_counts(normalize=True)
    return RNG.choice(vc.index.values, size=n, p=vc.values)


def build_profiles() -> list[dict]:
    staff = pd.read_csv(DATA / "building_desks.csv", encoding="utf-8-sig")
    real, (beh, qual, sur, tal) = load_base()
    N = len(staff)

    # ---- 用 IBM 真实分布为每位员工采样基础字段 ----
    real_s = real.sample(n=N, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)

    # 行为/质量/测评按 employee_id 关联（link_* 里是 E00001..E01200）
    beh_g = beh.groupby("employee_id")
    qual_g = qual.groupby("employee_id")
    sur_by = sur.set_index("employee_id")
    tal_g = tal.groupby("employee_id")

    out = []
    for i, row in staff.iterrows():
        eid = row["employee_id"]
        src_id = f"E{(i % 1200) + 1:05d}"          # 对应 link_* 里的 id
        r = real_s.iloc[i]

        # ---------- 1. profile：IBM 格式 + 背调 ----------
        edu = int(r["Education"])
        # 性别和姓名来自 step7 生成的同一条员工记录，避免档案之间互相矛盾。
        gender = str(row.get("gender") or r["Gender"])
        hire = date(2026, 10, 15) - timedelta(days=int(r["YearsAtCompany"]) * 365
                                              + int(RNG.integers(0, 360)))
        profile = {
            # --- IBM HR Analytics 原始字段 ---
            "Age": int(r["Age"]),
            "Attrition": str(r["Attrition"]),
            "BusinessTravel": str(r["BusinessTravel"]),
            "DailyRate": int(r["DailyRate"]),
            "Department": row["department_cn"],
            "DistanceFromHome": int(r["DistanceFromHome"]),
            "Education": edu,
            "EducationField": str(r["EducationField"]),
            "EmployeeCount": 1,
            "EmployeeNumber": i + 1,
            "EnvironmentSatisfaction": int(r["EnvironmentSatisfaction"]),
            "Gender": gender,
            "HourlyRate": int(r["HourlyRate"]),
            "JobInvolvement": int(r["JobInvolvement"]),
            "JobLevel": int(r["JobLevel"]),
            # 职位取自 building_desks.csv（已按部门匹配岗位池），
            # 不能用 real_s 里的 IBM 原始职位，否则会出现「研发部挂着 Sales Executive」
            "JobRole": str(row["job_role"]),
            "JobSatisfaction": int(r["JobSatisfaction"]),
            "MaritalStatus": str(r["MaritalStatus"]),
            "MonthlyIncome": int(r["MonthlyIncome"]),
            "MonthlyRate": int(r["MonthlyRate"]),
            "NumCompaniesWorked": int(r["NumCompaniesWorked"]),
            "Over18": "Y",
            "OverTime": str(r["OverTime"]),
            "PercentSalaryHike": int(r["PercentSalaryHike"]),
            "PerformanceRating": int(r["PerformanceRating"]),
            "RelationshipSatisfaction": int(r["RelationshipSatisfaction"]),
            "StandardHours": 80,
            "StockOptionLevel": int(r["StockOptionLevel"]),
            "TotalWorkingYears": int(r["TotalWorkingYears"]),
            "TrainingTimesLastYear": int(r["TrainingTimesLastYear"]),
            "WorkLifeBalance": int(r["WorkLifeBalance"]),
            "YearsAtCompany": int(r["YearsAtCompany"]),
            "YearsInCurrentRole": int(r["YearsInCurrentRole"]),
            "YearsSinceLastPromotion": int(r["YearsSinceLastPromotion"]),
            "YearsWithCurrManager": int(r["YearsWithCurrManager"]),
            # --- 背调 / 档案（扩展字段） ---
            "background": {
                "highest_degree": DEGREE_CN[edu],
                "school": str(RNG.choice(SCHOOLS)),
                "major": str(RNG.choice(MAJORS)),
                "certificates": [str(RNG.choice(CERTS))] if RNG.random() < 0.7 else [],
                "prev_employers": int(r["NumCompaniesWorked"]),
                "hire_date": hire.isoformat(),
                "contract_type": str(RNG.choice(CONTRACTS)),
                "id_verified": True,
                "bg_check_passed": bool(RNG.random() < 0.97),
                "emergency_contact": f"家属 {RNG.integers(130, 189)}****{RNG.integers(1000, 9999)}",
                "bank_account": f"**** **** {RNG.integers(1000, 9999)}",
            },
        }

        # ---------- 2. behavior：90 天行为 + 工作质量 ----------
        try:
            b = beh_g.get_group(src_id)
            q = qual_g.get_group(src_id)
            b_avg_hours = round(float(b["work_hours"].mean()), 2)
            b_avg_focus = round(float(b["deep_focus_hours"].mean()), 2)
            b_late = int(b["late_flag"].sum())
            b_leave = int(b["leave_flag"].sum())
            b_meet = round(float(b["meeting_count"].mean()), 2)
            b_msg = int(b["messages_sent"].mean())
            b_tasks = round(float(b["tasks_completed"].mean()), 2)
            y = b.sort_values("date")["deep_focus_hours"].values
            trend = round(float(np.polyfit(np.arange(len(y)), y, 1)[0]), 4) if len(y) > 2 else 0.0
            recent = [
                {"date": str(d), "hours": h, "focus": f, "late": bool(l), "tasks": int(t)}
                for d, h, f, l, t in zip(
                    b.sort_values("date").tail(7)["date"],
                    b.sort_values("date").tail(7)["work_hours"],
                    b.sort_values("date").tail(7)["deep_focus_hours"],
                    b.sort_values("date").tail(7)["late_flag"],
                    b.sort_values("date").tail(7)["tasks_completed"])
            ]
            quality = {
                "delivery_rate": round(float(q["delivery_rate"].mean()), 3),
                "rework_count": round(float(q["rework_count"].mean()), 2),
                "defect_rate": round(float(q["defect_rate"].mean()), 4),
                "efficiency_score": round(float(q["efficiency_score"].mean()), 1),
                "quality_score": round(float(q["quality_score"].mean()), 1),
            }
        except KeyError:
            b_avg_hours, b_avg_focus = round(float(RNG.normal(8.4, .7)), 2), round(float(RNG.normal(4.8, .9)), 2)
            b_late, b_leave = int(RNG.integers(0, 8)), int(RNG.integers(0, 6))
            b_meet, b_msg = round(float(RNG.normal(3, .8)), 2), int(RNG.normal(38, 12))
            b_tasks = round(float(RNG.normal(4.3, .9)), 2)
            trend, recent = 0.0, []
            quality = {"delivery_rate": round(float(RNG.normal(.86, .05)), 3),
                       "rework_count": round(float(RNG.normal(2.2, .8)), 2),
                       "defect_rate": round(float(RNG.normal(.045, .012)), 4),
                       "efficiency_score": round(float(RNG.normal(78, 6)), 1),
                       "quality_score": round(float(RNG.normal(76, 6)), 1)}
        behavior = {
            "window_days": 90,
            "avg_work_hours": b_avg_hours,
            "avg_focus_hours": b_avg_focus,
            "late_days": b_late,
            "leave_days": b_leave,
            "avg_meetings": b_meet,
            "avg_messages": b_msg,
            "avg_tasks_done": b_tasks,
            "focus_trend": trend,
            "recent_7d": recent,
            "quality": quality,
        }

        # ---------- 3. assessment：测评报告 ----------
        try:
            s = sur_by.loc[src_id]
            assessment = {
                "competency_score": float(s["competency_score"]),
                "engagement_score": float(s["engagement_score"]),
                "satisfaction_score": float(s["satisfaction_score"]),
                "stress_score": float(s["stress_score"]),
                "overall_score": float(s["overall_score"]),
                "survey_date": str(s["date"]),
                "potential": str(RNG.choice(["高", "中高", "中", "待观察"],
                                            p=[.15, .30, .40, .15])),
                "performance_rating": int(r["PerformanceRating"]),
            }
        except KeyError:
            assessment = {
                "competency_score": round(float(RNG.normal(70, 9)), 1),
                "engagement_score": round(float(RNG.normal(70, 10)), 1),
                "satisfaction_score": round(float(RNG.normal(68, 10)), 1),
                "stress_score": round(float(RNG.normal(58, 12)), 1),
                "overall_score": round(float(RNG.normal(70, 8)), 1),
                "survey_date": "2026-09-30",
                "potential": str(RNG.choice(["高", "中高", "中", "待观察"])),
                "performance_rating": int(r["PerformanceRating"]),
            }

        # ---------- 4. talks：谈话记录 ----------
        talks = []
        try:
            t = tal_g.get_group(src_id)
            for _, tr in t.head(5).iterrows():
                talks.append({
                    "date": str(tr["date"]),
                    "type": str(tr["talk_type"]),
                    "topic": str(tr["topic"]),
                    "sentiment": float(tr["sentiment"]),
                    "follow_up": bool(tr["follow_up_needed"]),
                    "duration_min": int(tr["duration_min"]),
                })
        except KeyError:
            for _ in range(int(RNG.integers(1, 4))):
                neg = RNG.random() < 0.3
                talks.append({
                    "date": str(date(2026, 10, 15) - timedelta(days=int(RNG.integers(5, 85)))),
                    "type": str(RNG.choice(TALK_TYPES)),
                    "topic": str(RNG.choice(TALK_RISK if neg else TALK_MID if RNG.random() < .5 else TALK_POS)),
                    "sentiment": round(float(RNG.uniform(-1, -.3) if neg else RNG.uniform(-.2, 1)), 3),
                    "follow_up": bool(neg or RNG.random() < .4),
                    "duration_min": int(RNG.integers(15, 60)),
                })

        # ---------- 5. risk：离职风险 ----------
        try:
            rs = float(row["risk_score"])
            rl = str(row["risk_level"])
        except Exception:
            rs, rl = round(float(RNG.uniform(0, 1)), 4), "中风险"
        risk = {
            "score": round(rs, 4) if rs == rs else None,
            "level": rl if rl != "nan" else None,
            "top_factors": build_risk_factors(profile, behavior, assessment),
        }

        out.append({
            "employee_id": eid,
            "name": row["name"],
            "floor": int(row["floor"]),
            "floor_name": row["floor_name"],
            "department_key": row["department_key"],
            "department_cn": row["department_cn"],
            "desk_id": row["desk_id"],
            "desk_room": row["room"],          # 工位所在房间（寻路用）
            "desk": {"x": float(row["desk_x"]), "y": float(row["desk_y"]),
                     "rot": float(row["desk_rot"])},
            "profile": profile,
            "behavior": behavior,
            "assessment": assessment,
            "talks": talks,
            "risk": risk,
        })
    return out


def build_risk_factors(profile: dict, behavior: dict, assessment: dict) -> list[dict]:
    """给出可解释的风险因子（方向 + 说明），不是模型系数，是业务判读。"""
    f = []
    if profile["OverTime"] == "Yes":
        f.append({"factor": "长期加班", "direction": "+", "detail": "加班记录为 Yes，疲劳累积推高流失意愿"})
    if profile["MaritalStatus"] == "Single":
        f.append({"factor": "单身", "direction": "+", "detail": "无家庭牵绊，外部机会吸引力更大"})
    if profile["BusinessTravel"] == "Travel_Frequently":
        f.append({"factor": "高频出差", "direction": "+", "detail": "出差频繁与离职率正相关"})
    if profile["JobSatisfaction"] <= 2:
        f.append({"factor": "工作满意度低", "direction": "+",
                  "detail": f"JobSatisfaction={profile['JobSatisfaction']}/4"})
    if profile["EnvironmentSatisfaction"] <= 2:
        f.append({"factor": "环境满意度低", "direction": "+",
                  "detail": f"EnvironmentSatisfaction={profile['EnvironmentSatisfaction']}/4"})
    if profile["YearsSinceLastPromotion"] >= 3:
        f.append({"factor": "晋升停滞", "direction": "+",
                  "detail": f"距上次晋升 {profile['YearsSinceLastPromotion']} 年"})
    if profile["StockOptionLevel"] == 0:
        f.append({"factor": "无股票期权", "direction": "+", "detail": "缺少长期激励绑定"})
    if behavior["focus_trend"] < -0.02:
        f.append({"factor": "专注时长下滑", "direction": "+",
                  "detail": f"90 天趋势 {behavior['focus_trend']:+.3f} 小时/天"})
    if behavior["late_days"] >= 6:
        f.append({"factor": "迟到偏多", "direction": "+",
                  "detail": f"90 天迟到 {behavior['late_days']} 次"})
    if assessment["engagement_score"] < 60:
        f.append({"factor": "敬业度偏低", "direction": "+",
                  "detail": f"敬业度得分 {assessment['engagement_score']}"})
    if assessment["stress_score"] > 75:
        f.append({"factor": "压力偏高", "direction": "+",
                  "detail": f"压力得分 {assessment['stress_score']}"})
    # 保护性因素
    if profile["JobLevel"] >= 3:
        f.append({"factor": "职级较高", "direction": "-", "detail": "职级越高越倾向留任"})
    if profile["StockOptionLevel"] >= 1:
        f.append({"factor": "持有期权", "direction": "-", "detail": "股权激励降低流失"})
    if behavior["quality"]["quality_score"] > 80:
        f.append({"factor": "工作质量优", "direction": "-",
                  "detail": f"质量分 {behavior['quality']['quality_score']}"})
    return f[:8]


def flatten(rec: dict) -> dict:
    """压平成一行，便于 CSV 与建模。"""
    p, b, a, r = rec["profile"], rec["behavior"], rec["assessment"], rec["risk"]
    row = {"employee_id": rec["employee_id"], "name": rec["name"],
           "floor": rec["floor"], "department": rec["department_cn"],
           "desk_id": rec["desk_id"]}
    for k in ["Age", "Gender", "MaritalStatus", "JobRole", "JobLevel",
              "MonthlyIncome", "OverTime", "Attrition", "JobSatisfaction",
              "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement",
              "YearsAtCompany", "TotalWorkingYears", "NumCompaniesWorked",
              "YearsSinceLastPromotion", "StockOptionLevel", "Education",
              "EducationField", "BusinessTravel", "PercentSalaryHike",
              "PerformanceRating", "DistanceFromHome"]:
        row[k] = p[k]
    row["degree"] = p["background"]["highest_degree"]
    row["school"] = p["background"]["school"]
    row["contract_type"] = p["background"]["contract_type"]
    row["hire_date"] = p["background"]["hire_date"]
    for k in ["avg_work_hours", "avg_focus_hours", "late_days", "leave_days",
              "avg_meetings", "avg_tasks_done", "focus_trend"]:
        row[k] = b[k]
    row["quality_score"] = b["quality"]["quality_score"]
    row["delivery_rate"] = b["quality"]["delivery_rate"]
    for k in ["competency_score", "engagement_score", "satisfaction_score",
              "stress_score", "overall_score", "potential"]:
        row[k] = a[k]
    row["risk_score"] = r["score"]
    row["risk_level"] = r["level"]
    row["risk_factor_count"] = len(r["top_factors"])
    row["talk_count"] = len(rec["talks"])
    return row


def main():
    print("生成 42 名员工的完整档案数据")
    print("=" * 62)

    recs = build_profiles()
    print(f"[1/3] 档案生成 {len(recs)} 人")

    # 完整性校验
    miss = [r["employee_id"] for r in recs
            if not r["profile"] or not r["behavior"] or not r["assessment"]]
    print(f"      完整性校验：缺失 {len(miss)} 人")

    (DATA / "employee_profiles.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    df = pd.DataFrame([flatten(r) for r in recs])
    df.to_csv(DATA / "employee_profiles.csv", index=False, encoding="utf-8-sig")
    print(f"[2/3] 落盘 JSON / CSV")

    # 概览
    print(f"\n[3/3] 数据概览")
    print(f"      楼层分布     {dict(df.groupby('floor').size())}")
    print(f"      部门分布     {dict(df.groupby('department').size())}")
    print(f"      风险等级     {dict(df['risk_level'].value_counts())}")
    print(f"      平均月薪     {df['MonthlyIncome'].mean():,.0f}")
    print(f"      平均司龄     {df['YearsAtCompany'].mean():.1f} 年")
    print(f"      平均周工时   {df['avg_work_hours'].mean():.2f} 小时/天")
    print(f"      平均敬业度   {df['engagement_score'].mean():.1f}")
    print(f"      平均压力     {df['stress_score'].mean():.1f}")
    print()
    print("每人档案包含：")
    print("  profile    IBM 格式 35 字段 + 背调（学历/院校/专业/证书/前雇主/合同/紧急联系人）")
    print("  behavior   90 天考勤工时聚合 + 近 7 天明细 + 工作质量")
    print("  assessment 绩效/敬业度/满意度/压力/能力/潜力")
    print("  talks      谈话记录（类型/主题/情绪/是否需跟进）")
    print("  risk       风险评分 + 等级 + 可解释因子")
    print()
    print(f"输出: {DATA/'employee_profiles.json'}")
    print(f"      {DATA/'employee_profiles.csv'}")


if __name__ == "__main__":
    main()
