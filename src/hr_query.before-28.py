#!/usr/bin/env python3
"""员工离职风险 · 本地数据查询工具。

设计原则：数字由 pandas 确定性计算，模型只负责把已经算好的事实
转成自然语言。任何进入回答的数字都必须先出现在 facts 里，模型
无权自己推算，避免小模型编造统计量。
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"

RISK_ORDER = {"低风险": 0, "中风险": 1, "高风险": 2, "极高风险": 3}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


class HRDatabase:
    """42 人真实档案 + 行为/质量/谈话/测评关联表的一次性装载。"""

    def __init__(self):
        self.profiles = _read("employee_profiles.csv")
        self.risk = _read("link_employee_risk.csv")
        self.behavior = _read("link_daily_behavior.csv")
        self.quality = _read("link_work_quality.csv")
        self.talks = _read("link_talk_reports.csv")
        self.surveys = _read("link_survey_reports.csv")

    # ---------- 基础检索 ----------

    def normalize_id(self, text: str) -> str | None:
        """只认明确的工号写法，避免把工位号/小数/职级误当成工号。

        认可：E00003 / e3 / 工号3 / 员工3
        拒绝：F1-SER-07、0.7005、职级5、迟到天数12
        """
        s = str(text).upper()
        m = re.search(r"\bE0*(\d{1,5})\b", s)
        if m:
            return f"E{int(m.group(1)):05d}"
        m = re.search(r"(?:工号|员工号|员工|编号)\s*[:：]?\s*(\d{1,5})", s)
        if m:
            return f"E{int(m.group(1)):05d}"
        # 纯数字输入（整句只有数字）才当作编号
        m = re.fullmatch(r"\s*(\d{1,5})\s*", s)
        if m:
            return f"E{int(m.group(1)):05d}"
        return None

    def lookup_employee(self, token: str) -> pd.Series | None:
        """函数调用入口：接受 E00024 / 24 / 谢心怡 / 客服中心 10 等多种写法。"""
        token = str(token).strip()
        if not token:
            return None
        eid = self.normalize_id(token)
        if eid:
            hit = self.profiles[self.profiles["employee_id"].astype(str).str.upper() == eid]
            if not hit.empty:
                return hit.iloc[0]
        # 姓名：先整体精确匹配，再尝试在文本里找
        names = self.profiles["name"].astype(str)
        exact = self.profiles[names == token]
        if not exact.empty:
            return exact.iloc[0]
        found = self.find_person_in_question(token)
        if found is not None:
            return found
        # 模糊匹配：只说了名或姓的一部分（如「浩然」→ 董浩然）
        # 仅当唯一命中时才采用，多个候选人交回上层让用户确认
        if len(token) >= 2:
            partial = self.profiles[names.str.contains(re.escape(token), na=False)]
            if len(partial) == 1:
                return partial.iloc[0]
            if len(partial) > 1:
                cands = "、".join(partial["name"].astype(str).head(6))
                raise LookupError(f"「{token}」对应多人：{cands}。请说明是哪一位")
        # 兜底：「客服中心 10」这类混合写法，单独取数字当编号
        m = re.search(r"(?<![A-Za-z0-9.])(\d{1,2})(?![0-9.])", token)
        if m and 1 <= int(m.group(1)) <= 42:
            eid = f"E{int(m.group(1)):05d}"
            hit = self.profiles[self.profiles["employee_id"].astype(str).str.upper() == eid]
            if not hit.empty:
                return hit.iloc[0]
        return None

    def find_by_name(self, name: str) -> pd.DataFrame:
        return self.profiles[self.profiles["name"].astype(str).str.contains(name, na=False)]

    def get_employee(self, key: str) -> pd.Series | None:
        key = str(key).strip()
        eid = self.normalize_id(key)
        if eid:
            hit = self.profiles[self.profiles["employee_id"].astype(str).str.upper() == eid]
            if not hit.empty:
                return hit.iloc[0]
        hit = self.find_by_name(key)
        if not hit.empty:
            return hit.iloc[0]
        return None

    # ---------- 确定性统计 ----------

    def overview(self) -> dict:
        df = self.profiles
        counts = df["risk_level"].value_counts().to_dict()
        return {
            "总人数": int(len(df)),
            "部门人数": df["department"].value_counts().to_dict(),
            "风险等级分布": {k: int(counts.get(k, 0)) for k in RISK_ORDER},
            "平均风险分": round(float(df["risk_score"].mean()), 4),
            "风险分范围": [round(float(df["risk_score"].min()), 4),
                       round(float(df["risk_score"].max()), 4)],
            "加班人数": int((df["OverTime"] == "Yes").sum()),
            "历史离职人数": int((df["Attrition"] == "Yes").sum()),
        }

    def roster(self, dept: str | None = None, level: str | None = None,
              extra: bool = True) -> list[dict]:
        """员工名单：返回全部（或按部门/等级筛选）成员的完整关键字段。

        extra=True 时附带薪资、司龄、加班等细节，供「把所有人信息给我」这类请求。
        """
        df = self.profiles
        if dept:
            df = df[df["department"].astype(str) == dept]
        if level:
            df = df[df["risk_level"].astype(str) == level]
        df = df.sort_values("risk_score", ascending=False)
        rows = []
        for r in df.itertuples():
            item = {"工号": r.employee_id, "姓名": r.name, "部门": r.department,
                    "岗位": r.JobRole, "职级": int(r.JobLevel),
                    "月薪": int(r.MonthlyIncome), "司龄": int(r.YearsAtCompany),
                    "加班": r.OverTime,
                    "风险分": round(float(r.risk_score), 4), "等级": r.risk_level}
            if extra:
                item.update({
                    "工作满意度": int(r.JobSatisfaction),
                    "绩效评分": int(r.PerformanceRating),
                    "迟到天数": int(r.late_days),
                    "平均交付率": round(float(r.delivery_rate), 4),
                    "测评_压力": round(float(r.stress_score), 1),
                    "测评_满意度": round(float(r.satisfaction_score), 1),
                    "潜力": r.potential,
                })
            rows.append(item)
        return rows

    def top_risk(self, n: int = 5, level: str | None = None) -> list[dict]:
        df = self.profiles
        if level:
            df = df[df["risk_level"] == level]
        df = df.sort_values("risk_score", ascending=False).head(n)
        return [
            {"工号": r.employee_id, "姓名": r.name, "部门": r.department,
             "风险分": round(float(r.risk_score), 4), "等级": r.risk_level}
            for r in df.itertuples()
        ]

    def department_stats(self, dept: str | None = None) -> list[dict]:
        df = self.profiles
        if dept is None:
            groups = [g for _, g in df.groupby("department")]
        else:
            g = df[df["department"] == dept]
            groups = [g] if not g.empty else []
        out = []
        for g in groups:
            if g.empty:
                continue
            out.append({
                "部门": str(g["department"].iloc[0]),
                "人数": int(len(g)),
                "平均风险分": round(float(g["risk_score"].mean()), 4),
                "高风险及以上": int(g["risk_level"].isin(["高风险", "极高风险"]).sum()),
                "加班人数": int((g["OverTime"] == "Yes").sum()),
            })
        return sorted(out, key=lambda x: -x["平均风险分"])

    def attrition_rate(self) -> dict:
        df = self.profiles
        total = len(df)
        left = int((df["Attrition"] == "Yes").sum())
        by_ot = df.groupby("OverTime")["Attrition"].apply(
            lambda s: round(float((s == "Yes").mean()), 4)).to_dict()
        by_dept = df.groupby("department")["Attrition"].apply(
            lambda s: round(float((s == "Yes").mean()), 4)).to_dict()
        return {
            "样本人数": total,
            "已离职人数": left,
            "历史离职率": round(left / total, 4) if total else 0.0,
            "按加班分组的离职率": {str(k): v for k, v in by_ot.items()},
            "按部门分组的离职率": {str(k): v for k, v in by_dept.items()},
        }

    def employee_detail(self, emp: pd.Series) -> dict:
        eid = emp["employee_id"]
        detail = {
            "工号": eid,
            "姓名": emp["name"],
            "部门": emp["department"],
            "楼层": int(emp["floor"]) if pd.notna(emp["floor"]) else None,
            "工位": emp["desk_id"],
            "职级": emp["JobLevel"],
            "岗位": emp["JobRole"],
            "月薪": int(emp["MonthlyIncome"]),
            "司龄": int(emp["YearsAtCompany"]),
            "加班": emp["OverTime"],
            "工作满意度": int(emp["JobSatisfaction"]),
            "风险分": round(float(emp["risk_score"]), 4),
            "风险等级": emp["risk_level"],
            "绩效评分": emp["PerformanceRating"],
        }
        q = self.quality[self.quality["employee_id"] == eid]
        if not q.empty:
            detail["近周平均工时"] = round(float(q["avg_hours"].mean()), 2)
            detail["近周平均专注时长"] = round(float(q["avg_focus"].mean()), 2)
            detail["平均交付率"] = round(float(q["delivery_rate"].mean()), 4)
        s = self.surveys[self.surveys["employee_id"] == eid]
        if not s.empty:
            detail["测评_压力"] = round(float(s["stress_score"].mean()), 2)
            detail["测评_投入度"] = round(float(s["engagement_score"].mean()), 2)
            detail["测评_满意度"] = round(float(s["satisfaction_score"].mean()), 2)
        t = self.talks[self.talks["employee_id"] == eid]
        if not t.empty:
            detail["谈话次数"] = int(len(t))
            detail["待跟进谈话"] = int(t["follow_up_needed"].astype(str).str.lower().isin(["true", "yes", "1"]).sum())
        b = self.behavior[self.behavior["employee_id"] == eid]
        if not b.empty:
            detail["迟到天数"] = int(b["late_flag"].astype(bool).sum())
        return detail

    def risk_factors(self, emp: pd.Series) -> list[dict]:
        raw = emp.get("risk_factor_count")
        factors = []
        prof_json = DATA / "employee_profiles.json"
        if prof_json.exists():
            for item in json.loads(prof_json.read_text(encoding="utf-8")):
                if item.get("employee_id") == emp["employee_id"]:
                    factors = item.get("risk", {}).get("top_factors", []) or []
                    break
        return factors

    def extract_factors(self, emp: pd.Series) -> list[dict]:
        """优先从预计算的 top_factors 取；没有则给出规则化提示。"""
        factors = self.risk_factors(emp)
        if factors:
            return factors
        hints = []
        if emp["OverTime"] == "Yes":
            hints.append({"factor": "加班", "direction": "+", "detail": "加班记录为 Yes"})
        if int(emp["JobSatisfaction"]) <= 2:
            hints.append({"factor": "工作满意度偏低", "direction": "+",
                          "detail": f"工作满意度 {emp['JobSatisfaction']}/4"})
        if int(emp["YearsAtCompany"]) <= 1:
            hints.append({"factor": "司龄短", "direction": "+",
                          "detail": f"司龄 {emp['YearsAtCompany']} 年"})
        return hints

    # ---------- 干预建议 ----------

    # 风险因子 -> 可执行干预动作。键用因子名片段匹配，避免改动上游文案后失效。
    INTERVENTIONS = {
        "加班": [
            "本周内排查该员工近 4 周加班来源，砍掉可转移的重复性任务",
            "与直属主管确认排班，把连续加班转为轮值，避免疲劳累积",
            "评估是否属于编制缺口，必要时补充临时人力",
        ],
        "迟到": [
            "单独沟通迟到原因：区分通勤、家庭照护、隐性缺勤三种情况",
            "若为通勤问题，可申请弹性上下班时间（如 10:00 到岗）",
            "若为隐性缺勤，重点排查工作倦怠而非考勤纪律",
        ],
        "晋升": [
            "明确下一职级的晋升标准与时间窗口，给出书面条件",
            "安排一次职业发展面谈，让员工看到 12 个月内的成长路径",
            "若无晋升空间，考虑横向调岗或增加职责范围",
        ],
        "满意度": [
            "安排一次不记录的低压力一对一沟通，聚焦不满来源",
            "检查近 3 个月的谈话记录中反复出现的话题",
            "确认薪酬是否显著低于同职级同岗位水平",
        ],
        "司龄": [
            "新人前 6 个月是流失高峰，指定导师并做 30/60/90 天回访",
            "加速融入：明确近期可交付的小任务，建立成就感",
        ],
        "出差": [
            "核对出差频次是否超出岗位合理范围",
            "长期出差人员安排轮换，避免连续多月外派",
        ],
        "期權": [],
        "期权": [
            "期权是留任因素，注意归属节点前后的沟通，避免到期即流失",
        ],
        "职级": [
            "高职级员工流失成本高，优先做留任面谈而非通用关怀",
        ],
    }

    def interventions(self, emp: pd.Series) -> list[dict]:
        """按该员工实际风险因子生成干预建议，只覆盖命中的因子。"""
        factors = self.extract_factors(emp)
        out = []
        for f in factors:
            if f.get("direction") != "+":
                continue  # 只针对推高风险的因子
            name = str(f.get("factor", ""))
            actions = []
            for key, acts in self.INTERVENTIONS.items():
                if key in name and acts:
                    actions = acts
                    break
            out.append({"因子": name, "说明": f.get("detail", ""),
                        "建议": actions or ["结合实际情况与该员工当面确认优先级"]})
        return out

    # ---------- 问答路由 ----------

    def find_person_in_question(self, question: str) -> pd.Series | None:
        """在整句里识别具体员工：工号优先，其次姓名，最后从粘贴的档案文本里找。"""
        eid = self.normalize_id(question)
        if eid:
            hit = self.profiles[self.profiles["employee_id"].astype(str).str.upper() == eid]
            if not hit.empty:
                return hit.iloc[0]
        # 姓名匹配：句首命中优先级最高（贴档案时姓名在最前），其次按长度降序
        names = self.profiles["name"].astype(str)
        head = re.match(r"^[\s,，、]*(.{1,4}?)[\s,，、]", question)
        if head:
            cand = names == head.group(1)
            if cand.any():
                return self.profiles[cand].iloc[0]
        for name in sorted(names, key=len, reverse=True):
            if name and name in question:
                hit = self.profiles[names == name]
                if not hit.empty:
                    return hit.iloc[0]
        return None

    def answer(self, question: str) -> dict:
        """返回 {intent, facts, structured, employees}，全部数字由本函数算出。"""
        q = str(question).strip()

        # 0. 问候 / 空泛输入：不要倒全表
        if re.fullmatch(r"[\s,，.。!！?？]*(你好|您好|hi|hello|在吗|嗨)[\s,，.。!！?？]*",
                        q, flags=re.I):
            return {"intent": "greeting",
                    "facts": ["这是一个查询终端，直接问具体问题即可，例如："
                              "「冯嘉豪的风险情况」「查看离职率」「最高风险前 5 名」"],
                    "structured": {}, "employees": []}

        # 1. 具体员工（最高优先级：点名问人就只看这个人）
        emp = self.find_person_in_question(q)
        if emp is not None:
            detail = self.employee_detail(emp)
            factors = self.extract_factors(emp)

            # 1a. 问的是「怎么干预 / 有什么办法」-> 只给风险因子与对症动作
            if re.search(r"干预|建议|怎么办|怎么做|措施|办法|对策|留住|挽留|降低风险|"
                         r"改善|应对|方案|如何", q):
                plans = self.interventions(emp)
                lines = [f"{detail['姓名']}（{detail['工号']}，{detail['部门']}）"
                         f"风险分 {detail['风险分']}，{detail['风险等级']}"]
                if plans:
                    lines.append(f"推高风险的因素共 {len(plans)} 项：")
                    for p in plans:
                        lines.append(f"因子「{p['因子']}」：{p['说明']}")
                        for a in p["建议"]:
                            lines.append(f"  建议行动：{a}")
                else:
                    lines.append("该员工没有命中任何推高风险的规则因子，"
                                 "维持常规关怀与季度回访即可")
                return {"intent": "intervention", "facts": lines,
                        "structured": {"employee": detail, "plans": plans},
                        "employees": [detail]}

            facts = [f"{k}：{v}" for k, v in detail.items()]
            for f in factors:
                facts.append(f"风险因子 {f['factor']}（{f['direction']}）：{f['detail']}")
            return {"intent": "employee", "facts": facts,
                    "structured": detail, "employees": [detail]}

        # 2. 离职率
        if re.search(r"离职率|离职人数|走人|流失率|attrition", q, re.I):
            facts = self.attrition_rate()
            lines = [f"样本 {facts['样本人数']} 人，已离职 {facts['已离职人数']} 人，"
                     f"历史离职率 {facts['历史离职率']:.2%}"]
            for k, v in facts["按加班分组的离职率"].items():
                lines.append(f"加班={k} 的员工离职率 {v:.2%}")
            for k, v in facts["按部门分组的离职率"].items():
                lines.append(f"部门 {k} 离职率 {v:.2%}")
            return {"intent": "attrition_rate", "facts": lines,
                    "structured": facts, "employees": []}

        # 3. 风险最高的人
        if re.search(r"最高风险|风险最高|最危险|top|排名|榜单|谁最", q, re.I):
            n = 5
            m = re.search(r"(\d+)\s*[个名位人]", q)
            if m:
                n = min(int(m.group(1)), 42)
            items = self.top_risk(n)
            lines = [f"第{i+1}名 {x['姓名']}（{x['工号']}，{x['部门']}）"
                     f"风险分 {x['风险分']}，{x['等级']}" for i, x in enumerate(items)]
            return {"intent": "top_risk", "facts": lines,
                    "structured": items, "employees": items}

        # 4. 部门统计
        if re.search(r"部门|研发|市场|运营|财务|客服|人力|高管", q):
            dept = None
            for d in self.profiles["department"].unique():
                if str(d) in q:
                    dept = d
                    break
            stats = self.department_stats(dept)
            lines = [f"{s['部门']}：{s['人数']} 人，平均风险分 {s['平均风险分']}，"
                     f"高风险及以上 {s['高风险及以上']} 人，加班 {s['加班人数']} 人"
                     for s in stats]
            return {"intent": "department", "facts": lines,
                    "structured": stats, "employees": []}

        # 5. 统计概览 / 风险分布
        if re.search(r"多少人|总人数|全部员工|所有员工|统计|概览|分布|风险等级|整体", q):
            ov = self.overview()
            lines = [f"总人数 {ov['总人数']} 人",
                     f"平均风险分 {ov['平均风险分']}，范围 {ov['风险分范围'][0]} 到 {ov['风险分范围'][1]}"]
            for k, v in ov["风险等级分布"].items():
                lines.append(f"{k} {v} 人")
            for k, v in ov["部门人数"].items():
                lines.append(f"部门 {k} {v} 人")
            lines.append(f"加班员工 {ov['加班人数']} 人，历史离职 {ov['历史离职人数']} 人")
            return {"intent": "overview", "facts": lines,
                    "structured": ov, "employees": []}

        # 6. 兜底：明确提示能查什么，不倒全表（全表用 /list）
        ov = self.overview()
        lines = [f"没有识别出你想查的对象。当前库内共 {ov['总人数']} 人，"
                 f"平均风险分 {ov['平均风险分']}",
                 "可以直接问：某个员工姓名/工号、「研发部情况」、「查看离职率」、"
                 "「最高风险前 5 名」、「<员工>有什么干预手段」",
                 "想看全部名单请输入 /list"]
        return {"intent": "fallback", "facts": lines,
                "structured": ov, "employees": []}


# ---------- 模型润色 ----------

def ollama_available(host: str = OLLAMA_HOST) -> bool:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=3):
            return True
    except Exception:
        return False


def narrate(question: str, facts: list[str], model: str = DEFAULT_MODEL,
            host: str = OLLAMA_HOST, timeout: int = 120) -> str:
    """只允许复述 facts，禁止模型自行计算新数字。"""
    fact_block = "\n".join(f"- {f}" for f in facts)
    prompt = (
        "你是一名 HR 数据分析助手。下面是从公司数据库中查出的准确事实。\n"
        f"{fact_block}\n\n"
        f"用户问题：{question}\n\n"
        "请依据上面的事实，用中文简洁回答（不超过 5 句）。"
        "严格禁止编造事实中不存在的数字、姓名或部门。"
        "如果事实不足以回答，就直接说明缺少哪类数据。\n回答："
    )
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="员工离职风险本地查询工具")
    ap.add_argument("question", nargs="*", help="自然语言问题，如：冯嘉豪的风险情况")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-model", action="store_true", help="只输出事实，不调用模型")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    db = HRDatabase()
    q = " ".join(args.question).strip()
    if not q:
        q = input("请输入问题（例如：全部员工风险分布）：").strip()
    result = db.answer(q)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"[意图] {result['intent']}")
    print("[事实]")
    for line in result["facts"]:
        print(f"  - {line}")

    if not args.no_model:
        print("\n[模型解读]")
        if ollama_available():
            try:
                print(narrate(q, result["facts"], model=args.model))
            except Exception as exc:
                print(f"模型调用失败（事实仍有效）：{exc}")
        else:
            print("Ollama 未启动，跳过。事实数据已在上方给出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
