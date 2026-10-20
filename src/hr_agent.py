#!/usr/bin/env python3
"""员工数据 Agent：把查询能力作为函数开放给模型，由模型自行决定查什么。

与关键词路由的区别：
  路由模式  —— 程序判断意图，模型只润色（快、稳定，但问法超出预设就失败）
  Agent 模式 —— 模型看工具清单自行选参数，多轮调用（灵活，能处理没预设过的问法）

数字仍然全部来自 pandas，模型只负责决定「去查谁」和「怎么说」。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hr_query import HRDatabase, OLLAMA_HOST, DEFAULT_MODEL, LOCAL_MODEL  # noqa: E402
from llm_client import (deepseek_ready, deepseek_chat, DEEPSEEK_MODEL,
                        DEEPSEEK_REASONER_MODEL)  # noqa: E402

import re  # noqa: E402

AGENT_MODEL = os.environ.get("HR_AGENT_MODEL", DEEPSEEK_MODEL if deepseek_ready() else DEFAULT_MODEL)
OLLAMA_FALLBACK_MODEL = os.environ.get("HR_OLLAMA_MODEL", "gpt-oss:120b-cloud")
MAX_ROUNDS = 4          # 最多几轮工具调用，防止死循环
NUM_CTX = 16384         # 上下文窗口。默认 4096 装不下 28 人全量名单（约 5000 tokens）
TOOL_CHAR_BUDGET = 12000  # 单个工具结果回送模型的字符上限，超出则压缩


def _fn(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


TOOLS_SPEC = [
    _fn("get_employee", "查询单个员工的主档：部门、岗位、月薪、司龄、加班、"
        "风险分与风险等级、以及推高风险的因子。只需基础档案时用；要做综合诊断用 get_employee_360。",
        {"query": {"type": "string",
                   "description": "员工姓名（如 谢心怡、冯嘉豪），或员工编号（如 E00024 或 24）。"
                                  "只传姓名或编号本身，不要带部门名。"},
         "department": {"type": "string",
                        "description": "可选。当用户说「客服中心的某某」时传入部门名，"
                                       "用于校验该员工确实属于这个部门。可选项：客服中心、研发部、"
                                       "市场部、运营部、财务部、人力与行政、高管办公区"}},
        ["query"]),
    _fn("get_interventions", "针对某个员工给出留任干预建议，会按他的风险因子逐条给动作。"
        "用户问「有什么办法」「怎么留人」「怎么干预」时用这个。",
        {"query": {"type": "string", "description": "员工姓名或工号"},
         "department": {"type": "string",
                        "description": "可选。用户限定了部门时传入，用于校验归属"}},
        ["query"]),
    _fn("list_employees", "列出员工名单及其关键信息（工号、姓名、部门、岗位、月薪、司龄、加班、"
        "风险分与等级，以及满意度、绩效、迟到、交付率等）。用于「列出所有员工」「把名单发我」"
        "「客服中心有哪些人」「高风险的都是谁」这类需要成批人员信息的请求。"
        "不传任何参数即返回当前库全部人员。",
        {"department": {"type": "string",
                        "description": "可选，限定部门。可选项：客服中心、研发部、市场部、"
                                       "运营部、财务部、人力与行政、高管办公区"},
         "level": {"type": "string",
                   "description": "可选，限定风险等级：低风险、中风险、高风险、极高风险"},
         "brief": {"type": "boolean",
                   "description": "true 时只返回工号/姓名/部门/风险分/等级这几个精简字段，"
                                  "适合只要名单不要细节的场景"}},
        []),
    _fn("get_department", "查询某个部门整体情况：人数、平均风险分、高风险人数、加班人数，"
        "并附带该部门员工名单。部门名可选值：客服中心、研发部、市场部、运营部、财务部、"
        "人力与行政、高管办公区。不传 name 则返回所有部门的对比。这只是入口，"
        "做风险归因时继续用 get_employee_360 / query_dataset 查行为和谈话。",
        {"name": {"type": "string", "description": "部门名称，不传则返回全部门对比"}},
        []),
    _fn("get_department_workforce", "跨表部门画像：把员工主档、逐日行为、工作质量、测评和谈话记录按部门合并，"
        "一次返回人数、平均风险、高风险占比、加班、历史离职率、平均日工时/专注、迟到、交付率、质量分、"
        "压力、满意度、谈话次数和需跟进数。部门对比、优先级排序、找最该干预的部门时优先用这个。",
        {}, []),
    _fn("get_attrition_rate", "查询历史离职率：整体离职率，以及按加班情况、按部门分组的离职率。",
        {}, []),
    _fn("get_top_risk", "按风险分从高到低列出员工，用于「风险最高的是谁」「前几名」这类问题。",
        {"n": {"type": "integer", "description": "返回前几名，默认 5"}},
        []),
    _fn("get_overview", "公司整体概览：总人数、平均风险分、各风险等级人数分布、各部门人数。"
        "用户问「整体情况」「多少人」「风险分布」时用。",
        {}, []),
    _fn("get_data_catalog", "列出当前可查询的全部数据集、字段、行数，以及模型和办公层数据概览。"
        "遇到没预设过的分析问题、或需要知道有哪些字段时先用这个。",
        {}, []),
    _fn("query_dataset", "通用数据查询：对 profiles/risk/behavior/quality/talks/surveys/desks "
        "任意数据集做字段筛选、分组、聚合、排序。适合自定义分析，如「研发部加班与风险的关系」"
        "「按部门比较交付率和压力」「找出迟到多且满意度低的人」。大表请用 group_by + metrics。"
        "常用字段：behavior=work_hours/deep_focus_hours/meeting_count/messages_sent/tasks_completed/"
        "late_flag/leave_flag；quality=avg_hours/avg_focus/delivery_rate/rework_count/defect_rate/"
        "efficiency_score/quality_score；surveys=competency_score/engagement_score/satisfaction_score/"
        "stress_score/overall_score；talks=talk_type/topic/sentiment/follow_up_needed/duration_min/date；"
        "risk=risk_score/risk_level/feature_set；profiles=department/JobRole/risk_score/risk_level/OverTime。"
        "join_profiles=true 可把 behavior/quality/surveys 关联到部门、岗位和风险等级。",
        {"dataset": {"type": "string",
                     "description": "profiles / risk / behavior / quality / talks / surveys / desks"},
         "filters": {"type": "object",
                     "description": "字段过滤，如 {\"employee_id\":\"E00001\"} 或 "
                                    "{\"risk_score\":{\"min\":0.6}}；支持 min/max/in/contains"},
         "columns": {"type": "array", "items": {"type": "string"},
                     "description": "返回字段，省略则返回全部字段"},
         "group_by": {"type": "string", "description": "分组字段"},
         "metrics": {"type": "object",
                     "description": "聚合，如 {\"work_hours\":\"mean\",\"late_flag\":\"sum\"}"},
         "sort_by": {"type": "string", "description": "排序字段"},
         "descending": {"type": "boolean", "description": "是否降序，默认 true"},
         "limit": {"type": "integer", "description": "返回行数，默认 20，最大 50"},
         "join_profiles": {"type": "boolean",
                           "description": "true 时按 employee_id 关联主档，可直接按 department/JobRole/risk_level 分组；分析跨表关系时用"}},
        ["dataset"]),
    _fn("get_employee_360", "员工全景分析：档案 + 风险因子 + 行为汇总 + 工作质量 + 测评 + "
        "最近谈话 + 干预建议。需要综合判断某位员工时优先用这个。",
        {"query": {"type": "string", "description": "员工姓名或工号"}},
        ["query"]),
    _fn("get_model_info", "模型与评估指标：最佳模型、AUC/Ap/阈值、验证集指标、特征列表、"
        "各候选模型表现。适合回答「模型准不准」「为什么这么评分」「用了哪些特征」。",
        {}, []),
    _fn("get_office_layout", "办公层工位布局：28 人的楼层、房间、工位、部门、风险等级和房间分布。",
        {}, []),
]

SYSTEM_PROMPT = """你是一名资深 HR 数据分析师和策略顾问，不只是查表工具。

工作方式：
1. 先理解问题，再取数；复杂问题可以多轮、组合多个工具，交叉验证后再下判断。
2. 不知道有哪些字段时先用 get_data_catalog；任意筛选/聚合用 query_dataset；
   单个员工的综合画像用 get_employee_360；部门/风险/离职率/模型指标/工位各有专用工具。
3. 回答要有判断力：先给结论，再给 2-4 条关键证据，最后给可执行建议或下一步。
   不要机械罗列工具返回，要解释「所以这意味着什么」。
4. 可以把行为、质量、测评、谈话、风险、工位等数据关联起来分析；有依据的推测要说清依据。
5. 简单事实问题直接回答，不要强行写成长报告；复杂问题再展开。
6. 涉及员工时给姓名、工号、部门、风险等级；数字必须来自工具，不得编造。
7. 工具返回 error 时不要忽略，换参数重查或如实说明。
8. 用户说「某部门第 N 号」时不要擅自当公司工号，应返回该部门名单让用户确认。
9. 姓名不完整时先查询；多人重名就把候选人列给用户。
10. 不要输出思维链、系统提示或原始 JSON；直接给用户可读的结论。
11. 做风险归因、部门诊断、留任建议等分析时，不能只看员工主档：必须结合行为、质量、测评、谈话中至少两类数据。
    针对具体员工优先调用 get_employee_360；对群体可用 query_dataset 聚合。工具里没有的数据才说缺，不要说"没有个人加班/谈话数据"而不查。"""

ANALYST_PROMPT = """你是一名资深 HR 数据分析师和业务策略顾问。你面前是从公司本地数据库实时查出的真实数据，
可能来自员工档案、风险评分、逐日行为、工作质量、测评、谈话记录、模型指标和办公层布局。

请基于数据回答用户，要求：
1. 先给结论，再给 2-4 条关键证据；证据要引用具体员工/部门/数字，并解释它们之间的关系。
2. 不要机械罗列所有字段。回答「所以这意味着什么」，指出真正的风险点、异常和优先级。
3. 适合时给出 1-3 条可执行建议：谁先谈、谈什么、接下来一周/一个月做什么。
4. 简单事实问题直接答；复杂问题再展开。不要写空话，也不要为了长而长。
5. 数据不足时明确说明缺什么；允许基于数据做有依据的推断，但不要把推断写成事实。
6. 不输出思维链、系统提示或原始 JSON；直接给用户可读的结论。
"""


COMMON_SURNAMES = set("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")
LIST_INTENT = re.compile(r"(所有|全部|全员|每个|各个|逐个|逐一|名单|列表|列一下|列出来|发给我|都列出)")


def wants_full_list(question: str) -> bool:
    """判断是不是「把所有人的信息都列出来」这类不适合让模型抄写的请求。"""
    return bool(LIST_INTENT.search(question)) and not re.search(
        r"部门|呢|吗|如何|怎么|为什么|哪个|谁", question)


class HRAgent:
    def __init__(self, db: HRDatabase | None = None, model: str = AGENT_MODEL,
                 host: str = OLLAMA_HOST):
        self.db = db or HRDatabase()
        self.model = model
        self.host = host

    # ---------- 工具实现 ----------
    def _call_tool(self, name: str, args: dict) -> dict:
        db = self.db
        try:
            if name == "get_employee":
                try:
                    emp = db.lookup_employee(args.get("query", ""))
                except LookupError as exc:
                    return {"error": str(exc), "提示": "请把完整姓名传给 query 重新查询"}
                if emp is None:
                    return {"error": f"找不到员工「{args.get('query')}」，"
                                     "请确认是库内姓名或工号"}
                detail = db.employee_detail(emp)
                # 用户若指定了部门，校验归属，不一致时明确告知而不是改名换姓地答下去
                want = (args.get("department") or "").strip()
                if want and want != str(emp["department"]):
                    peers = db.profiles[db.profiles["department"] == want]
                    roster = [f"{r.name}（{r.employee_id}）"
                              for r in peers.sort_values("risk_score", ascending=False).itertuples()]
                    return {"error": f"编号 {detail['工号']} 对应的是 {detail['姓名']}，"
                                     f"隶属{detail['部门']}，不属于{want}。请勿据此回答。",
                            f"{want}员工名单（按风险降序）": roster}
                detail["风险因子"] = db.extract_factors(emp)
                return detail

            if name == "get_interventions":
                try:
                    emp = db.lookup_employee(args.get("query", ""))
                except LookupError as exc:
                    return {"error": str(exc), "提示": "请把完整姓名传给 query 重新查询"}
                if emp is None:
                    return {"error": f"找不到员工「{args.get('query')}」"}
                want = (args.get("department") or "").strip()
                if want and want != str(emp["department"]):
                    return {"error": f"{emp['name']} 隶属{emp['department']}，不属于{want}，"
                                     "不能据此给建议。请用户确认具体姓名。"}
                detail = db.employee_detail(emp)
                return {"员工": f"{detail['姓名']}（{detail['工号']}，{detail['部门']}）",
                        "风险分": detail["风险分"], "风险等级": detail["风险等级"],
                        "干预方案": db.interventions(emp)}

            if name == "list_employees":
                dept = (args.get("department") or "").strip() or None
                lvl = (args.get("level") or "").strip() or None
                if dept and dept not in set(db.profiles["department"].astype(str)):
                    valid = sorted(set(db.profiles["department"].astype(str)))
                    return {"error": f"没有部门「{dept}」，可选：{'、'.join(valid)}"}
                levels = set(db.profiles["risk_level"].astype(str))
                if lvl and lvl not in levels:
                    return {"error": f"没有等级「{lvl}」，可选：{'、'.join(sorted(levels))}"}
                rows = db.roster(dept=dept, level=lvl,
                                 extra=not bool(args.get("brief")))
                if not rows:
                    return {"error": "该筛选条件下没有员工"}
                head = f"共 {len(rows)} 人"
                if dept:
                    head += f"（{dept}）"
                if lvl:
                    head += f"（{lvl}）"
                return {"说明": head + "，按风险分降序", "员工": rows}

            if name == "get_department":
                name_arg = (args.get("name") or "").strip()
                if not name_arg:
                    return {"部门对比": db.department_stats()}
                valid = set(db.profiles["department"].astype(str))
                if name_arg not in valid:
                    return {"error": f"没有部门「{name_arg}」，可选：{'、'.join(sorted(valid))}"}
                stats = db.department_stats(name_arg)
                people = db.roster(dept=name_arg, extra=False)
                return {"统计": stats,
                        "员工名单（按风险降序）":
                            [f"{p['姓名']}（{p['工号']}，{p['等级']}，{p['风险分']}）"
                             for p in people]}

            if name == "get_department_workforce":
                return {"部门跨表画像": db.department_workforce()}

            if name == "get_attrition_rate":
                return db.attrition_rate()

            if name == "get_top_risk":
                n = int(args.get("n") or 5)
                return {"前N名": db.top_risk(max(1, min(n, len(db.profiles))))}

            if name == "get_overview":
                return db.overview()

            if name == "get_data_catalog":
                return db.data_catalog()

            if name == "query_dataset":
                return db.query_dataset(
                    dataset=str(args.get("dataset") or ""),
                    filters=args.get("filters") or {},
                    columns=args.get("columns") or None,
                    group_by=args.get("group_by") or None,
                    metrics=args.get("metrics") or {},
                    sort_by=args.get("sort_by") or None,
                    descending=bool(args.get("descending", True)),
                    limit=int(args.get("limit") or 20),
                    join_profiles=bool(args.get("join_profiles", False)),
                )

            if name == "get_employee_360":
                try:
                    return db.employee_360(str(args.get("query") or ""))
                except LookupError as exc:
                    return {"error": str(exc)}

            if name == "get_model_info":
                return db.model_info()

            if name == "get_office_layout":
                return db.office_layout()

            return {"error": f"未知工具 {name}"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _shrink(result: dict, payload: str) -> str:
        """结果超出字符预算时，逐步降级：先去细节、再截断条数，保证不撑爆上下文。"""
        emp = result.get("员工")
        if isinstance(emp, list) and emp:
            slim = [{k: e.get(k) for k in ("姓名", "工号", "部门", "风险分", "等级")}
                    for e in emp]
            out = {"说明": result.get("说明", ""), "员工": slim,
                   "注意": "明细已省略，如需某人的完整档案请用 get_employee 单独查"}
            text = json.dumps(out, ensure_ascii=False, default=str)
            if len(text) <= TOOL_CHAR_BUDGET:
                return text
            # 仍然过长：只留前 20 条 + 汇总
            out["员工"] = slim[:20]
            out["注意"] += f"；共 {len(emp)} 人，此处仅列前 20 人"
            return json.dumps(out, ensure_ascii=False, default=str)
        # 非名单类结果：直接硬截断，保住可解析性
        return payload[:TOOL_CHAR_BUDGET] + "…（结果过长已截断）"

    # ---------- 与 ollama 交互 ----------
    def _chat(self, messages: list[dict], timeout: int = 180) -> dict:
        def post(model: str) -> dict:
            body = json.dumps({
                "model": model, "messages": messages, "stream": False,
                "think": False,
                "tools": TOOLS_SPEC,
                "options": {"temperature": 0.0, "num_predict": 500, "num_ctx": NUM_CTX},
            }).encode()
            req = urllib.request.Request(
                self.host.rstrip("/") + "/api/chat", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            return post(self.model)
        except Exception:
            if self.model != LOCAL_MODEL:
                # 云端不可用时保留能力，退回本机模型
                self.model = LOCAL_MODEL
                return post(self.model)
            raise

    def _full_list_answer(self) -> dict:
        rows = self.db.roster(extra=False)
        lines = [f"{i:>2}. {p['姓名']:<4} {p['工号']}  {p['部门']:<6} "
                 f"{p['风险分']:.4f}  {p['等级']}" for i, p in enumerate(rows, 1)]
        answer = (f"当前 {len(rows)} 人展示团队，按风险分从高到低：\n" + "\n".join(lines)
                  + "\n\n（名单由数据库直接生成，姓名与分数可逐条核对）")
        return {"answer": answer,
                "tool_calls": [{"name": "list_employees", "args": {}, "result": {"人数": len(rows)}}],
                "rounds": 1, "model": self.model}

    @staticmethod
    def _trace_context(trace: list[dict], budget: int = 18000) -> str:
        chunks = []
        for i, item in enumerate(trace, 1):
            payload = json.dumps({"工具": item.get("name"), "参数": item.get("args"),
                                  "结果": item.get("result")}, ensure_ascii=False, default=str)
            if len(payload) > 5500:
                payload = payload[:5500] + "…（已截断）"
            chunks.append(f"[查询 {i}]\n{payload}")
        text = "\n\n".join(chunks)
        return text[:budget]

    def _synthesize(self, question: str, trace: list[dict], draft: str = "") -> dict:
        scope = f"当前数据库：{len(self.db.profiles)} 人办公展示团队（与 3D 办公层一致）。"
        context = self._trace_context(trace)
        user = (
            f"用户问题：{question}\n\n"
            f"{scope}\n\n"
            f"数据库查询结果：\n{context}\n\n"
            "请基于以上真实数据回答。先结论，再关键证据和关联，最后给建议；"
            "简单问题不要写成报告。不要提到工具调用过程。"
        )
        messages = [{"role": "system", "content": ANALYST_PROMPT},
                    {"role": "user", "content": user}]
        # 先用推理模型做一次综合；失败/为空则用 chat 兜底
        for model, temp in ((DEEPSEEK_REASONER_MODEL, 0.5), (DEEPSEEK_MODEL, 0.45)):
            try:
                reply = deepseek_chat(messages, model=model, temperature=temp,
                                      max_tokens=1100, timeout=120)
                content = (reply.get("message", {}).get("content") or "").strip()
                if content:
                    return {"answer": content, "model": model}
            except Exception:
                continue
        if draft.strip():
            return {"answer": draft.strip(), "model": DEEPSEEK_MODEL}
        return {"answer": "已查到数据，但云端模型暂时没有生成回答，请稍后重试。",
                "model": "deepseek"}

    def _run_deepseek(self, question: str, verbose: bool = False) -> dict:
        tool_specs = TOOLS_SPEC
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT +
             f"\n\n当前查询库：{len(self.db.profiles)} 人办公展示团队（与 3D 办公层一致）。"},
            {"role": "user", "content": question},
        ]
        trace = []
        draft = ""
        rounds = 0
        for rnd in range(1, MAX_ROUNDS + 1):
            rounds = rnd
            reply = deepseek_chat(messages, tools=tool_specs, model=DEEPSEEK_MODEL,
                                  temperature=0.2, max_tokens=800, timeout=120)
            msg = reply.get("message", {})
            content = (msg.get("content") or "").strip()
            if content:
                draft = content
            calls = msg.get("tool_calls") or []
            if not calls:
                if verbose:
                    print(f"  [第 {rnd} 轮] DeepSeek 准备回答")
                break

            clean_calls = []
            for idx, tc in enumerate(calls):
                fn = tc.get("function", {}) or {}
                clean_calls.append({
                    "id": tc.get("id") or f"call_{rnd}_{idx}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""),
                                 "arguments": fn.get("arguments") or "{}"},
                })
            messages.append({"role": "assistant", "content": content, "tool_calls": clean_calls})
            if verbose:
                print(f"  [第 {rnd} 轮] DeepSeek 请求 {len(clean_calls)} 个工具")
            for tc in clean_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn["arguments"] or "{}")
                except Exception:
                    args = {}
                result = self._call_tool(name, args)
                trace.append({"name": name, "args": args, "result": result})
                payload = json.dumps(result, ensure_ascii=False, default=str)
                if len(payload) > TOOL_CHAR_BUDGET:
                    payload = self._shrink(result, payload) if result.get("员工") else payload[:TOOL_CHAR_BUDGET] + "…（已截断）"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": payload})
                if verbose:
                    print(f"    - {name}({args}) -> {len(payload)} chars")
        if trace:
            synth = self._synthesize(question, trace, draft)
            return {"answer": synth["answer"], "tool_calls": trace,
                    "rounds": rounds, "model": synth["model"]}
        return {"answer": draft or "已查询，但模型没有返回可读结果。",
                "tool_calls": [], "rounds": rounds, "model": DEEPSEEK_MODEL}

    def run(self, question: str, verbose: bool = False) -> dict:
        """返回 {answer, tool_calls:[{name,args,result}], rounds}"""
        if wants_full_list(question) and not re.search(r"[A-Za-z]|\d", re.sub(r"\d+\s*[人名位]", "", question)):
            return self._full_list_answer()
        if deepseek_ready():
            try:
                return self._run_deepseek(question, verbose=verbose)
            except Exception as exc:
                if verbose:
                    print(f"  [DeepSeek 失败，回退 Ollama] {type(exc).__name__}: {str(exc)[:120]}")
        self.model = OLLAMA_FALLBACK_MODEL
        return self._run_ollama(question, verbose=verbose)

    def _run_ollama(self, question: str, verbose: bool = False) -> dict:
        """本地/ Ollama 云兜底路径。"""
        # 名单类请求：28 条明细让 8b 抄写会出现漏人和重复，直接由程序渲染更可靠
        if wants_full_list(question) and not re.search(r"[A-Za-z]|\d", re.sub(r"\d+\s*[人名位]", "", question)):
            rows = self.db.roster(extra=False)
            lines = [f"{i:>2}. {p['姓名']:<4} {p['工号']}  {p['部门']:<6} "
                     f"{p['风险分']:.4f}  {p['等级']}"
                     for i, p in enumerate(rows, 1)]
            answer = (f"公司共 {len(rows)} 人，按风险分从高到低：\n" + "\n".join(lines)
                      + "\n\n（此名单由数据库直接生成，非模型复述，姓名与分数可逐条核对）")
            return {"answer": answer, "tool_calls":
                    [{"name": "list_employees", "args": {}, "result": {"人数": len(rows)}}],
                    "rounds": 1, "model": self.model}

        messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\n当前查询库：{len(self.db.profiles)} 人办公展示团队（与 3D 办公层一致）。"}, {"role": "user", "content": question}]
        trace = []

        for rnd in range(1, MAX_ROUNDS + 1):
            reply = self._chat(messages)
            msg = reply.get("message", {})
            calls = msg.get("tool_calls") or []

            if not calls:
                answer = (msg.get("content") or "").strip()
                return {"answer": answer or "（模型没有给出回答）",
                        "tool_calls": trace, "rounds": rnd, "model": self.model}

            # 记录助手这一轮的工具调用，再逐个执行并回填结果
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": calls})
            for tc in calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                result = self._call_tool(name, raw_args)
                trace.append({"name": name, "args": raw_args, "result": result})
                if verbose:
                    print(f"  [工具 {rnd}] {name}({raw_args})")
                payload = json.dumps(result, ensure_ascii=False, default=str)
                if len(payload) > TOOL_CHAR_BUDGET:
                    payload = self._shrink(result, payload)
                    if verbose:
                        print(f"         （结果较长，已压缩至 {len(payload)} 字符回送模型）")
                messages.append({"role": "tool", "content": payload})
            if verbose:
                print(f"  [第 {rnd} 轮] 模型请求了 {len(calls)} 个工具，已返回结果，继续…")

        return {"answer": "（调用轮次过多，已中止。可缩小问题范围后重试）",
                "tool_calls": trace, "rounds": MAX_ROUNDS, "model": self.model}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="员工数据 Agent（模型自主调用工具）")
    ap.add_argument("question", nargs="*", help="自然语言问题")
    ap.add_argument("--model", default=AGENT_MODEL)
    ap.add_argument("--verbose", action="store_true", help="显示工具调用过程")
    args = ap.parse_args(argv)

    q = " ".join(args.question).strip() or "公司整体情况怎么样"
    agent = HRAgent(model=args.model)
    out = agent.run(q, verbose=args.verbose)
    print("\n回答：", out["answer"])
    print(f"\n（调用 {len(out['tool_calls'])} 个工具，{out['rounds']} 轮）")


if __name__ == "__main__":
    main()
