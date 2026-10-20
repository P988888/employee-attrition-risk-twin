#!/usr/bin/env python3
"""员工数据 Agent：把查询能力作为函数开放给模型，由模型自行决定查什么。

与关键词路由的区别：
  路由模式  —— 程序判断意图，模型只润色（快、稳定，但问法超出预设就失败）
  Agent 模式 —— 模型看工具清单自行选参数，多轮调用（灵活，能处理没预设过的问法）

数字仍然全部来自 pandas，模型只负责决定「去查谁」和「怎么说」。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hr_query import HRDatabase, OLLAMA_HOST  # noqa: E402

import re  # noqa: E402

AGENT_MODEL = "dengcao/Qwen3-8B:Q5_K_M"
MAX_ROUNDS = 4          # 最多几轮工具调用，防止死循环
NUM_CTX = 16384         # 上下文窗口。默认 4096 装不下 42 人全量名单（约 7140 tokens）
TOOL_CHAR_BUDGET = 12000  # 单个工具结果回送模型的字符上限，超出则压缩


def _fn(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


TOOLS_SPEC = [
    _fn("get_employee", "查询单个员工的完整档案：部门、岗位、月薪、司龄、加班、"
        "风险分与风险等级、以及推高风险的因子。用户提到具体人名或工号时用这个。",
        {"query": {"type": "string",
                   "description": "员工姓名（如 谢心怡、冯嘉豪），或员工编号（如 E00024 或 24）。"
                                  "编号范围 1-42。只传姓名或编号本身，不要带部门名。"},
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
        "不传任何参数即返回全部 42 人。",
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
        "人力与行政、高管办公区。不传 name 则返回所有部门的对比。",
        {"name": {"type": "string", "description": "部门名称，不传则返回全部门对比"}},
        []),
    _fn("get_attrition_rate", "查询历史离职率：整体离职率，以及按加班情况、按部门分组的离职率。",
        {}, []),
    _fn("get_top_risk", "按风险分从高到低列出员工，用于「风险最高的是谁」「前几名」这类问题。",
        {"n": {"type": "integer", "description": "返回前几名，默认 5"}},
        []),
    _fn("get_overview", "公司整体概览：总人数、平均风险分、各风险等级人数分布、各部门人数。"
        "用户问「整体情况」「多少人」「风险分布」时用。",
        {}, []),
]

SYSTEM_PROMPT = """你是这家公司的人力资源数据助手，可以调用工具查询真实员工数据。

规则：
1. 必须先用工具取数据，再回答。不要凭空说任何数字。
2. 工具返回什么就用什么，不要自己加总、换算或推测没给出的数值。
3. 回答用中文，简洁直接，一般 2-5 句。不要把工具的原始 JSON 贴给用户。
4. 涉及员工时要说出姓名和风险等级；涉及数字时保留工具给出的精度。
5. 用户要「所有员工」「名单」「全部信息」时，必须用 list_employees，不要回答「没有这个功能」。
6. 工具返回里出现 error 字段时，绝对不能忽略它去编答案。要么换参数重查，要么把错误如实告诉用户。
7. 用户问题里若出现部门名，务必把部门传给工具的 department 参数做校验。
8. 用户说「某部门的第 N 号/第 N 个人」属于歧义表达，此时优先返回该部门名单让用户确认，不要擅自把 N 当成公司工号。
9. 用户只给了姓名的一部分（如「浩然」）时，先用 get_employee 试；若工具提示多人重名，把候选人列给用户确认。
10. 不要输出思考过程，直接给结论。"""


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
                                     "请确认是库内姓名或 1-42 的工号"}
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

            if name == "get_attrition_rate":
                return db.attrition_rate()

            if name == "get_top_risk":
                n = int(args.get("n") or 5)
                return {"前N名": db.top_risk(max(1, min(n, 42)))}

            if name == "get_overview":
                return db.overview()

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
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "think": False,            # Qwen3 关闭思考，否则把 token 全花在推理上
            "tools": TOOLS_SPEC,
            "options": {"temperature": 0.0, "num_predict": 500, "num_ctx": NUM_CTX},
        }).encode()
        req = urllib.request.Request(
            self.host.rstrip("/") + "/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def run(self, question: str, verbose: bool = False) -> dict:
        """返回 {answer, tool_calls:[{name,args,result}], rounds}"""
        # 名单类请求：42 条明细让 8b 抄写会出现漏人和重复，直接由程序渲染更可靠
        if wants_full_list(question) and not re.search(r"[A-Za-z]|\d", question.replace("42", "")):
            rows = self.db.roster(extra=False)
            lines = [f"{i:>2}. {p['姓名']:<4} {p['工号']}  {p['部门']:<6} "
                     f"{p['风险分']:.4f}  {p['等级']}"
                     for i, p in enumerate(rows, 1)]
            answer = (f"公司共 {len(rows)} 人，按风险分从高到低：\n" + "\n".join(lines)
                      + "\n\n（此名单由数据库直接生成，非模型复述，姓名与分数可逐条核对）")
            return {"answer": answer, "tool_calls":
                    [{"name": "list_employees", "args": {}, "result": {"人数": len(rows)}}],
                    "rounds": 1, "model": self.model}

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
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
