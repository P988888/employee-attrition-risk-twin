#!/usr/bin/env python3
"""员工离职风险 · 交互式查询终端。

用法：
    python3 src/hr_chat.py              # 默认 qwen2.5:1.5b
    python3 src/hr_chat.py --model qwen3:8b
    python3 src/hr_chat.py --no-model   # 只用查表，不加载模型（最快）

特点：数字来自 data/*.csv 的确定性查询，模型只把事实转成自然语言。
支持上下文追问：问完"冯嘉豪"再说"他的加班情况"，会自动沿用上一位员工。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hr_query import (  # noqa: E402
    DEFAULT_MODEL,
    HRDatabase,
    narrate,
    ollama_available,
)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   员工离职风险 · 本地数据查询终端                         ║
║   数据源：data/employee_profiles.csv 等 7 张表（28 人展示团队）   ║
╚══════════════════════════════════════════════════════════╝
 试试这些问法：
   查冯嘉豪              研发部情况            查看离职率
   全部员工              最高风险前 5 名        各部门平均风险
   谢心怡有什么干预手段   研发和市场哪个风险高
 上下文追问：问过某人后，直接说「他的加班情况」「他该怎么留」
 其他命令：/list 全员  /help 帮助  /model 当前模型  /quit 退出
 启动参数：--route 快速路由模式   --no-model 只用查表
"""


def hr_line(title: str = "") -> None:
    print("\n" + "─" * 58)
    if title:
        print(f" {title}")
        print("─" * 58)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="交互式员工数据查询终端")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-model", action="store_true", help="只用查表，不调用模型")
    ap.add_argument("--route", action="store_true",
                    help="用关键词路由模式（快、确定），默认用 Agent 模式")
    args = ap.parse_args(argv)

    print(BANNER)
    db = HRDatabase()
    print(f"已载入档案：{len(db.profiles)} 人")

    has_model = False if args.no_model else ollama_available()
    agent = None
    if args.no_model:
        print("模式：仅查表（不调用模型，结果 100% 来自 CSV）")
    elif not has_model:
        print("Ollama 未启动，自动降级为仅查表模式。")
        print("启动模型后再运行本工具即可获得自然语言解读。")
    elif args.route:
        print(f"模式：关键词路由 + {args.model} 润色（快，问法需在预设内）")
    else:
        from hr_agent import AGENT_MODEL, HRAgent
        agent = HRAgent(db=db, model=AGENT_MODEL)
        print(f"模式：Agent + {AGENT_MODEL}（模型自行选择查询，能处理灵活问法）")
        print("      加 --route 切回快速路由模式，加 --no-model 只用查表")

    history: list[tuple[str, str]] = []  # 仅用于上下文提示，不进入模型对话
    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not raw:
            continue
        if raw in ("/quit", "/exit", "/bye", "q"):
            print("再见。")
            return 0
        if raw in ("/help", "/?"):
            print(BANNER.strip())
            continue
        if raw == "/model":
            mode = "仅查表" if args.no_model else ("Agent" if agent else "路由")
            print(f"模型：{args.model if not agent else agent.model}"
                  f"（{'在线' if ollama_available() else '离线'}）  模式：{mode}")
            continue
        if raw == "/list":
            hr_line("全部员工（按风险分降序）")
            for i, x in enumerate(db.top_risk(999), 1):
                print(f"  {i:>2}. {x['姓名']:<6} {x['工号']}  {x['部门']:<7} "
                      f"{x['风险分']:.4f}  {x['等级']}")
            continue

        # ---------- Agent 模式：模型自主调用工具 ----------
        if agent is not None:
            # 上下文：若本句没点名任何人，把上一位员工的信息附上供模型参考
            question = raw
            if history and not db.find_person_in_question(raw):
                prev_name = history[-1][1]
                if re.search(r"他的|她的|他呢|她呢|这个人|该员工|此人|刚才|上面", raw):
                    question = f"（上一位提到的是 {prev_name}）{raw}"
            hr_line("模型正在查询数据…")
            try:
                out = agent.run(question, verbose=True)
                print("\n" + out["answer"])
            except Exception as exc:
                print(f"（Agent 调用失败：{exc}）")
                print("可改用 --route 模式重试。")
                continue
            # 记录本次涉及的人，供下轮指代（兼容不同工具返回结构）
            for tc in out["tool_calls"]:
                r = tc.get("result") or {}
                if not isinstance(r, dict):
                    continue
                nm = r.get("姓名")
                if not nm and isinstance(r.get("员工"), str):
                    nm = r["员工"].split("（")[0].strip()
                if nm:
                    history.append(("employee", nm))
            continue

        # ---------- 路由模式 ----------
        question = raw
        last = history[-1][1] if history else None
        if last and not db.find_person_in_question(raw):
            if re.search(r"他的|她的|他呢|她呢|这个人|该员工|此人", raw):
                question = re.sub(r"^(他的|她的|他呢|她呢|这个人|该员工|此人)\s*", f"{last}的", raw)
            elif re.search(r"干预|建议|怎么办|怎么做|措施|办法|对策|挽留|降低风险|改善|应对|方案",
                           raw):
                question = f"{last}{raw}"

        result = db.answer(question)
        if result["intent"] == "employee" and result["employees"]:
            history.append(("employee", result["employees"][0]["姓名"]))

        hr_line(f"[{result['intent']}] 数据依据")
        for line in result["facts"]:
            print(f"  · {line}")

        if has_model:
            print()
            try:
                print(narrate(question, result["facts"], model=args.model))
            except Exception as exc:
                print(f"（模型解读失败：{exc}；上面的数据仍然有效）")


if __name__ == "__main__":
    raise SystemExit(main())
