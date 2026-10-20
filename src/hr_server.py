#!/usr/bin/env python3
"""大屏 + 查询 API 一体的本地服务。

静态文件从 output/dashboard 提供（与原 http.server 行为一致），
额外挂上 /api/* 供前端查数据库：

    GET  /api/health
    GET  /api/ask?q=冯嘉豪的风险&model=1
    GET  /api/employees
    GET  /api/overview
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "output" / "dashboard"
sys.path.insert(0, str(ROOT / "src"))

from hr_query import HRDatabase, DEFAULT_MODEL, ollama_available, narrate  # noqa: E402
from llm_client import deepseek_ready, provider_info  # noqa: E402

DB = HRDatabase()


def _plain(obj):
    """把 pandas / numpy 标量转成原生类型，否则 json 无法序列化。"""
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write("[api] " + (fmt % args) + "\n")

    def _json(self, payload, status=200):
        body = json.dumps(_plain(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            return super().do_GET()

        qs = urllib.parse.parse_qs(parsed.query)
        route = parsed.path

        try:
            if route == "/api/health":
                return self._json({
                    "ok": True,
                    "provider": provider_info().get("provider"),
                    "model": provider_info().get("model"),
                    "reasoner": provider_info().get("reasoner"),
                    "ollama": ollama_available(),
                    "employees": int(len(DB.profiles)),
                })

            if route == "/api/overview":
                return self._json({"ok": True, "data": DB.overview()})

            if route == "/api/employees":
                rows = [
                    {"employee_id": r.employee_id, "name": r.name,
                     "department": r.department, "risk_score": round(float(r.risk_score), 4),
                     "risk_level": r.risk_level}
                    for r in DB.profiles.sort_values("risk_score", ascending=False).itertuples()
                ]
                return self._json({"ok": True, "data": rows})

            if route == "/api/ask":
                q = (qs.get("q") or [""])[0].strip()
                if not q:
                    return self._json({"ok": False, "error": "缺少参数 q"}, 400)
                use_model = (qs.get("model") or ["1"])[0] not in ("0", "false", "no")
                mode = (qs.get("mode") or ["route"])[0].strip().lower()

                # Agent 模式：模型自主选择查询工具
                if mode == "agent" and use_model and (deepseek_ready() or ollama_available()):
                    try:
                        from hr_agent import AGENT_MODEL, HRAgent
                        out = HRAgent(db=DB, model=AGENT_MODEL).run(q)
                        return self._json({
                            "ok": True, "intent": "agent", "answer": out["answer"],
                            "model_used": True, "model": out["model"],
                            "tools": [{"name": t["name"], "args": t["args"]}
                                      for t in out["tool_calls"]],
                            "facts": [], "employees": [],
                        })
                    except Exception as exc:
                        return self._json({
                            "ok": False,
                            "error": f"Agent 调用失败：{exc}。可改用 mode=route。"}, 500)

                result = DB.answer(q)
                payload = {"ok": True, **result, "answer": None, "model_used": False}
                if use_model and (deepseek_ready() or ollama_available()):
                    try:
                        payload["answer"] = narrate(q, result["facts"])
                        payload["model_used"] = True
                    except Exception as exc:
                        payload["model_error"] = str(exc)
                return self._json(payload)

            return self._json({"ok": False, "error": f"未知接口 {route}"}, 404)

        except Exception as exc:  # 保证前端永远拿到 JSON
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[server] 大屏 + 查询 API: http://127.0.0.1:{port}/employee_archive.html")
    info = provider_info()
    print(f"[server] 档案人数 {len(DB.profiles)}，模型 {info.get('provider')}/{info.get('model')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 已停止")


if __name__ == "__main__":
    main()
