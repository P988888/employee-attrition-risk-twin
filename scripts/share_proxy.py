#!/usr/bin/env python3
"""带 Basic Auth + AI 限流的只读分享代理。

本地大屏服务（默认 127.0.0.1:8791）不直接暴露到公网；
Cloudflare Tunnel 指向本代理，所有页面/API 请求先经过口令校验。
"""
from __future__ import annotations

import base64
import hmac
import http.client
import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

UPSTREAM_HOST = os.environ.get("SHARE_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("SHARE_UPSTREAM_PORT", "8791"))
LISTEN_PORT = int(os.environ.get("SHARE_PORT", "8788"))
USER = os.environ.get("SHARE_USER", "nexus")
PASSWORD = os.environ.get("SHARE_PASSWORD", "")
AI_LIMIT = int(os.environ.get("AI_RATE_LIMIT", "6"))
AI_WINDOW = int(os.environ.get("AI_RATE_WINDOW", "60"))
TIMEOUT = int(os.environ.get("SHARE_TIMEOUT", "240"))

if not PASSWORD:
    raise SystemExit("SHARE_PASSWORD 未设置")

_hits: dict[str, deque] = defaultdict(deque)
_locksh = threading.Lock()


def client_ip(handler: BaseHTTPRequestHandler) -> str:
    for key in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
        value = handler.headers.get(key)
        if value:
            return value.split(",")[0].strip()
    return handler.client_address[0]


def allowed(ip: str) -> bool:
    now = time.time()
    with _locksh:
        q = _hits[ip]
        while q and now - q[0] > AI_WINDOW:
            q.popleft()
        if len(q) >= AI_LIMIT:
            return False
        q.append(now)
        return True


class Handler(BaseHTTPRequestHandler):
    server_version = "NexusShare/1.0"

    def log_message(self, fmt, *args):
        print(f"{time.strftime('%H:%M:%S')} {client_ip(self)} {fmt % args}", flush=True)

    def _authorized(self) -> bool:
        raw = self.headers.get("Authorization", "")
        if not raw.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(raw[6:]).decode("utf-8")
            user, password = decoded.split(":", 1)
        except Exception:
            return False
        return hmac.compare_digest(user, USER) and hmac.compare_digest(password, PASSWORD)

    def _challenge(self):
        body = "<!doctype html><meta charset='utf-8'><title>NEXUS 访问验证</title><body style='background:#070b18;color:#bfe8ff;font:14px/1.7 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;padding:48px'><h2>NEXUS 28F 数据看板</h2><p>需要访问口令。请在弹窗中输入分享账号和密码。</p></body>".encode("utf-8")
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="NEXUS 28F Dashboard", charset="UTF-8"')
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_text(self, status: int, text: str, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _proxy(self):
        if not self._authorized():
            self._challenge()
            return
        path = self.path or "/"
        endpoint = urlsplit(path).path
        if endpoint == "/api/ask" and not allowed(client_ip(self)):
            self._send_text(429, json.dumps({"ok": False, "error": "AI 查询太频繁，请稍后再试"}, ensure_ascii=False),
                            "application/json; charset=utf-8")
            return
        if self.command not in ("GET", "HEAD"):
            self._send_text(405, "Method Not Allowed")
            return
        headers = {
            "Host": f"{UPSTREAM_HOST}:{UPSTREAM_PORT}",
            "Accept": self.headers.get("Accept", "*/*"),
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=TIMEOUT)
        try:
            conn.request(self.command, path, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            self.send_response(resp.status)
            for key in ("Content-Type", "Content-Disposition", "Cache-Control", "ETag",
                        "Last-Modified", "Location", "Content-Range", "Accept-Ranges"):
                val = resp.getheader(key)
                if val:
                    self.send_header(key, val)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except Exception as exc:
            self._send_text(502, f"上游服务不可用：{type(exc).__name__}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def do_GET(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()


def main():
    print(f"[share] http://127.0.0.1:{LISTEN_PORT} -> http://{UPSTREAM_HOST}:{UPSTREAM_PORT} | user={USER} | AI {AI_LIMIT}/{AI_WINDOW}s", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
