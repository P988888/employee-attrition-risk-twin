#!/usr/bin/env python3
"""DeepSeek 云端模型客户端。

Key 只从环境变量或项目根目录 .env 读取，不写进源码，也不在日志/返回值中输出。
DeepSeek 走 OpenAI 兼容协议；工具调用使用标准 tools/tool_calls。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_env()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_REASONER_MODEL = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner").strip()
DEEPSEEK_TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", "90"))


def deepseek_ready() -> bool:
    return bool(DEEPSEEK_API_KEY)


def provider_info() -> dict:
    if deepseek_ready():
        return {"provider": "deepseek", "model": DEEPSEEK_MODEL,
                "reasoner": DEEPSEEK_REASONER_MODEL}
    return {"provider": "ollama", "model": os.environ.get("HR_MODEL", "gpt-oss:120b-cloud")}


def deepseek_chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.35,
    max_tokens: int = 900,
    timeout: int | None = None,
) -> dict:
    """调用 DeepSeek /chat/completions，返回原始 message 与 usage。"""
    if not deepseek_ready():
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    body = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    try:
        resp = requests.post(
            DEEPSEEK_BASE_URL + "/chat/completions",
            headers={
                "Authorization": "Bearer " + DEEPSEEK_API_KEY,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout or DEEPSEEK_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"DeepSeek 网络请求失败：{type(exc).__name__}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
        except Exception:
            detail = resp.text
        # 不把 Authorization/Key 带入错误信息
        raise RuntimeError(f"DeepSeek API {resp.status_code}: {str(detail)[:500]}")
    data = resp.json()
    try:
        message = data["choices"][0]["message"]
    except Exception as exc:
        raise RuntimeError("DeepSeek 返回结构异常") from exc
    return {"message": message, "usage": data.get("usage", {}), "raw": data}
