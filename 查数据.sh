#!/usr/bin/env bash
# 员工数据查询终端快捷入口
#
#   ./查数据.sh               Agent 模式（默认，8b 模型自主决定查什么，能处理灵活问法）
#   ./查数据.sh --route       快速路由模式（1.5b 润色，问法需在预设关键词内）
#   ./查数据.sh --no-model    只用查表，不调用任何模型，结果 100% 来自 CSV
cd "$(dirname "$0")"
exec python3 src/hr_chat.py "$@"
