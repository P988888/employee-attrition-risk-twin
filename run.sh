#!/usr/bin/env bash
# 一键跑通：数据准备 -> 训练评估 -> 数据体检 -> 线性回归 -> 行为链路 -> 笔记本 -> 启动大屏
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-8777}"

echo "============================================"
echo " 员工离职风险预测 · 办公场景完整行为链路"
echo " （含办公楼 3D 场景布局）"
echo "============================================"

echo
echo "[1/8] 准备数据（真实数据 + 分布对齐模拟数据）..."
python3 src/step1_prepare_data.py

echo
echo "[2/8] 特征工程 / 多模型训练 / 三段式评估 / 导出大屏数据..."
python3 src/step2_train_model.py

echo
echo "[3/8] 数据整理与体检（缺失/异常/分布/相关性）..."
python3 src/step3_data_profile.py

echo
echo "[4/8] 线性回归建模（收入回归 + 线性概率模型）..."
python3 src/step4_linear_regression.py

echo
echo "[5/8] 办公场景行为链路（5 张关联表 + 链路分析）..."
python3 src/step5_build_behavior_link.py
python3 src/step6_analyze_link.py

echo
echo "[6/8] 公司总部大楼布局（4 层 / 42 工位 / 电梯核心筒）..."
python3 src/step7_office_layout.py

echo
echo "[7/8] 42 名员工完整档案（IBM 格式 + 背调 + 行为 + 测评）..."
python3 src/step8_employee_profiles.py

echo
echo "[8/8] 生成 Jupyter notebook ..."
python3 src/build_notebook.py

# 把建筑与档案数据同步给前端大屏
cp data/building_layout.json output/dashboard/building.json
cp data/employee_profiles.json output/dashboard/profiles.json
python3 src/validate_building_consistency.py

echo
echo "--------------------------------------------"
echo " Notebook   : notebooks/线性回归_员工离职分析.ipynb"
echo " 启动       : jupyter notebook notebooks/"
echo " 数字孪生   : http://127.0.0.1:${PORT}/building.html  <- 4 层大楼，可点人查看档案"
echo " 分析大屏   : http://127.0.0.1:${PORT}/index.html"
echo " 3D 场景    : company_3d/ （用 Blender 运行 03_scripts 下的构建脚本）"
echo "--------------------------------------------"
echo
cd output/dashboard
echo " 查询接口   : /api/ask?q=问题 （模型在线时附加解读，离线时只返回查表事实）"
echo " 离线查询   : python3 src/hr_query.py \"冯嘉豪的风险情况\""
exec python3 ../src/hr_server.py "$PORT"
