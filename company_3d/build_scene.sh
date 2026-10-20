#!/usr/bin/env bash
# 一键：生成公司平面 -> Blender 建场景（真家具 + 会走路的人物）-> 渲染预览
#
# 用法:
#   ./build_scene.sh                 # 默认 8 个巡游人物，渲染 6 张预览
#   ./build_scene.sh 12              # 12 个巡游人物
#   ./build_scene.sh 0               # 全部坐在工位，无人走动
set -euo pipefail

cd "$(dirname "$0")"

WALKERS="${1:-8}"
LOD="${2:-full}"

BLENDER=""
for c in "blender" "/Applications/Blender.app/Contents/MacOS/Blender"; do
  if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then BLENDER="$c"; break; fi
done
if [ -z "$BLENDER" ]; then
  echo "找不到 Blender。请安装 Blender 或把它加入 PATH。" >&2
  exit 1
fi
echo "使用 Blender: $BLENDER"
"$BLENDER" --version 2>/dev/null | head -1 || true

OUT="02_scenes/wip/scn_company-hq_v02.00_wip.blend"

echo
echo "[1/3] 生成公司平面布局（30 工位 / 13 房间 / 家具 / 行走路径）..."
( cd ../src && python3 step7_office_layout.py )

echo
echo "[2/3] Blender 构建场景 (巡游人物=${WALKERS}, LOD=${LOD}) ..."
"$BLENDER" --background --factory-startup \
  --python 03_scripts/scr_build-company-scene_v02.00.py -- \
  --layout ../data/company_layout.json \
  --out "$OUT" --walkers "$WALKERS" --lod "$LOD" --anim 1

echo
echo "[3/3] 渲染预览 ..."
mkdir -p 05_renders/preview
"$BLENDER" --background "$OUT" --python-expr "
import bpy
s = bpy.context.scene
s.render.resolution_x = 1600
s.render.resolution_y = 1000
try: s.eevee.taa_render_samples = 64
except Exception: pass
s.frame_set(12)
for cam, fp in [
    ('cam_overview',  '05_renders/preview/rnd_company-hq_overview_v02.00.png'),
    ('cam_top',       '05_renders/preview/rnd_company-hq_top_v02.00.png'),
    ('cam_open',      '05_renders/preview/rnd_company-hq_openoffice_v02.00.png'),
    ('cam_chairman',  '05_renders/preview/rnd_company-hq_chairman_v02.00.png'),
    ('cam_corridor',  '05_renders/preview/rnd_company-hq_corridor_v02.00.png'),
    ('cam_reception', '05_renders/preview/rnd_company-hq_reception_v02.00.png'),
]:
    s.camera = bpy.data.objects[cam]
    s.render.filepath = fp
    bpy.ops.render.render(write_still=True)
print('渲染完成')
"

echo
echo "--------------------------------------------"
echo " 场景工程 : $OUT"
echo " 导出的   : ${OUT%.blend}.glb   (含走路动画)"
echo " 预览渲染 : 05_renders/preview/  (6 张)"
echo "--------------------------------------------"
echo
echo "在 Blender 里打开 .blend，按空格播放即可看到人物走动。"
