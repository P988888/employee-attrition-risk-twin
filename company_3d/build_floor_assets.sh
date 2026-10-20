#!/usr/bin/env bash
# 从权威四层布局导出网页使用的 Blender 静态环境资产。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLENDER=""
for c in blender "/Applications/Blender.app/Contents/MacOS/Blender"; do
  if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then
    BLENDER="$c"
    break
  fi
done
if [ -z "$BLENDER" ]; then
  echo "找不到 Blender。请安装 Blender 或把它加入 PATH。" >&2
  exit 1
fi

SCRIPT="$ROOT/company_3d/03_scripts/scr_build-company-scene_v02.00.py"
LAYOUT="$ROOT/data/building_layout.json"
WIP="$ROOT/company_3d/02_scenes/wip"
WEB="$ROOT/output/dashboard/assets/office"
mkdir -p "$WIP" "$WEB"

# 当前大屏使用 building_layout.json 的单层 1F 42 人版本；其它楼层
# 尚未有对应的员工档案，不在这里伪造人物资产。
floor=1
out="$WIP/floor_${floor}_blender_people.blend"
echo "[floor-assets] 导出 ${floor}F Blender 人物资产 -> ${out}"
"$BLENDER" --background --factory-startup \
  --python "$SCRIPT" -- \
  --layout "$LAYOUT" --floor "$floor" \
  --out "$out" --characters 1 --walkers 8 --lod full --anim 1
cp "${out%.blend}.glb" "$WEB/floor_${floor}_blender_people.glb"

echo "[floor-assets] 已更新：$WEB/floor_1_blender_people.glb"
