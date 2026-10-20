"""公司办公场景构建脚本 v02（Blender 4.x / 5.x）

相比 v01 的升级
--------------
  1. 默认兼容旧的 `data/company_layout.json`，也可直接读取权威四层 `data/building_layout.json`
  2. **真家具**：每种家具由多个部件组成（桌面+桌腿+挡板+抽屉、五爪椅、
     沙发坐垫+靠背+扶手、书柜层板、绿植花盆+枝叶…），不再是单一方块
  3. **人物角色**：低多边形人形（头/颈/躯干/髋/四肢/手脚），
     带 Armature 骨骼 + 顶点组蒙皮
  4. **行走动画**：24 帧循环走路，人物沿走廊路径网络自动巡游
  5. **配色**：按部门/房间类型分配材质，地板分材质（地毯/木地板/瓷砖）
  6. 房间隔墙、门洞、吊顶、灯光

坐标系：Z 向上，原点在楼层西南角，单位米。

用法
----
    blender --background --python scr_build-company-scene_v02.00.py -- \
        --layout ../data/building_layout.json --floor 3 \
        --out 02_scenes/wip/floor_3_environment.blend \
        --characters 0 --lod full

参数
----
--layout   布局 JSON（推荐使用 data/building_layout.json）
--floor    使用 building_layout.json 时导出的楼层（默认 3）
--characters 1/0  是否生成 Blender 人物；网页环境资产建议设为 0
--out      输出 .blend（同时导出 .glb，含动画）
--walkers  巡游人物数量（默认 8；0 = 全部坐在工位）
--lod      full（全部家具细节）| simple（简化）
--anim     1/0 是否生成走路动画（默认 1）
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Euler

# ================================================================ 参数
def parse_args() -> dict:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    a = {"layout": "../data/company_layout.json", "floor": "3", "out": None,
         "walkers": "8", "characters": "1", "lod": "full", "anim": "1"}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            k = argv[i][2:]
            v = argv[i + 1] if i + 1 < len(argv) else ""
            if k in a:
                a[k] = v
            i += 2
        else:
            i += 1
    return a


ARGS = parse_args()
HERE = Path(__file__).resolve().parent
LP = Path(ARGS["layout"])
if not LP.is_absolute():
    LP = (Path.cwd() / LP).resolve()
if not LP.exists():
    LP = (HERE.parent.parent / "data" / "company_layout.json").resolve()

with open(LP, encoding="utf-8") as f:
    RAW_LAYOUT = json.load(f)

# 当前大屏的员工主数据与 building_layout.json 分开存放；Blender 通过 desk_id
# 把两者合并，保证导出的角色节点可以直接回到 profiles.json。
PROFILE_PATH = LP.parent / "employee_profiles.json"
if PROFILE_PATH.exists():
    with open(PROFILE_PATH, encoding="utf-8") as f:
        EMPLOYEE_PROFILES = json.load(f)
else:
    EMPLOYEE_PROFILES = []
PROFILE_BY_DESK = {
    p.get("desk_id"): p for p in EMPLOYEE_PROFILES if p.get("desk_id")
}


def normalize_layout(raw: dict, floor_level: int) -> dict:
    """把 building_layout 转成当前单层构建器接口，并保留真实 employee_id。

    building_layout 是空间坐标源，employee_profiles.json 是员工身份源；两者
    通过 desk_id 连接。旧 company_layout.json 仍可用于兼容旧工程，但不会
    擅自把当前 42 人场景改名成 F1-DESK-xx。
    """
    if "floor" in raw:
        return raw
    floors = raw.get("floors", [])
    floor = next((f for f in floors if int(f.get("level", 0)) == int(floor_level)), None)
    if floor is None:
        raise ValueError(f"building_layout 中不存在楼层 {floor_level}")
    building = raw["building"]
    footprint = building["footprint"]
    staff = []
    for i, desk in enumerate(floor.get("desks", []), start=1):
        desk_id = desk.get("desk_id")
        profile = PROFILE_BY_DESK.get(desk_id, {})
        desk_meta = profile.get("desk") or {}
        staff.append({
            "employee_id": profile.get("employee_id") or f"F{floor['level']}-DESK-{i:02d}",
            "name": profile.get("name", ""),
            "department_key": profile.get("department_key") or desk.get("dept") or "rnd",
            "desk_id": desk_id,
            "desk_x": desk.get("x", desk_meta.get("x", 0)),
            "desk_y": desk.get("y", desk_meta.get("y", 0)),
            "desk_rot": desk.get("rot", desk_meta.get("rot", 0)),
        })
    return {
        "coordinate_system": {"up": "Z", "unit": "meter", "origin": "floor_south_west"},
        "floor": {
            "width": footprint["w"], "depth": footprint["d"],
            "ceiling_height": building.get("floor_height", 3.6) - building.get("slab_thickness", 0.35),
            "wall_thickness": building.get("wall_thickness", 0.2),
            "door_width": building.get("door_width", 1.1),
        },
        "desk_module": raw.get("desk_module", {}),
        "headcount": len(floor.get("desks", [])),
        "departments": raw.get("departments", []),
        "rooms": floor.get("rooms", []), "desks": floor.get("desks", []),
        "furniture": floor.get("furniture", []), "waypoints": floor.get("waypoints", []),
        "staff": staff,
    }


L = normalize_layout(RAW_LAYOUT, int(ARGS["floor"]))

LOD = ARGS["lod"]
N_WALKERS = int(ARGS["walkers"])
DO_ANIM = ARGS["anim"] == "1"
INCLUDE_CHARACTERS = ARGS["characters"] == "1"

FW = L["floor"]["width"]
FD = L["floor"]["depth"]
CH = L["floor"]["ceiling_height"]
WT = L["floor"]["wall_thickness"]
ROOMS = {r["rid"]: r for r in L["rooms"]}
FURN = L["furniture"]
DESKS = L["desks"]
WP = {w["id"]: w for w in L["waypoints"]}
STAFF = L["staff"]
DEPT_COLOR = {d["key"]: d["color"] for d in L["departments"]}

print(f"[build] 布局: {LP}")
print(f"[build] 楼层 {FW}×{FD}m | 房间 {len(ROOMS)} | 工位 {len(DESKS)} | "
      f"家具 {len(FURN)} | 员工 {len(STAFF)}")


# ================================================================ 基础工具
def hex2rgba(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255, a)


def mat(name, color, rough=0.6, metal=0.0, alpha=1.0, emit=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    c = hex2rgba(color, alpha)
    b.inputs["Base Color"].default_value = c
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit and "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = c
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0 and hasattr(m, "surface_render_method"):
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass
    return m


def ensure_coll(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def box_data(cx, cy, cz, sx, sy, sz):
    """返回一个长方体的 (verts, faces)（局部坐标）。"""
    hx, hy, hz = sx/2, sy/2, sz/2
    v = [(cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz),
         (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
         (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz),
         (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz)]
    f = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (1,2,6,5), (0,3,7,4)]
    return v, f


def merge(parts):
    """把多个 (verts, faces) 合并成一份网格数据。"""
    V, F = [], []
    for vs, fs in parts:
        off = len(V)
        V += vs
        F += [tuple(i+off for i in fc) for fc in fs]
    return V, F


def mesh_from(name, parts_or_vf, material=None, coll=None):
    if isinstance(parts_or_vf, tuple) and len(parts_or_vf) == 2 \
            and isinstance(parts_or_vf[0], list) and isinstance(parts_or_vf[1], list):
        V, F = parts_or_vf
    else:
        V, F = merge(parts_or_vf)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    if material:
        me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def proto(name, parts, material):
    """建一份可复用的原型网格（后续所有实例共享它）。"""
    key = f"proto_{name}"
    me = bpy.data.meshes.get(key)
    if me is None:
        V, F = merge(parts)
        me = bpy.data.meshes.new(key)
        me.from_pydata(V, [], F)
        me.update()
        if material:
            me.materials.append(material)
    return me


def place(mesh, name, x, y, z, rot=0.0, coll=None, scale=1.0):
    ob = bpy.data.objects.new(name, mesh)
    ob.location = (x, y, z)
    ob.rotation_euler[2] = rot
    if scale != 1.0:
        ob.scale = (scale, scale, scale)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


# ================================================================ 材质
M = {}


def build_materials():
    M.clear()
    M.update({
        "slab":    mat("mat_slab", "#D5DAE2", rough=.9),
        "wall":    mat("mat_wall", "#F4F6FA", rough=.92),
        "ceiling": mat("mat_ceiling", "#FFFFFF", rough=.95),
        "glass":   mat("mat_glass", "#B6DAF2", rough=.06, alpha=.22),
        "mullion": mat("mat_mullion", "#8A93A0", rough=.3, metal=.85),
        "wood":    mat("mat_wood", "#8A5D3B", rough=.45),
        "wood_d":  mat("mat_wood_dark", "#5E3D28", rough=.42),
        "metal":   mat("mat_metal", "#6E7783", rough=.32, metal=.9),
        "black":   mat("mat_black", "#22262C", rough=.4),
        "screen":  mat("mat_screen", "#10161E", rough=.15, emit=.35),
        "fabric":  mat("mat_fabric", "#3E4756", rough=.85),
        "fab_bl":  mat("mat_fabric_blue", "#2E4A6B", rough=.85),
        "fab_gr":  mat("mat_fabric_green", "#2F5D46", rough=.85),
        "fab_gray": mat("mat_fabric_gray", "#7C8592", rough=.88),
        "plant_g": mat("mat_leaf", "#2E7D4F", rough=.75),
        "plant_g2": mat("mat_leaf2", "#3E9A63", rough=.75),
        "pot":     mat("mat_pot", "#9A6B4F", rough=.8),
        "white":   mat("mat_white", "#F0F2F5", rough=.7),
        "stone":   mat("mat_stone", "#C9CDD4", rough=.5),
        "steel":   mat("mat_steel", "#B9C0C9", rough=.28, metal=.8),
        "skin":    mat("mat_skin", "#E8B98F", rough=.72),
        "hair":    mat("mat_hair", "#2B2320", rough=.8),
        "shoe":    mat("mat_shoe", "#23262B", rough=.6),
        "paper":   mat("mat_paper", "#F5F5F0", rough=.9),
        "rug_r":   mat("mat_rug_red", "#7A4A4A", rough=.95),
        "carpet":  mat("mat_carpet", "#C9CDD6", rough=.98),
        "tile":    mat("mat_tile", "#E4E8ED", rough=.35),
        "door":    mat("mat_door", "#B98A5E", rough=.5),
    })
    for k, c in DEPT_COLOR.items():
        M[f"dept_{k}"] = mat(f"mat_dept_{k}", c, rough=.85)
    for k, c in [("office", "#8E7CC3"), ("meeting", "#5B8FF9"),
                 ("facility", "#9AA7B8")]:
        M[f"zone_{k}"] = mat(f"mat_zone_{k}", c, rough=.95, alpha=.30)


# ================================================================ 家具部件库
def f_desk():
    """工位桌：桌面 + 两侧板 + 挡板 + 抽屉柜。"""
    w, d, h = 1.5, 0.75, 0.74
    return [
        box_data(0, 0, h, w, d, 0.05),
        box_data(-w/2+0.03, 0, h/2, 0.06, d*0.9, h),
        box_data(w/2-0.03, 0, h/2, 0.06, d*0.9, h),
        box_data(0, -d/2+0.02, h-0.25, w, 0.04, 0.3),
        box_data(w/4, 0.03, 0.28, 0.42, d*0.7, 0.56),
    ]


def f_chair_task():
    """办公椅：座面 + 靠背 + 扶手 + 气压杆 + 五爪底盘 + 滚轮。"""
    parts = [
        box_data(0, 0, 0.46, 0.5, 0.5, 0.08),
        box_data(0, -0.24, 0.72, 0.48, 0.07, 0.46),
        box_data(0, 0, 0.30, 0.06, 0.06, 0.34),
        box_data(0, 0, 0.16, 0.34, 0.34, 0.05),
    ]
    for a in range(5):
        th = a * 2 * math.pi / 5
        parts.append(box_data(math.cos(th)*0.22, math.sin(th)*0.22, 0.06,
                              0.30, 0.06, 0.04))
        parts.append(box_data(math.cos(th)*0.36, math.sin(th)*0.36, 0.035,
                              0.06, 0.06, 0.06))
    parts += [box_data(-0.26, -0.04, 0.60, 0.05, 0.26, 0.05),
              box_data(0.26, -0.04, 0.60, 0.05, 0.26, 0.05)]
    return parts


def f_chair_meeting():
    return [
        box_data(0, 0, 0.45, 0.46, 0.46, 0.07),
        box_data(0, -0.22, 0.68, 0.44, 0.06, 0.42),
        box_data(-0.19, 0, 0.22, 0.05, 0.05, 0.46),
        box_data(0.19, 0, 0.22, 0.05, 0.05, 0.46),
        box_data(0, 0, 0.05, 0.42, 0.42, 0.05),
    ]


def f_chair_exec():
    p = f_chair_task()
    p[1] = box_data(0, -0.25, 0.90, 0.50, 0.08, 0.72)   # 高靠背
    return p


def f_exec_desk(w=2.4, d=1.1):
    """老板台：厚桌面 + 侧板 + 前挡板 + 侧柜。"""
    h = 0.76
    return [
        box_data(0, 0, h, w, d, 0.07),
        box_data(-w/2+0.04, 0, h/2, 0.08, d*0.95, h),
        box_data(w/2-0.04, 0, h/2, 0.08, d*0.95, h),
        box_data(0, -d/2+0.03, h-0.32, w, 0.06, 0.4),
        box_data(w/2+0.55, 0, 0.32, 0.9, 0.5, 0.64),
    ]


def f_table_oval(w=5.2, d=1.8):
    """会议长桌（椭圆近似：桌面 + 两块支撑腿板）。"""
    return [
        box_data(0, 0, 0.75, w, d, 0.07),
        box_data(-w/4, 0, 0.36, 0.12, d*0.6, 0.72),
        box_data(w/4, 0, 0.36, 0.12, d*0.6, 0.72),
    ]


def f_table_rect(w=2.6, d=1.2):
    return [
        box_data(0, 0, 0.75, w, d, 0.06),
        box_data(-w/2+0.1, -d/2+0.1, 0.36, 0.08, 0.08, 0.72),
        box_data(w/2-0.1, -d/2+0.1, 0.36, 0.08, 0.08, 0.72),
        box_data(-w/2+0.1, d/2-0.1, 0.36, 0.08, 0.08, 0.72),
        box_data(w/2-0.1, d/2-0.1, 0.36, 0.08, 0.08, 0.72),
    ]


def f_table_round(r=0.6):
    return [
        box_data(0, 0, 0.74, r*2, r*2, 0.06),
        box_data(0, 0, 0.37, 0.09, 0.09, 0.74),
        box_data(0, 0, 0.03, r*1.1, r*1.1, 0.05),
    ]


def f_sofa(w=1.9, d=0.85):
    """沙发：底座 + 靠背 + 两侧扶手 + 坐垫。"""
    return [
        box_data(0, 0, 0.20, w, d, 0.40),
        box_data(0, -d/2+0.12, 0.62, w, 0.24, 0.46),
        box_data(-w/2+0.09, 0, 0.44, 0.18, d, 0.40),
        box_data(w/2-0.09, 0, 0.44, 0.18, d, 0.40),
        box_data(0, 0.06, 0.44, w-0.36, d-0.30, 0.10),
    ]


def f_armchair():
    return f_sofa(0.85, 0.85)


def f_coffee_table(w=1.2, d=0.6):
    return [
        box_data(0, 0, 0.40, w, d, 0.05),
        box_data(-w/2+0.08, -d/2+0.08, 0.20, 0.07, 0.07, 0.40),
        box_data(w/2-0.08, -d/2+0.08, 0.20, 0.07, 0.07, 0.40),
        box_data(-w/2+0.08, d/2-0.08, 0.20, 0.07, 0.07, 0.40),
        box_data(w/2-0.08, d/2-0.08, 0.20, 0.07, 0.07, 0.40),
    ]


def f_bookshelf(w=0.4, d=2.4, h=2.2):
    parts = [box_data(0, 0, h, w, 0.04, 0.04),
             box_data(0, 0, 0.02, w, d, 0.04)]
    for z in (0.02, 0.56, 1.10, 1.64, h):
        parts.append(box_data(0, 0, z, w, d, 0.05))
    for y in (-d/2, d/2):
        parts.append(box_data(0, y, h/2, 0.05, 0.05, h))
    # 书
    for i, y in enumerate([-0.8, -0.2, 0.5, 1.0]):
        parts.append(box_data(0, y, 0.80, w*0.7, 0.28, 0.22))
    return parts


def f_cabinet(w=0.4, d=0.5, h=0.6):
    return [
        box_data(0, 0, h/2, w, d, h),
        box_data(0, d/2+0.01, h*0.72, w*0.8, 0.02, 0.26),
        box_data(0, d/2+0.01, h*0.28, w*0.8, 0.02, 0.26),
        box_data(0, d/2+0.04, h*0.72, w*0.3, 0.03, 0.03),
        box_data(0, d/2+0.04, h*0.28, w*0.3, 0.03, 0.03),
    ]


def f_reception(w=3.4, d=0.9, h=1.1):
    return [
        box_data(0, 0, h/2, w, d, h),
        box_data(0, 0, h+0.03, w+0.2, d+0.2, 0.06),
        box_data(0, -d/2-0.25, 0.30, w*0.8, 0.45, 0.6),
    ]


def f_counter(w=6.0, d=0.65, h=0.9):
    return [
        box_data(0, 0, h/2, w, d, h),
        box_data(0, 0, h+0.03, w+0.1, d+0.05, 0.06),
        box_data(0, -d/2-0.02, h+0.9, w, 0.35, 0.7),   # 吊柜
    ]


def f_fridge(w=0.75, d=0.7, h=1.8):
    return [box_data(0, 0, h/2, w, d, h),
            box_data(0, d/2+0.02, h*0.62, 0.04, 0.03, 0.5),
            box_data(0, d/2+0.02, h*0.28, 0.04, 0.03, 0.3)]


def f_plant(h=1.4, big=True):
    parts = [box_data(0, 0, 0.22, 0.42, 0.42, 0.44),
             box_data(0, 0, 0.22+h*0.25, 0.07, 0.07, h*0.5)]
    if big:
        for (dx, dy, dz, s) in [(0, 0, h*0.78, 0.55), (0.16, 0.1, h*0.68, 0.4),
                                (-0.15, -0.08, h*0.72, 0.42),
                                (0.05, -0.16, h*0.62, 0.36)]:
            parts.append(box_data(dx, dy, dz, s, s, s*0.8))
    else:
        parts.append(box_data(0, 0, h*0.7, 0.34, 0.34, 0.3))
    return parts


def f_monitor():
    return [box_data(0, 0, 0.28, 0.58, 0.04, 0.34),
            box_data(0, 0, 0.09, 0.16, 0.12, 0.18),
            box_data(0, 0, 0.01, 0.24, 0.16, 0.03)]


def f_keyboard():
    return [box_data(0, 0, 0.01, 0.40, 0.15, 0.02),
            box_data(0.26, 0.02, 0.015, 0.09, 0.13, 0.03)]


def f_screen(w=2.6, h=1.5):
    return [box_data(0, 0, 1.0+h/2-0.75, w, 0.08, h),
            box_data(0, 0, 0.5, 0.1, 0.1, 1.0),
            box_data(0, 0, 0.03, w*0.4, 0.3, 0.06)]


def f_whiteboard(w=2.4, h=1.2):
    return [box_data(0, 0, 1.5, w, 0.06, h),
            box_data(0, 0.02, 1.5, w*0.96, 0.02, h*0.92)]


def f_printer(w=0.7, d=0.7, h=1.1):
    return [box_data(0, 0, h/2, w, d, h),
            box_data(0, d/2, h+0.06, w*0.7, 0.3, 0.12)]


def f_shredder():
    return [box_data(0, 0, 0.4, 0.45, 0.45, 0.8),
            box_data(0, 0, 0.86, 0.35, 0.3, 0.12)]


def f_shelf(w=0.5, d=4.0, h=2.0):
    parts = [box_data(0, 0, 0.02, w, d, 0.04)]
    for z in (0.5, 1.0, 1.5, h):
        parts.append(box_data(0, 0, z, w, d, 0.05))
    for y in (-d/2, d/2):
        parts.append(box_data(0, y, h/2, 0.05, 0.05, h))
    return parts


def f_sink():
    return [box_data(0, 0, 0.02, 0.7, 0.44, 0.06),
            box_data(0, 0, 0.22, 0.04, 0.04, 0.42),
            box_data(0, -0.06, 0.44, 0.04, 0.22, 0.04)]


def f_coffee_machine():
    return [box_data(0, 0, 0.18, 0.34, 0.4, 0.36),
            box_data(0, 0.06, 0.40, 0.3, 0.28, 0.14)]


def f_water_cooler():
    return [box_data(0, 0, 0.45, 0.32, 0.32, 0.9),
            box_data(0, 0, 1.0, 0.26, 0.26, 0.24)]


def f_bar_stool():
    return [box_data(0, 0, 0.72, 0.38, 0.38, 0.06),
            box_data(0, 0, 0.36, 0.06, 0.06, 0.72),
            box_data(0, 0, 0.03, 0.3, 0.3, 0.05),
            box_data(0, 0, 0.24, 0.26, 0.26, 0.04)]


def f_tv(w=1.3, h=0.75):
    return [box_data(0, 0, 1.35, w, 0.06, h),
            box_data(0, 0, 0.5, 0.08, 0.08, 0.95),
            box_data(0, 0, 0.03, 0.5, 0.3, 0.06)]


def f_toilet_stall(w=1.0, d=1.4, h=2.2):
    return [box_data(0, -d/2, h/2, w, 0.04, h),
            box_data(-w/2, 0, h/2, 0.04, d, h),
            box_data(w/2, 0, h/2, 0.04, d, h),
            box_data(0, -d/2+0.25, 0.2, 0.38, 0.5, 0.4)]


def f_basin():
    return [box_data(0, 0, 0.42, 0.5, 1.2, 0.06),
            box_data(0, 0, 0.2, 0.12, 1.0, 0.4),
            box_data(0, 0, 0.88, 0.36, 0.06, 0.6)]


def f_logo_wall(w=2.4, h=1.2):
    return [box_data(0, 0, 1.7, w, 0.08, h),
            box_data(0, -0.06, 1.7, w*0.8, 0.02, h*0.6)]


def f_floor_lamp():
    return [box_data(0, 0, 0.02, 0.3, 0.3, 0.04),
            box_data(0, 0, 0.85, 0.05, 0.05, 1.7),
            box_data(0, 0, 1.76, 0.3, 0.3, 0.26)]


def f_rug(w=3.2, d=2.4):
    return [box_data(0, 0, 0.012, w, d, 0.02)]


def f_bin():
    return [box_data(0, 0, 0.25, 0.28, 0.28, 0.5)]


FERNS = {
    "desk": (f_desk, "wood", 1.5, 0.75, 0.78),
    "chair": (f_chair_task, "fabric", 0.55, 0.55, 0.95),
    "meeting_chair": (f_chair_meeting, "fab_gray", 0.48, 0.48, 0.9),
    "guest_chair": (f_chair_meeting, "fab_bl", 0.5, 0.5, 0.9),
    "exec_chair": (f_chair_exec, "black", 0.6, 0.6, 1.2),
    "exec_desk": (f_exec_desk, "wood_d", 2.4, 1.1, 0.78),
    "table_oval": (f_table_oval, "wood_d", 5.2, 1.8, 0.78),
    "table_rect": (f_table_rect, "wood_d", 2.6, 1.2, 0.78),
    "table_round": (f_table_round, "wood", 1.2, 1.2, 0.77),
    "sofa": (f_sofa, "fab_bl", 1.9, 0.85, 0.85),
    "armchair": (f_armchair, "fab_bl", 0.85, 0.85, 0.85),
    "coffee_table": (f_coffee_table, "wood", 1.2, 0.6, 0.43),
    "bookshelf": (f_bookshelf, "wood_d", 0.4, 2.4, 2.2),
    "cabinet": (f_cabinet, "metal", 0.4, 0.5, 0.6),
    "reception_desk": (f_reception, "wood", 3.4, 0.9, 1.16),
    "counter": (f_counter, "stone", 6.0, 0.65, 1.6),
    "fridge": (f_fridge, "steel", 0.75, 0.7, 1.8),
    "plant": (f_plant, "plant_g", 0.6, 0.6, 1.6),
    "monitor": (f_monitor, "screen", 0.58, 0.16, 0.45),
    "keyboard": (f_keyboard, "black", 0.42, 0.18, 0.05),
    "screen": (f_screen, "screen", 2.6, 0.3, 1.9),
    "whiteboard": (f_whiteboard, "white", 2.4, 0.1, 2.1),
    "printer": (f_printer, "white", 0.7, 0.7, 1.16),
    "shredder": (f_shredder, "black", 0.45, 0.45, 0.92),
    "shelf": (f_shelf, "metal", 0.5, 4.0, 2.0),
    "sink": (f_sink, "steel", 0.7, 0.5, 0.48),
    "coffee_machine": (f_coffee_machine, "black", 0.34, 0.4, 0.47),
    "water_cooler": (f_water_cooler, "steel", 0.32, 0.32, 1.12),
    "bar_stool": (f_bar_stool, "metal", 0.38, 0.38, 0.75),
    "tv": (f_tv, "screen", 1.3, 0.3, 1.73),
    "toilet_stall": (f_toilet_stall, "white", 1.0, 1.4, 2.2),
    "basin": (f_basin, "white", 0.5, 1.2, 0.9),
    "logo_wall": (f_logo_wall, "wood_d", 2.4, 0.1, 2.3),
    "floor_lamp": (f_floor_lamp, "black", 0.3, 0.3, 1.9),
    "rug": (f_rug, "rug_r", 3.2, 2.4, 0.03),
    "bin": (f_bin, "metal", 0.28, 0.28, 0.5),
}


def build_furniture():
    """按布局 JSON 实例化全部家具（共享原型网格）。"""
    colls = {}
    n = 0
    for f in FURN:
        t = f["ftype"]
        if t not in FERNS:
            continue
        fn, mkey, pw, pd, ph = FERNS[t]
        key = f"{t}_{mkey}"
        me = proto(key, fn(), M[mkey])
        cname = f"furn_{t}"
        if cname not in colls:
            colls[cname] = ensure_coll(cname)
        # 优先用布局里给的尺寸缩放（保留原型比例）
        sc = 1.0
        if f.get("w") and pw:
            sc = max(0.5, min(1.6, f["w"] / pw))
        place(me, f["fid"], f["x"], f["y"], 0.0, f.get("rot", 0.0),
              colls[cname], sc)
        n += 1
    return n


# ================================================================ 人物
# 比普通低多边形人形略高，保证楼层俯视镜头下仍能辨认头部和四肢。
CHAR_H = 1.88
# 材质槽顺序（与 build_character 里 append 的顺序一致）
CM_SKIN, CM_SHIRT, CM_PANTS, CM_HAIR, CM_SHOE = 0, 1, 2, 3, 4


def char_parts():
    """角色各部位（局部坐标，原点在双脚中心，面朝 +Y）。

    返回 [(顶点组名, verts, faces, 材质索引), ...]。
    材质**显式指定**，不再靠顶点高度猜测 —— 否则头发会被误判成肤色。
    """
    P = []

    def add(group, parts, mi):
        V, F = merge(parts)
        P.append((group, V, F, mi))

    h = CHAR_H
    # 头 / 头发 / 颈
    add("head", [box_data(0, 0, h*0.925, 0.175, 0.19, 0.21)], CM_SKIN)
    # 头发：一顶干净的「帽子」，略大于头顶，不再单独加前刘海那块（会显得破相）
    add("head", [box_data(0, 0, h*0.985, 0.185, 0.20, 0.06),
                 box_data(0, 0.075, h*0.955, 0.185, 0.055, 0.12)], CM_HAIR)
    add("neck", [box_data(0, 0, h*0.835, 0.08, 0.08, 0.075)], CM_SKIN)
    # 躯干（上衣）+ 肩 + 髋（裤）
    add("spine", [box_data(0, 0, h*0.70, 0.33, 0.19, 0.33),
                  box_data(0, 0, h*0.795, 0.40, 0.20, 0.10)], CM_SHIRT)
    add("hips", [box_data(0, 0, h*0.555, 0.29, 0.19, 0.13)], CM_PANTS)

    for s, sx in (("L", 0.215), ("R", -0.215)):
        add(f"upperarm.{s}", [box_data(sx, 0, h*0.665, 0.082, 0.082, 0.215)],
            CM_SHIRT)
        add(f"forearm.{s}", [box_data(sx, 0, h*0.505, 0.072, 0.072, 0.20)],
            CM_SKIN)
        add(f"hand.{s}", [box_data(sx, 0, h*0.40, 0.068, 0.085, 0.085)], CM_SKIN)
    for s, sx in (("L", 0.085), ("R", -0.085)):
        add(f"thigh.{s}", [box_data(sx, 0, h*0.42, 0.12, 0.13, 0.35)], CM_PANTS)
        add(f"shin.{s}", [box_data(sx, 0, h*0.175, 0.10, 0.105, 0.35)], CM_PANTS)
        add(f"foot.{s}", [box_data(sx, 0.045, h*0.025, 0.095, 0.21, 0.065)],
            CM_SHOE)
    return P


def build_armature(name, coll):
    """建立骨架：root -> hips -> spine -> neck/head；四肢分左右。"""
    h = CHAR_H
    arm = bpy.data.armatures.new(f"{name}_arm")
    ob = bpy.data.objects.new(name, arm)
    coll.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones

    def mk(bn, head, tail, parent=None, connect=False):
        b = eb.new(bn)
        b.head = head
        b.tail = tail
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    mk("root", (0, 0, 0), (0, 0, h*0.5))
    mk("hips", (0, 0, h*0.50), (0, 0, h*0.58), "root")
    mk("spine", (0, 0, h*0.58), (0, 0, h*0.80), "hips", True)
    mk("neck", (0, 0, h*0.80), (0, 0, h*0.87), "spine", True)
    mk("head", (0, 0, h*0.87), (0, 0, h*1.03), "neck", True)
    for s, sx in (("L", 0.21), ("R", -0.21)):
        mk(f"upperarm.{s}", (sx, 0, h*0.78), (sx, 0, h*0.55), "spine")
        mk(f"forearm.{s}", (sx, 0, h*0.55), (sx, 0, h*0.35), f"upperarm.{s}", True)
        mk(f"hand.{s}", (sx, 0, h*0.35), (sx, 0, h*0.28), f"forearm.{s}", True)
    for s, sx in (("L", 0.085), ("R", -0.085)):
        mk(f"thigh.{s}", (sx, 0, h*0.50), (sx, 0, h*0.26), "hips")
        mk(f"shin.{s}", (sx, 0, h*0.26), (sx, 0, h*0.06), f"thigh.{s}", True)
        mk(f"foot.{s}", (sx, 0, h*0.06), (sx, 0.12, h*0.06), f"shin.{s}", True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


def build_character(name, shirt_color, coll, employee_id=None, employee_name=""):
    """建角色网格 + 骨骼 + 蒙皮，返回 (mesh_obj, arm_obj)。"""
    parts = char_parts()
    V, F = [], []
    groups = {}
    vmat = {}                      # 顶点索引 -> 材质索引
    for gname, vs, fs, mi in parts:
        off = len(V)
        V += vs
        F += [tuple(i + off for i in fc) for fc in fs]
        groups.setdefault(gname, []).extend(range(off, len(V)))
        for i in range(off, len(V)):
            vmat[i] = mi

    me = bpy.data.meshes.new(f"{name}_mesh")
    me.from_pydata(V, [], F)
    me.update()

    shirt = mat(f"mat_shirt_{shirt_color.lstrip('#')}", shirt_color, rough=.85)
    for m in (M["skin"], shirt, M["black"], M["hair"], M["shoe"]):
        me.materials.append(m)
    # 按顶点材质索引给每个面指定材质（取该面首个顶点的索引）
    for poly in me.polygons:
        poly.material_index = vmat.get(poly.vertices[0], CM_SKIN)

    ob = bpy.data.objects.new(f"{name}_body", me)
    coll.objects.link(ob)

    for g, idxs in groups.items():
        vg = ob.vertex_groups.new(name=g)
        vg.add(idxs, 1.0, "REPLACE")

    arm = build_armature(name, coll)
    ob.parent = arm
    md = ob.modifiers.new("Armature", "ARMATURE")
    md.object = arm
    if employee_id:
        arm["employee_id"] = employee_id
        ob["employee_id"] = employee_id
    if employee_name:
        arm["employee_name"] = employee_name
        ob["employee_name"] = employee_name
    arm["asset_role"] = "employee"
    ob["asset_role"] = "employee_body"
    return ob, arm


def pose_seated(arm, x, y, rot):
    """把角色摆成坐姿并放到指定位置。

    注意：这里**必须**同时设置位置与朝向 —— 早期版本在摆完姿势后又写
    arm.location.y = 0.10，把调用方设好的工位坐标覆盖掉了，
    导致 30 个人全被堆到楼层南侧外面。位置参数由此函数统一负责。
    """
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
    for s in ("L", "R"):
        arm.pose.bones[f"thigh.{s}"].rotation_euler.x = math.radians(-78)
        arm.pose.bones[f"shin.{s}"].rotation_euler.x = math.radians(74)
        arm.pose.bones[f"upperarm.{s}"].rotation_euler.x = math.radians(-56)
        arm.pose.bones[f"forearm.{s}"].rotation_euler.x = math.radians(-40)
    # 髋部降到座面高度(≈0.46m)：站立时髋在 h*0.50=0.86m，故整体下沉 0.40m
    arm.location = (x, y, -0.40)
    arm.rotation_euler = (0, 0, rot)


def chair_pos_for(desk):
    """由工位推出椅子的落点与朝向（与 step7 的椅子摆位保持一致）。"""
    off = 0.75 / 2 + 0.45            # DESK_D/2 + 0.45
    rot = desk.get("rot", 0.0)
    return (desk["x"] - math.sin(rot) * off,
            desk["y"] + math.cos(rot) * off,
            rot + math.pi)


def iter_fcurves(act):
    """取 Action 里的全部 F-Curve，兼容两个时代的 API：

      * Blender 4.x 及更早： act.fcurves
      * Blender 4.4+/5.x   ： 分层 Action —— layers → strips → channelbags → fcurves
    """
    if act is None:
        return []
    fcs = getattr(act, "fcurves", None)
    if fcs is not None:
        return list(fcs)
    out = []
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            for cb in getattr(strip, "channelbags", []):
                out.extend(cb.fcurves)
    return out


def set_interp(obj, mode="BEZIER"):
    """把对象当前 Action 的所有关键帧设为指定插值方式。"""
    ad = getattr(obj, "animation_data", None)
    if not ad:
        return
    for fc in iter_fcurves(ad.action):
        for kp in fc.keyframe_points:
            kp.interpolation = mode


def make_walk_action(arm, n_frames=24, stride=28.0, name="walk"):
    """生成平滑的 24 帧步态循环；根节点 bob 与脚掌滚动由骨骼动作承载。"""
    act = bpy.data.actions.new(name)
    arm.animation_data_create()
    arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"

    # 9 个相位点：脚跟着地、支撑、蹬地、摆动、换脚，最后一帧与第 0 帧一致。
    poses = {
        0:  {"tL": stride, "tR": -stride, "sL": -8,  "sR": -22,
             "fL": -5, "fR": 14, "aL": -19, "aR": 19, "el": -14,
             "hip": 1.6, "chest": -1.2, "twist": 1.0, "bob": 0.012},
        3:  {"tL": 18, "tR": -stride, "sL": -10, "sR": -28,
             "fL": 1,  "fR": 17, "aL": -13, "aR": 13, "el": -16,
             "hip": 1.0, "chest": -0.8, "twist": 0.7, "bob": 0.028},
        6:  {"tL": 3,  "tR": -8, "sL": -5,  "sR": -18,
             "fL": 3,  "fR": 9,  "aL": -4,  "aR": 4,  "el": -18,
             "hip": 0.0, "chest": 0.0, "twist": 0.0, "bob": 0.038},
        9:  {"tL": -18, "tR": 12, "sL": -22, "sR": -10,
             "fL": 11, "fR": 2,  "aL": 13, "aR": -13, "el": -16,
             "hip": -1.0, "chest": 0.8, "twist": -0.7, "bob": 0.028},
        12: {"tL": -stride, "tR": stride, "sL": -22, "sR": -8,
             "fL": 14, "fR": -5, "aL": 19, "aR": -19, "el": -14,
             "hip": -1.6, "chest": 1.2, "twist": -1.0, "bob": 0.012},
        15: {"tL": -stride, "tR": 18, "sL": -28, "sR": -10,
             "fL": 17, "fR": 1, "aL": 13, "aR": -13, "el": -16,
             "hip": -1.0, "chest": 0.8, "twist": -0.7, "bob": 0.028},
        18: {"tL": -8, "tR": 3, "sL": -18, "sR": -5,
             "fL": 9, "fR": 3, "aL": 4, "aR": -4, "el": -18,
             "hip": 0.0, "chest": 0.0, "twist": 0.0, "bob": 0.038},
        21: {"tL": 12, "tR": -18, "sL": -10, "sR": -22,
             "fL": 2, "fR": 11, "aL": -13, "aR": 13, "el": -16,
             "hip": 1.0, "chest": -0.8, "twist": 0.7, "bob": 0.028},
        24: {"tL": stride, "tR": -stride, "sL": -8, "sR": -22,
             "fL": -5, "fR": 14, "aL": -19, "aR": 19, "el": -14,
             "hip": 1.6, "chest": -1.2, "twist": 1.0, "bob": 0.012},
    }
    for fr, p in poses.items():
        bpy.context.scene.frame_set(fr)
        for side in ("L", "R"):
            arm.pose.bones[f"thigh.{side}"].rotation_euler.x = math.radians(p["t" + side[0]])
            arm.pose.bones[f"shin.{side}"].rotation_euler.x = math.radians(p["s" + side[0]])
            arm.pose.bones[f"foot.{side}"].rotation_euler.x = math.radians(p["f" + side[0]])
            arm.pose.bones[f"upperarm.{side}"].rotation_euler.x = math.radians(p["a" + side[0]])
            arm.pose.bones[f"forearm.{side}"].rotation_euler.x = math.radians(p["el"])
            for bn in (f"thigh.{side}", f"shin.{side}", f"foot.{side}",
                       f"upperarm.{side}", f"forearm.{side}"):
                arm.pose.bones[bn].keyframe_insert("rotation_euler", frame=fr)
        arm.pose.bones["hips"].rotation_euler.y = math.radians(p["hip"])
        arm.pose.bones["hips"].rotation_euler.z = math.radians(p["twist"])
        arm.pose.bones["spine"].rotation_euler.y = math.radians(p["chest"])
        arm.pose.bones["spine"].rotation_euler.z = math.radians(-p["twist"] * .7)
        arm.pose.bones["hips"].keyframe_insert("rotation_euler", frame=fr)
        arm.pose.bones["spine"].keyframe_insert("rotation_euler", frame=fr)
        arm.pose.bones["root"].location = (0, 0, p["bob"])
        arm.pose.bones["root"].keyframe_insert("location", frame=fr)

    # 只对姿态曲线加循环修饰器；路径位移和转向曲线保持一次性播放。
    for fc in iter_fcurves(act):
        if fc.data_path.startswith('pose.bones['):
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'
                # Blender 5.2 将 AUTO_CLAMPED 暴露为控制柄类型，而非插值枚举。
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
            fc.modifiers.new('CYCLES')
    act.use_fake_user = True
    return act


def random_walk_path(start_id, hops, rng):
    """在走廊节点图上随机游走，返回节点 id 序列。"""
    cur = start_id
    path = [cur]
    prev = None
    for _ in range(hops):
        nbrs = [n for n in WP[cur]["links"] if n != prev]
        if not nbrs:
            nbrs = WP[cur]["links"]
        nxt = rng.choice(nbrs)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def animate_along_path(obj, path_ids, speed=1.1, start_frame=1):
    """让人物沿节点路径匀速行走，并在节点间连续转向。"""
    pts = [(WP[i]["x"], WP[i]["y"]) for i in path_ids]
    if len(pts) < 2:
        return start_frame
    yaws = []
    for i in range(len(pts) - 1):
        raw = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]) - math.pi / 2
        if yaws:
            while raw - yaws[-1] > math.pi:
                raw -= math.tau
            while raw - yaws[-1] < -math.pi:
                raw += math.tau
        yaws.append(raw)
    # 节点朝向取下一段方向；线性插值会在行进中提前完成转向，避免原地瞬转。
    node_yaws = yaws + [yaws[-1]]
    frames = [start_frame]
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        frames.append(frames[-1] + max(4, int(round(d / speed * 24))))
    obj.rotation_mode = "XYZ"
    for i, ((x, y), fr) in enumerate(zip(pts, frames)):
        obj.location = (x, y, 0.0)
        obj.rotation_euler = (0, 0, node_yaws[i])
        obj.keyframe_insert("location", frame=fr)
        obj.keyframe_insert("rotation_euler", frame=fr)
    ad = getattr(obj, "animation_data", None)
    for fc in iter_fcurves(ad.action if ad else None):
        if fc.data_path in {"location", "rotation_euler"}:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    return frames[-1]


def build_characters():
    """30 名员工就座 + N 名巡游者沿走廊行走。"""
    import random
    rng = random.Random(20261016)
    coll_seat = ensure_coll("chars_seated")
    coll_walk = ensure_coll("chars_walking")
    n_seat = n_walk = 0

    walker_ids = set()
    if N_WALKERS > 0 and STAFF:
        idxs = rng.sample(range(len(STAFF)), min(N_WALKERS, len(STAFF)))
        walker_ids = {STAFF[i]["employee_id"] for i in idxs}

    for st in STAFF:
        dept = st.get("department_key", "rnd")
        color = DEPT_COLOR.get(dept, "#4C7DFF")
        eid = st.get("employee_id", f"E{n_seat:03d}")
        name = f"char_{eid}"

        if eid in walker_ids:
            ob, arm = build_character(name, color, coll_walk,
                                      employee_id=eid, employee_name=st.get("name", ""))
            # 起点用离他部门最近的走廊节点
            r = ROOMS.get(dept)
            start = f"door_{dept}" if f"door_{dept}" in WP else "h2_0"
            if DO_ANIM:
                make_walk_action(arm, name=f"walk_{eid}")
                path = random_walk_path(start, hops=rng.randint(3, 6), rng=rng)
                animate_along_path(arm, path, speed=1.0 + rng.random() * 0.6,
                                   start_frame=rng.randint(1, 30))
            else:
                arm.location = (WP[start]["x"], WP[start]["y"], 0.0)
                arm.rotation_euler.z = 0.0
            n_walk += 1
        else:
            ob, arm = build_character(name, color, coll_seat,
                                      employee_id=eid, employee_name=st.get("name", ""))
            dk = next((d for d in DESKS if d["desk_id"] == st.get("desk_id")), None)
            if dk is None:
                dk = {"x": st["desk_x"], "y": st["desk_y"],
                      "rot": st.get("desk_rot", 0.0)}
            cx, cy, cr = chair_pos_for(dk)
            pose_seated(arm, cx, cy, cr)
            # 让同一工位的人略微错开，避免完全重叠
            arm.rotation_euler.z += rng.uniform(-0.14, 0.14)
            n_seat += 1
    # ---- 高管与前台（否则董事长室/总经理室/前台是空的，不像个公司）----
    extras = [
        # (名称, 上衣色, x, y, 朝向, 姿势)
        ("chairman",  "#2B3A55", 3.80, 20.55, math.pi, "sit"),   # 董事长室
        ("gm",        "#2B3A55", 12.00, 20.85, math.pi, "sit"),  # 总经理室
        ("reception", "#C97B84", 21.80, 4.60, math.pi, "sit"),   # 前台
        ("walker_guest", "#4A5568", 21.80, 1.60, 0.0, "stand"),  # 门口访客
    ]
    for nm, col, x, y, rot, pose in extras:
        ob, arm = build_character(f"char_{nm}", col, coll_seat,
                                  employee_id=None, employee_name=nm)
        if pose == "sit":
            pose_seated(arm, x, y, rot)
        else:
            arm.location = (x, y, 0.0)
            arm.rotation_euler = (0, 0, rot)
        n_seat += 1

    return n_seat, n_walk


# ================================================================ 建筑
def build_shell(coll, ceil_coll):
    def bx(nm, cx, cy, cz, sx, sy, sz, m, c):
        return mesh_from(nm, box_data(cx, cy, cz, sx, sy, sz), m, c)

    bx("slab", FW/2, FD/2, -0.15, FW, FD, 0.3, M["slab"], coll)

    # 吊顶做成「单面朝下的面片 + 背面剔除」：
    #   从室内往上看 → 能看到吊顶（不再是漆黑天空）
    #   从上方俯瞰   → 背面被剔除，能直接看进房间
    cme = bpy.data.meshes.new("ceiling_plane")
    cme.from_pydata([(0, 0, CH), (FW, 0, CH), (FW, FD, CH), (0, FD, CH)],
                    [], [(3, 2, 1, 0)])          # 顶点逆序 → 法线朝下
    cme.update()
    cmat = mat("mat_ceiling_oneside", "#FAFBFD", rough=.95)
    cmat.use_backface_culling = True
    cme.materials.append(cmat)
    cp = bpy.data.objects.new("ceiling", cme)
    ceil_coll.objects.link(cp)
    # 外墙：南北实墙，东西玻璃幕墙
    bx("wall_s", FW/2, -WT/2, CH/2, FW, WT, CH, M["wall"], coll)
    bx("wall_n", FW/2, FD+WT/2, CH/2, FW, WT, CH, M["wall"], coll)
    bx("wall_w", -WT/2, FD/2, CH/2, WT, FD, CH, M["glass"], coll)
    bx("wall_e", FW+WT/2, FD/2, CH/2, WT, FD, CH, M["glass"], coll)
    n = max(1, int(FD // 4))
    for i in range(n + 1):
        y = i * (FD / n)
        bx(f"mullion_{i}", FW + 0.1, y, CH/2, 0.12, 0.12, CH, M["mullion"], coll)
    # 分户地板（按房间类型换材质）
    for r in ROOMS.values():
        mk = "carpet"
        if r["kind"] == "office":
            mk = "wood"
        elif r["rid"] in ("pantry", "restroom"):
            mk = "tile"
        elif r["kind"] == "meeting":
            mk = "wood"
        bx(f"floor_{r['rid']}", r["x"]+r["w"]/2, r["y"]+r["d"]/2, 0.012,
           r["w"], r["d"], 0.024, M[mk], coll)


def build_room_walls(coll):
    """房间隔墙 + 门洞（在靠走廊一侧留口）。"""
    def bx(nm, cx, cy, cz, sx, sy, sz, c):
        return mesh_from(nm, box_data(cx, cy, cz, sx, sy, sz), M["wall"], c)

    DOOR = L["floor"]["door_width"]
    for r in ROOMS.values():
        x0, y0, w, d = r["x"], r["y"], r["w"], r["d"]
        # 四边墙，逐段避开门口
        for (side, hx, hy, sw, sd) in [
            ("s", x0+w/2, y0, w, WT),
            ("n", x0+w/2, y0+d, w, WT),
            ("w", x0, y0+d/2, WT, d),
            ("e", x0+w, y0+d/2, WT, d),
        ]:
            if side == r.get("door", "n"):
                # 门口居中留洞：拆成左右两段
                if sw > sd:
                    seg = (sw - DOOR) / 2
                    bx(f"w_{r['rid']}_{side}_a", hx - (DOOR+seg)/2, hy, CH/2,
                       seg, sd, CH, coll)
                    bx(f"w_{r['rid']}_{side}_b", hx + (DOOR+seg)/2, hy, CH/2,
                       seg, sd, CH, coll)
                    # 门楣
                    bx(f"w_{r['rid']}_{side}_top", hx, hy, 2.35, DOOR, sd, 0.5, coll)
                else:
                    seg = (sd - DOOR) / 2
                    bx(f"w_{r['rid']}_{side}_a", hx, hy - (DOOR+seg)/2, CH/2,
                       sw, seg, CH, coll)
                    bx(f"w_{r['rid']}_{side}_b", hx, hy + (DOOR+seg)/2, CH/2,
                       sw, seg, CH, coll)
                    bx(f"w_{r['rid']}_{side}_top", hx, hy, 2.35, sw, DOOR, 0.5, coll)
            else:
                bx(f"w_{r['rid']}_{side}", hx, hy, CH/2, sw, sd, CH, coll)


# ================================================================ 灯光 / 相机
def build_lights(coll):
    def add(nm, t, loc, e, size=20.0, rot=(0, 0, 0)):
        bpy.ops.object.light_add(type=t, location=loc)
        o = bpy.context.active_object
        o.name = nm
        o.data.energy = e
        o.rotation_euler = rot
        if t == "AREA":
            o.data.size = size
        if t == "SUN":
            o.data.angle = math.radians(5)
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)

    add("sun", "SUN", (FW*0.9, -FD*0.5, 40), 4.0,
        rot=(math.radians(48), math.radians(4), math.radians(35)))
    # 分区吊顶灯盘：功率按房间面积给足，否则 EEVEE 下室内发灰
    for nm, x, y, e, sz in [
        ("fill_n1", 4.4, 20.5, 2000, 20), ("fill_n2", 12.0, 20.5, 1500, 18),
        ("fill_n3", 20.1, 20.5, 2200, 20), ("fill_n4", 29.4, 20.5, 2000, 20),
        ("fill_m1", 8.0, 11.3, 3200, 26), ("fill_m2", 22.6, 11.3, 1800, 20),
        ("fill_m3", 30.4, 11.3, 1500, 18),
        ("fill_s1", 5.0, 3.0, 1800, 20), ("fill_s2", 13.8, 3.0, 1400, 18),
        ("fill_s3", 21.8, 3.0, 1600, 18), ("fill_s4", 27.8, 4.6, 700, 12),
        ("fill_corr", 17.2, 16.0, 1600, 30), ("fill_corr2", 17.2, 7.0, 1400, 30),
    ]:
        add(nm, "AREA", (x, y, CH - 0.25), e, sz)


def build_cameras(coll):
    def add(nm, loc, tgt, lens=32):
        bpy.ops.object.camera_add(location=loc)
        c = bpy.context.active_object
        c.name = nm
        d = Vector(tgt) - Vector(loc)
        c.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        c.data.lens = lens
        c.data.clip_end = 2000
        for cc in list(c.users_collection):
            cc.objects.unlink(c)
        coll.objects.link(c)
        return c

    bpy.context.scene.camera = add("cam_overview", (FW*0.95, -FD*0.7, 30),
                                   (FW/2, FD/2, 1.2), 30)
    add("cam_corridor", (17.2, 8.0, 1.65), (17.2, 22, 1.5), 24)
    add("cam_chairman", (4.4, 15.2, 1.7), (5.0, 21, 1.3), 26)
    add("cam_open", (10, 9.0, 1.7), (14, 15, 1.4), 26)
    add("cam_top", (FW*0.5, -FD*0.35, FD*1.5), (FW/2, FD/2, 1.0), 40)
    add("cam_reception", (21.8, 2.0, 1.7), (21.8, 14, 1.4), 24)


# ================================================================ 主流程
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.render.fps = 24
    build_materials()

    print(f"[build] LOD={LOD} 巡游人物={N_WALKERS} 人物资产={INCLUDE_CHARACTERS} 动画={DO_ANIM}")

    c_shell = ensure_coll("shell")
    c_ceil = ensure_coll("ceiling")
    c_wall = ensure_coll("walls")
    build_shell(c_shell, c_ceil)
    print("[build] 楼板 / 幕墙 / 房间地板 完成")
    build_room_walls(c_wall)
    print(f"[build] 房间隔墙（{len(ROOMS)} 间，带门洞）完成")

    nf = build_furniture()
    print(f"[build] 家具 {nf} 件 完成")

    if INCLUDE_CHARACTERS:
        ns, nw = build_characters()
        print(f"[build] 人物 就座 {ns} / 巡游 {nw} 完成")
    else:
        print("[build] 环境资产模式：跳过 Blender 人物，网页由 sim.js 驱动动态角色")

    build_lights(ensure_coll("lighting"))
    build_cameras(ensure_coll("cameras"))
    print("[build] 灯光 / 相机 完成")

    scn.render.resolution_x = 1920
    scn.render.resolution_y = 1080
    if DO_ANIM:
        scn.frame_start = 1
        # 路径时长可能超过原先固定的 240 帧；以所有位置曲线的末帧为准。
        max_frame = 24
        for ob in bpy.data.objects:
            ad = getattr(ob, "animation_data", None)
            act = getattr(ad, "action", None) if ad else None
            for fc in iter_fcurves(act):
                if fc.keyframe_points:
                    max_frame = max(max_frame, int(max(k.co.x for k in fc.keyframe_points)))
        scn.frame_end = max(240, max_frame)
    try:
        scn.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scn.render.engine = "BLENDER_EEVEE"
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.04, 0.05, 0.08, 1)
        bg.inputs[1].default_value = 1.0

    out = ARGS["out"]
    if out:
        p = Path(out)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(p))
        print(f"[build] 已保存: {p}")
        g = p.with_suffix(".glb")
        bpy.ops.export_scene.gltf(filepath=str(g), export_format="GLB",
                                  export_animations=True)
        print(f"[build] 已导出: {g}")

    print(f"[build] 完成 — 对象 {len(bpy.data.objects)}，"
          f"材质 {len(bpy.data.materials)}")


if __name__ == "__main__":
    main()
