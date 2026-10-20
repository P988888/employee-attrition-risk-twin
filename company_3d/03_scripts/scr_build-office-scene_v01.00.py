"""办公楼室内场景构建脚本（Blender 4.x）

用途
----
读取 `data/office_layout.json`（由 src/step7_office_layout.py 生成的真实平面图纸），
在 Blender 里自动搭出办公楼的**场景骨架**：
    楼板 / 外墙 / 玻璃幕墙 / 结构柱 / 隔墙
    四个部门工位区（含桌椅、显示器、椅子）
    会议室 / 前台 / 休息区 / 电话间 / 打印区 / 储物间 / 卫生间
    部门分区色块地面 + 空间文字标签
    可选：在每个工位放置人物占位体（后续替换为真实绑定角色）

坐标系
------
与布局 JSON 一致：X = 宽度方向，Y = 进深方向，Z = 高度（向上），单位米。
Blender 默认即为 Z-up，因此坐标可直接使用，无需转换。

用法
----
1) Blender 内运行：
     打开 Blender → Scripting 工作区 → 打开本脚本 → Run Script
     （脚本顶部 DEFAULT_LAYOUT 指向布局 JSON，若路径不同请修改）

2) 命令行运行（无界面，推荐）：
     blender --background --python scr_build-office-scene_v01.00.py -- \
         --layout ../../data/office_layout.json \
         --out ../02_scenes/wip/scn_office-hq_blockout_v01.00_wip.blend \
         --floor A栋-3F \
         --lod blockout

参数
----
--layout   布局 JSON 路径（默认 ../../data/office_layout.json）
--out      输出 .blend 路径
--floor    只构建指定楼层（默认 A栋-3F，即主展示层）
--lod      blockout（仅体块，快）| detailed（含桌椅显示器椅子，慢）
--people   是否放置人物占位体（1/0，默认 0）

注意
----
本脚本产出的是**体块级别的场景骨架**，用于确定空间关系、尺度与数据绑定；
精细模型（真实办公家具、幕墙节点、灯具、材质贴图）需在 Blender 中手工制作，
详见 company_3d/06_docs/scene-spec-office-hq.md 的资产清单。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# ------------------------------------------------------------------ 参数
DEFAULT_LAYOUT = "../../data/office_layout.json"


def parse_args() -> dict:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {"layout": DEFAULT_LAYOUT, "out": None, "floor": None,
            "lod": "blockout", "people": "0"}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            val = argv[i + 1] if i + 1 < len(argv) else ""
            if key in args:
                args[key] = val
            i += 2
        else:
            i += 1
    return args


ARGS = parse_args()
HERE = Path(__file__).resolve().parent

LAYOUT_PATH = (HERE / ARGS["layout"]).resolve() if not os.path.isabs(ARGS["layout"]) \
    else Path(ARGS["layout"])
if not LAYOUT_PATH.exists():
    LAYOUT_PATH = (HERE.parent.parent / "data" / "office_layout.json").resolve()

LOD = ARGS["lod"]
PLACE_PEOPLE = ARGS["people"] == "1"

print(f"[build] 布局文件: {LAYOUT_PATH}")
with open(LAYOUT_PATH, encoding="utf-8") as f:
    LAYOUT = json.load(f)

FLOOR = LAYOUT["floor"]
MOD = LAYOUT["desk_module"]
ZONES = LAYOUT["zones"]
DEPTS = {d["key"]: d for d in LAYOUT["departments"]}
TARGET_FLOOR = ARGS["floor"] or FLOOR["names"][0]

FW, FD, CH = FLOOR["width"], FLOOR["depth"], FLOOR["ceiling_height"]
WALL_T = FLOOR["wall_thickness"]
SLAB_T = FLOOR["slab_thickness"]


# ------------------------------------------------------------------ 工具
def hex2rgba(h: str, a: float = 1.0) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255, a)


def ensure_collection(name: str):
    """取或建集合（用于分组管理场景对象）。"""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def make_material(name: str, color_hex: str, alpha: float = 1.0,
                  roughness: float = 0.55, metallic: float = 0.0,
                  emission: float = 0.0):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    rgba = hex2rgba(color_hex, alpha)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    if alpha < 1.0:
        # Blender 4.2+ (EEVEE Next) 移除了 blend_method，改用 surface_render_method
        if hasattr(mat, "blend_method"):
            try:
                mat.blend_method = "BLEND"
            except Exception:
                pass
        elif hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
    return mat


def make_box_mesh(name: str, sx: float, sy: float, sz: float, material=None):
    """用 from_pydata 直接建一个长方体**原型网格**（不走 bpy.ops，快得多）。

    后续所有同类物体都用「链接复制」(linked duplicate) 复用这份网格数据，
    这样 351 个工位只需 1 份网格数据 + 351 个轻量对象，文件与渲染都轻。
    """
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if material:
        me.materials.append(material)
    return me


def place(mesh, name: str, cx: float, cy: float, cz: float,
          rot: float = 0.0, collection=None):
    """用已有网格数据建一个实例对象（复用 mesh，不复制几何）。"""
    obj = bpy.data.objects.new(name, mesh)      # 共享 mesh -> 轻量
    obj.location = (cx, cy, cz)
    obj.rotation_euler[2] = rot
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def box(name: str, cx: float, cy: float, cz: float,
        sx: float, sy: float, sz: float,
        material=None, collection=None):
    """一次性长方体（少量物体用，走 bpy.ops 便于后续手工编辑）。"""
    me = make_box_mesh(f"{name}_mesh", sx, sy, sz, material)
    return place(me, name, cx, cy, cz, 0.0, collection)


def add_text_label(text: str, x: float, y: float, z: float,
                   size: float = 1.6, color_hex: str = "#1A2A3A",
                   collection=None):
    """空间中的三维文字标签（用于分区标识）。"""
    bpy.ops.object.text_add(location=(x, y, z))
    obj = bpy.context.active_object
    obj.data.body = text
    obj.data.size = size
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    mat = make_material(f"mat_label_{color_hex.lstrip('#')}", color_hex,
                        roughness=0.9, emission=0.4)
    obj.data.materials.append(mat)
    if collection:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        collection.objects.link(obj)
    return obj


# ------------------------------------------------------------------ 材质
# 注意：材质必须**在 clear_scene() 之后**创建。
# read_factory_settings() 会清空整个 bpy.data，若提前建材质，
# 之后引用它会抛 "StructRNA of type Material has been removed"。
M: dict = {}


def build_materials():
    """创建全部材质（须在清场之后调用）。"""
    M.clear()
    M.update({
        "slab":     make_material("mat_floor_slab", "#E8EAF0", roughness=0.85),
        "wall":     make_material("mat_wall_white", "#F2F4F8", roughness=0.9),
        "glass":    make_material("mat_glass_curtain", "#A8D2F0", alpha=0.28,
                                  roughness=0.08, metallic=0.0),
        "mullion":  make_material("mat_mullion_alu", "#7C8798", roughness=0.35,
                                  metallic=0.8),
        "column":   make_material("mat_column_concrete", "#C4C9D2", roughness=0.75),
        "ceiling":  make_material("mat_ceiling", "#FFFFFF", roughness=0.95),
        "corridor": make_material("mat_floor_corridor", "#D8DEE9", roughness=0.7),
        "desk_top": make_material("mat_desk_top", "#C89B6A", roughness=0.5),
        "desk_leg": make_material("mat_desk_leg", "#5A6570", roughness=0.4,
                                  metallic=0.6),
        "screen":   make_material("mat_monitor_screen", "#1A2230", roughness=0.25,
                                  emission=0.3),
        "chair":    make_material("mat_chair", "#3A4250", roughness=0.65),
        "person":   make_material("mat_person_proxy", "#8FA8D8", alpha=0.75,
                                  roughness=0.8),
    })
    for k, d in DEPTS.items():
        M[f"zone_{k}"] = make_material(f"mat_zone_{k}", d["color"], alpha=0.30,
                                       roughness=0.85, emission=0.15)
    M["zone_facility"] = make_material("mat_zone_facility", "#8FA8D8", alpha=0.35,
                                       roughness=0.85)


# ------------------------------------------------------------------ 清场
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0


# ------------------------------------------------------------------ 构建
def build_shell(col, ceiling_col=None):
    """楼板 + 外墙 + 玻璃幕墙 + 吊顶。"""
    # 楼板（顶面为 z=0，即楼层地面）
    box("slab_floor", FW / 2, FD / 2, -SLAB_T / 2,
        FW, FD, SLAB_T, M["slab"], col)

    # 吊顶单独放一个集合：俯视总览时它会挡住整个室内，
    # 因此默认设置为「不参与渲染」，需要时可手动打开。
    ccol = ceiling_col or col
    ceil = box("ceiling", FW / 2, FD / 2, CH + 0.05,
               FW, FD, 0.1, M["ceiling"], ccol)
    ceil.hide_render = True
    ceil.hide_viewport = True

    # 外墙（南北实墙）
    box("wall_south", FW / 2, -WALL_T / 2, CH / 2, FW, WALL_T, CH, M["wall"], col)
    box("wall_north", FW / 2, FD + WALL_T / 2, CH / 2, FW, WALL_T, CH, M["wall"], col)
    # 东西玻璃幕墙
    box("wall_west", -WALL_T / 2, FD / 2, CH / 2, WALL_T, FD, CH, M["glass"], col)
    box("wall_east", FW + WALL_T / 2, FD / 2, CH / 2, WALL_T, FD, CH, M["glass"], col)

    # 幕墙竖挺（东立面，8m 间距）
    n = max(1, int(FD // 8))
    for i in range(n + 1):
        y = i * (FD / n)
        box(f"mullion_e_{i:02d}", FW + 0.15, y, CH / 2, 0.16, 0.16, CH,
            M["mullion"], col)

    # 入口门框（南侧中部，宽 6m）
    box("door_frame", FW / 2, -WALL_T / 2, 1.5, 6.0, 0.35, 3.0, M["mullion"], col)


def build_columns(col):
    for i, c in enumerate(LAYOUT["columns"]):
        s = c["size"]
        box(f"column_{i:02d}", c["x"], c["y"], c["height"] / 2,
            s, s, c["height"], M["column"], col)


def build_zone_floor(col):
    """分区地面色块 —— 让四个部门在视觉上一眼可分。"""
    for z in ZONES:
        if z.get("dept_key"):
            mat = M[f"zone_{z['dept_key']}"]
        else:
            mat = M["zone_facility"]
        box(f"zonefloor_{z['zone_id']}",
            z["x"] + z["w"] / 2, z["y"] + z["d"] / 2, 0.02,
            z["w"], z["d"], 0.04, mat, col)


def build_corridor(col):
    """主通道（与平面图一致的十字通道）。"""
    box("corridor_h", (0 + 50) / 2, 22.5, 0.03, 50, 3.0, 0.05,
        M["corridor"], col)
    box("corridor_v", 23.75, (6 + 44) / 2, 0.03, 3.0, 38, 0.05,
        M["corridor"], col)


def build_partition_walls(col):
    """配套房间的隔墙（会议室/电话间/储物间/卫生间）。"""
    for z in ZONES:
        if z.get("kind") != "facility":
            continue
        # 跳过前台与休息区（开放空间，不做封闭隔墙）
        if z["zone_id"] in ("f_reception", "f_break"):
            continue
        x0, y0, w, d = z["x"], z["y"], z["w"], z["d"]
        box(f"part_{z['zone_id']}_n", x0 + w / 2, y0 + d, CH / 2,
            w, WALL_T, CH, M["wall"], col)
        box(f"part_{z['zone_id']}_s", x0 + w / 2, y0, CH / 2,
            w, WALL_T, CH, M["wall"], col)
        box(f"part_{z['zone_id']}_w", x0, y0 + d / 2, CH / 2,
            WALL_T, d, CH, M["wall"], col)
        box(f"part_{z['zone_id']}_e", x0 + w, y0 + d / 2, CH / 2,
            WALL_T, d, CH, M["wall"], col)


def build_desk(desk, dept_key, col, prefix=""):
    """单个工位：桌板 + 桌腿 + 显示器 + 椅子。"""
    x, y, rot = desk["x"], desk["y"], desk.get("rot", 0.0)
    dw, dd, dh = MOD["desk_width"], MOD["desk_depth"], 0.74

    # 桌板
    top = box(f"{prefix}desk_{desk['desk_id']}", x, y, dh, dw, dd, 0.04,
              M["desk_top"], col)
    top.rotation_euler[2] = rot
    # 桌腿
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        box(f"{prefix}leg_{desk['desk_id']}_{sx}{sy}",
            x + sx * (dw / 2 - 0.06), y + sy * (dd / 2 - 0.06), dh / 2,
            0.05, 0.05, dh, M["desk_leg"], col)
    # 显示器
    box(f"{prefix}mon_{desk['desk_id']}", x, y - dd / 2 + 0.12, dh + 0.26,
        0.62, 0.03, 0.38, M["screen"], col)
    box(f"{prefix}monstand_{desk['desk_id']}", x, y - dd / 2 + 0.12, dh + 0.06,
        0.18, 0.14, 0.12, M["desk_leg"], col)
    # 椅子（在工位朝向侧）
    import math as _m
    cy_off = _m.cos(rot) * (dd / 2 + 0.32)
    cx_off = -_m.sin(rot) * (dd / 2 + 0.32)
    box(f"{prefix}chair_{desk['desk_id']}", x + cx_off, y + cy_off, 0.45,
        0.5, 0.5, 0.06, M["chair"], col)
    box(f"{prefix}chairback_{desk['desk_id']}", x + cx_off, y + cy_off + 0.2, 0.62,
        0.48, 0.06, 0.5, M["chair"], col)


def build_desks(col):
    """按部门构建全部工位。

    性能关键：先为每种部件各建**一份原型网格**，再对所有工位实例化复用。
    351 工位 × 8 部件若逐个 bpy.ops 建，会有约 2800 次 API 调用而极慢；
    这里只需 8 份网格 + 对象实例，秒级完成。
    """
    dw, dd, dh = MOD["desk_width"], MOD["desk_depth"], 0.74

    # 原型网格（每种一份）
    proto = {
        "desk":   make_box_mesh("proto_desk", dw, dd, 0.04, M["desk_top"]),
        "leg":    make_box_mesh("proto_leg", 0.05, 0.05, dh, M["desk_leg"]),
        "mon":    make_box_mesh("proto_mon", 0.62, 0.03, 0.38, M["screen"]),
        "stand":  make_box_mesh("proto_stand", 0.18, 0.14, 0.12, M["desk_leg"]),
        "seat":   make_box_mesh("proto_seat", 0.5, 0.5, 0.06, M["chair"]),
        "back":   make_box_mesh("proto_back", 0.48, 0.06, 0.5, M["chair"]),
    }
    blk = {k: make_box_mesh(f"proto_blk_{k}", dw, dd, 0.72, M[f"zone_{k}"])
           for k in DEPTS}

    made = 0
    dep_cols = {}
    for desk in LAYOUT["desks"]:
        k = desk["dept_key"]
        if k not in dep_cols:
            dep_cols[k] = ensure_collection(f"desks_{k}")
        dep = dep_cols[k]
        x, y, rot = desk["x"], desk["y"], desk.get("rot", 0.0)
        did = desk["desk_id"]

        if LOD == "blockout":
            place(blk[k], f"deskblk_{did}", x, y, 0.36, rot, dep)
            made += 1
            continue

        place(proto["desk"], f"desk_{did}", x, y, dh, rot, dep)
        for sx in (-1, 1):
            for sy in (-1, 1):
                place(proto["leg"], f"leg_{did}_{sx}{sy}",
                      x + sx * (dw / 2 - 0.06), y + sy * (dd / 2 - 0.06),
                      dh / 2, rot, dep)
        place(proto["mon"], f"mon_{did}", x, y - dd / 2 + 0.12, dh + 0.26, rot, dep)
        place(proto["stand"], f"stand_{did}", x, y - dd / 2 + 0.12, dh + 0.06, rot, dep)

        off = dd / 2 + 0.32
        cx = x - math.sin(rot) * off
        cy = y + math.cos(rot) * off
        place(proto["seat"], f"chair_{did}", cx, cy, 0.45, rot, dep)
        place(proto["back"], f"chairback_{did}", cx, cy + 0.2, 0.62, rot, dep)
        made += 1
    return made


def build_people(col):
    """每个工位放一个半透明人物占位体（后续替换为绑定角色）。"""
    body = make_box_mesh("proto_person_body", 0.42, 0.30, 1.30, M["person"])
    head = make_box_mesh("proto_person_head", 0.22, 0.22, 0.24, M["person"])
    n = 0
    for desk in LAYOUT["desks"]:
        x, y, rot = desk["x"], desk["y"], desk.get("rot", 0.0)
        off = MOD["desk_depth"] / 2 + 0.32
        px = x - math.sin(rot) * off
        py = y + math.cos(rot) * off
        place(body, f"person_{desk['desk_id']}", px, py, 0.85, rot, col)
        place(head, f"head_{desk['desk_id']}", px, py, 1.62, rot, col)
        n += 1
    return n


def build_labels(col):
    """分区文字标签。

    每个分区只放**一个**标签，悬浮在 z=3.9（人群与家具之上），
    这样俯视/等轴测总览时不会被桌子挡住，也不会出现重影。
    部门分区用大字号，配套小房间用小字号以免互相压盖。
    """
    for z in ZONES:
        label = z.get("label_cn") or z.get("label_en") or z["zone_id"]
        cx, cy = z["x"] + z["w"] / 2, z["y"] + z["d"] / 2
        size = 3.0 if z.get("kind") == "dept" else 1.5
        zpos = 3.9 if z.get("kind") == "dept" else 3.5
        add_text_label(label, cx, cy, zpos, size=size,
                       color_hex="#123A6B", collection=col)


# ------------------------------------------------------------------ 主流程
def main():
    clear_scene()
    build_materials()
    print(f"[build] 目标楼层: {TARGET_FLOOR} | LOD: {LOD} | 人物占位: {PLACE_PEOPLE}")

    col_shell = ensure_collection("shell")
    col_struct = ensure_collection("structure")
    col_zones = ensure_collection("zone_floors")
    col_parts = ensure_collection("partitions")
    col_people = ensure_collection("people")
    col_labels = ensure_collection("labels")

    build_shell(col_shell)
    print("[build] 楼板 / 外墙 / 幕墙 / 吊顶 完成")

    build_columns(col_struct)
    print(f"[build] 结构柱 {len(LAYOUT['columns'])} 根 完成")

    build_zone_floor(col_zones)
    build_corridor(col_zones)
    print(f"[build] 分区地面 {len(ZONES)} 块 + 主通道 完成")

    build_partition_walls(col_parts)
    print("[build] 配套房间隔墙 完成")

    n = build_desks(col_shell)
    print(f"[build] 工位 {n} 个 完成（LOD={LOD}）")

    if PLACE_PEOPLE:
        p = build_people(col_people)
        print(f"[build] 人物占位体 {p} 个 完成")

    build_labels(col_labels)
    print("[build] 分区标签 完成")

    # ---------------- 灯光（基础布光，便于预览）----------------
    # 吊顶默认隐藏，因此室内只靠这几盏灯；
    # 主光偏暖、补光偏冷，避免室内死黑。
    col_light = ensure_collection("lighting")

    def add_light(name, ltype, loc, energy, size=20.0, rot=(0, 0, 0)):
        bpy.ops.object.light_add(type=ltype, location=loc)
        lo = bpy.context.active_object
        lo.name = name
        lo.data.energy = energy
        lo.rotation_euler = rot
        if ltype == "AREA":
            lo.data.size = size
        if ltype == "SUN":
            lo.data.angle = math.radians(6)
        for cc in list(lo.users_collection):
            cc.objects.unlink(lo)
        col_light.objects.link(lo)
        return lo

    # 太阳（主光，从东南上方斜射，穿透玻璃幕墙）
    add_light("sun_key", "SUN", (FW * 0.9, -FD * 0.4, 45), 3.5,
              rot=(math.radians(52), math.radians(6), math.radians(38)))
    # 室内面光（分区补光，模拟吊顶灯盘）
    add_light("fill_marketing", "AREA", (12, 33, CH - 0.25), 900, size=22)
    add_light("fill_rnd", "AREA", (37, 33, CH - 0.25), 1100, size=24)
    add_light("fill_ops", "AREA", (38, 14, CH - 0.25), 800, size=20)
    add_light("fill_finance", "AREA", (13, 14, CH - 0.25), 700, size=20)
    add_light("fill_lobby", "AREA", (32, 3, CH - 0.25), 500, size=18)
    # 走廊条形补光
    add_light("fill_corridor", "AREA", (25, 22.7, CH - 0.3), 600, size=26)

    # ---------------- 相机 ----------------
    # 用 to_track_quat 精确对准目标点，避免手填欧拉角导致「拍到场外」
    col_cam = ensure_collection("cameras")

    def add_cam(name, loc, target, lens=35):
        bpy.ops.object.camera_add(location=loc)
        c = bpy.context.active_object
        c.name = name
        d = Vector(target) - Vector(loc)
        c.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        c.data.lens = lens
        c.data.clip_end = 1000
        for cc in list(c.users_collection):
            cc.objects.unlink(c)
        col_cam.objects.link(c)
        return c

    C = (FW / 2, FD / 2, 1.2)        # 楼层中心

    # 主相机：等轴测总览（能看到四个分区）
    bpy.context.scene.camera = add_cam(
        "cam_iso_overview", (FW * 0.95, -FD * 0.55, FD * 0.80), C, lens=32)
    # 高处俯瞰（更接近平面图视角）
    add_cam("cam_top_high", (FW * 0.52, -FD * 0.42, FD * 1.45), C, lens=40)
    # 入口人视角：必须放在**室内**（Y>0），否则只能拍到南侧外墙
    add_cam("cam_entry", (FW / 2, 6.5, 2.7), (FW / 2, FD * 0.75, 1.5), lens=24)
    # 研发区特写（从东侧上方看向研发工位区中心）
    add_cam("cam_rnd", (FW * 0.98, 6, FD * 0.75), (37, 33.5, 1.2), lens=38)

    # ---------------- 渲染设置 ----------------
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in \
        [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] \
        else "BLENDER_EEVEE"
    scn.render.resolution_x = 1920
    scn.render.resolution_y = 1080
    scn.render.film_transparent = False
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.05, 0.06, 0.09, 1.0)
        bg.inputs[1].default_value = 1.0

    # ---------------- 保存 ----------------
    out = ARGS["out"]
    if out:
        out_path = Path(out)
        # 相对路径按**当前工作目录**解析（与调用方的直觉一致），
        # 不要按脚本所在目录解析，否则会意外往 03_scripts/ 里建目录树。
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
        print(f"[build] 已保存: {out_path}")

        gltf = out_path.with_suffix(".glb")
        bpy.ops.export_scene.gltf(filepath=str(gltf), export_format="GLB")
        print(f"[build] 已导出: {gltf}")
    else:
        print("[build] 未指定 --out，场景仅存在于当前会话（未保存）")

    total = len(bpy.data.objects)
    print(f"[build] 完成 — 对象总数 {total}")


if __name__ == "__main__":
    main()
