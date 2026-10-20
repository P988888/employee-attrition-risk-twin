"""Step 7 公司总部办公层布局（单层 · 42 人）

建筑
----
占地 40 × 26 m，**单层**，层高 3.6 m，总建筑面积 1040 m²。
中部是一条 3 m 宽的横向主走廊，南北两侧房间都开门接走廊；
单层平面不设电梯，水平动线就是全部动线。

部门与工位（42 人全部在同一层）
------------------------------
  北区（靠窗大空间）：研发部 10 · 市场部 7 · 运营部 7
  南区（靠门职能区）：客服中心 6 · 财务部 4 · 人力与行政 2 · 高管 2 · 前台接待 4
  公共设施：大/小会议室 · 茶水间 · 咖啡吧 · 专注电话间 · 打印储物 · 机房 · 安保

产出
----
  data/building_layout.json     完整平面规格（房间/工位/家具/路径）
  data/building_desks.csv       42 名员工 → 工位 绑定
  data/building_furniture.csv   全部家具
  output/figures/18_大楼各层平面图.png
  output/figures/19_大楼3D预览.png
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Patch

from config import DATA, FIGURES, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)
_CJK = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Hiragino Sans GB"]
_av = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in _CJK if f in _av), "Arial Unicode MS")
plt.rcParams["font.sans-serif"] = [CJK] + _CJK
plt.rcParams["axes.unicode_minus"] = False

# ================================================================ 建筑参数
FW, FD = 40.0, 26.0          # 占地（单层要装下 42 个工位 + 全部配套，比四层版放宽）
FLOOR_H = 3.6                # 层高
SLAB = 0.35                  # 楼板厚
WALL = 0.2
DOOR = 1.1
N_FLOORS = 1

# 走廊（横向主通道；单层不需要电梯，中段留出公共区横带）
CORR_Y0, CORR_Y1 = 10.5, 13.5

DESK_W, DESK_D = 1.5, 0.75
# 单层要在一层里排下 42 个工位，工位间距比四层版略紧（仍留出正常通行和椅子空间）
PITCH_X, PITCH_Y = 1.55, 1.45

# 部门（含配色）
DEPT = {
    "service":   {"cn": "客服中心",     "en": "Customer Service", "color": "#7C6BD6"},
    "rnd":       {"cn": "研发部",       "en": "R&D",              "color": "#4C7DFF"},
    "marketing": {"cn": "市场部",       "en": "Marketing",        "color": "#FF6B35"},
    "ops":       {"cn": "运营部",       "en": "Operations",       "color": "#28C76F"},
    "finance":   {"cn": "财务部",       "en": "Finance",          "color": "#FFB020"},
    "hr":        {"cn": "人力与行政",   "en": "HR & Admin",       "color": "#E85D8A"},
    "exec":      {"cn": "高管办公区",   "en": "Executive",        "color": "#2B3A55"},
}
ROOM_COLOR = {"office": "#8E7CC3", "meeting": "#5B8FF9", "facility": "#9AA7B8",
              "public": "#C9A227"}

# 各部门的岗位池 —— 职位必须与部门匹配，不能直接从 IBM 数据乱采样
# （否则会出现「研发部挂着 Sales Executive」这种不合理结果）
ROLES_BY_DEPT = {
    "service":   ["客服专员", "客服主管", "呼叫中心坐席", "客户关系专员", "售后支持"],
    "rnd":       ["软件工程师", "高级工程师", "技术负责人", "系统架构师",
                  "测试工程师", "产品经理", "算法工程师"],
    "marketing": ["市场营销经理", "市场专员", "品牌经理", "内容运营", "市场分析师"],
    "ops":       ["运营经理", "运营专员", "供应链专员", "客户成功经理", "数据分析师"],
    "finance":   ["会计", "财务分析师", "财务经理", "审计专员", "出纳"],
    "hr":        ["人力资源专员", "招聘主管", "行政专员", "薪酬绩效专员", "HRBP"],
    "exec":      ["董事长", "总经理", "副总经理", "技术总监", "市场总监",
                  "财务总监", "运营总监"],
}

# 3D 场景中的姓名使用确定性的模拟中文姓名，避免把「客服中001」这类
# 布局编号直接暴露给用户。性别沿用行为链路中的 Male/Female 字段，
# step8 会把同一字段写入员工档案，保证姓名、性别和档案一致。
SURNAME_POOL = ["陈", "林", "张", "李", "王", "赵", "黄", "周", "吴", "徐",
                "孙", "胡", "朱", "高", "何", "郭", "马", "罗", "梁", "宋",
                "郑", "谢", "韩", "唐", "冯", "于", "董", "萧", "程", "沈"]
GIVEN_MALE = ["子轩", "浩然", "宇航", "俊杰", "嘉豪", "博文", "明远", "泽宇",
              "天佑", "昊霖", "奕辰", "景行", "思远", "承宇", "睿哲", "文昊", "嘉言"]
GIVEN_FEMALE = ["语嫣", "欣怡", "梓涵", "若曦", "诗涵", "雨桐", "芷晴", "思妍",
                "婉婷", "可欣", "静怡", "梦瑶", "安琪", "雅雯", "心怡", "沐晴", "知夏"]


def simulated_name(index: int, gender: str) -> str:
    """按员工序号稳定生成一个可读的模拟中文姓名。"""
    surname = SURNAME_POOL[(index * 7 + 3) % len(SURNAME_POOL)]
    pool = GIVEN_FEMALE if str(gender).lower() == "female" else GIVEN_MALE
    given = pool[(index * 11 + 5) % len(pool)]
    return surname + given


@dataclass
class Room:
    rid: str
    cn: str
    en: str
    kind: str            # dept | office | meeting | facility | public
    x: float
    y: float
    w: float
    d: float
    color: str
    door: str = "s"      # 门朝向（接走廊那侧）
    dept: str | None = None
    seats: int = 0

    @property
    def cx(self): return round(self.x + self.w / 2, 2)
    @property
    def cy(self): return round(self.y + self.d / 2, 2)


@dataclass
class Desk:
    desk_id: str
    floor: int
    room: str
    dept: str
    x: float
    y: float
    rot: float


@dataclass
class Furn:
    fid: str
    floor: int
    ftype: str
    room: str
    x: float
    y: float
    rot: float
    w: float = 0.0
    d: float = 0.0
    h: float = 0.0


# ================================================================ 平面房间
def floor_rooms(level: int = 1) -> list[Room]:
    """返回单层平面的房间清单。

    布局逻辑
    --------
    - 走廊 y = 10.5~13.5 横贯东西，核心筒嵌在走廊北侧。
    - **北区**（y 13.5~25.5）是面积最大的靠窗空间，放三个最大的部门：
      研发部（左）、市场部 + 运营部（右）。西侧从 y=18.0 起退让，
      留出 7.5×7.5m 的大会议室。
    - **南区**（y 0.5~10.5）是职能与接待区：西侧咖啡吧纵向贯通，
      其余自上而下分两排，保证每间房都贴走廊、门朝北开。
    """
    R: list[Room] = []

    def add(rid, cn, en, kind, x, y, w, d, color, door="s", dept=None, seats=0):
        R.append(Room(rid, cn, en, kind, x, y, w, d, color, door, dept, seats))

    # ---------------- 中段横带（走廊两端）：公共区 ----------------
    add("meeting_l", "大会议室", "Conference", "meeting",
        0.5, 13.5, 7.5, 7.5, ROOM_COLOR["meeting"], door="s")
    add("pantry", "茶水间", "Pantry", "facility",
        8.5, 13.5, 7.0, 7.5, ROOM_COLOR["facility"], door="s")
    add("focus", "专注区 / 电话间", "Focus Area", "facility",
        16.0, 13.5, 6.0, 7.5, ROOM_COLOR["facility"], door="s")
    add("storage", "打印 / 储物", "Print & Storage", "facility",
        22.5, 13.5, 6.0, 7.5, ROOM_COLOR["facility"], door="s")
    add("lounge", "休息区", "Lounge", "facility",
        29.0, 13.5, 10.5, 7.5, ROOM_COLOR["facility"], door="s")

    # ---------------- 北区（y 21~25.5）：两个大部门靠窗 ----------------
    add("rnd", "研发部 · 开放办公区", "R&D Open Office", "dept",
        0.5, 21.0, 19.5, 4.5, DEPT["rnd"]["color"],
        door="s", dept="rnd", seats=10)
    add("marketing", "市场部", "Marketing", "dept",
        20.5, 21.0, 19.0, 4.5, DEPT["marketing"]["color"],
        door="s", dept="marketing", seats=7)

    # ---------------- 南区上排（y 5.3~10.5）：客服 / 运营 ----------------
    add("service", "客服中心 / 前台接待", "Customer Service", "dept",
        8.5, 5.3, 12.0, 5.2, DEPT["service"]["color"],
        door="n", dept="service", seats=9)
    add("ops", "运营部", "Operations", "dept",
        20.5, 5.3, 13.0, 5.2, DEPT["ops"]["color"],
        door="n", dept="ops", seats=7)

    # ---------------- 南区下排（y 0.5~5.3）：咖啡吧 / 小会议室 / 职能 ----------------
    add("cafe", "咖啡吧", "Cafe", "facility",
        0.5, 0.5, 7.0, 4.8, ROOM_COLOR["facility"], door="n")
    add("meeting_s", "小会议室", "Meeting Room", "meeting",
        8.5, 0.5, 7.0, 4.8, ROOM_COLOR["meeting"], door="n")
    add("exec", "高管办公区", "Executive Area", "office",
        16.0, 0.5, 8.0, 4.8, DEPT["exec"]["color"],
        door="n", dept="exec", seats=2)
    add("hr", "人力与行政", "HR & Admin", "dept",
        24.5, 0.5, 8.0, 4.8, DEPT["hr"]["color"],
        door="n", dept="hr", seats=3)
    add("finance", "财务部", "Finance", "dept",
        33.0, 0.5, 6.5, 4.8, DEPT["finance"]["color"],
        door="n", dept="finance", seats=4)
    return R


def build_all_rooms() -> dict[int, list[Room]]:
    return {1: floor_rooms(1)}


# ================================================================ 工位
def fill_floor_desks(level: int, rooms: list[Room]) -> list[Desk]:
    """按房间排布工位（边距自适应，保证满足席位数）。"""
    out: list[Desk] = []
    # 同一部门可能分布在多个房间（客服中心 + 前台接待），编号必须在
    # 楼层内连续递增，不能每个房间都从 01 重新开始。
    dept_counts: dict[str, int] = {}
    for r in rooms:
        if r.kind not in ("dept", "office") or r.seats <= 0:
            continue
        need = r.seats
        best = None
        for mx, my in [(1.0, 1.1), (0.8, 0.9), (0.55, 0.65)]:
            uw, ud = r.w - 2 * mx, r.d - 2 * my
            if uw <= 0 or ud <= 0:
                continue
            cols, rows = int(uw // PITCH_X), int(ud // PITCH_Y)
            if cols * rows >= need:
                while rows > 1 and (rows - 1) * cols >= need:
                    rows -= 1
                while cols > 1 and (cols - 1) * rows >= need:
                    cols -= 1
                best = (mx, my, cols, rows)
                break
        if best is None:
            mx, my = 0.55, 0.65
            cols = max(1, int((r.w - 2 * mx) // PITCH_X))
            rows = max(1, int((r.d - 2 * my) // PITCH_Y))
            # 只有真的放不下才告警（窄房间退化成单排是正常的）
            if cols * rows < need:
                print(f"      [!] {level}F {r.cn} 空间不足：需 {need}，最多 {cols*rows}")
            best = (mx, my, cols, rows)
        mx, my, cols, rows = best
        uw, ud = r.w - 2 * mx, r.d - 2 * my
        gw, gd = cols * PITCH_X, rows * PITCH_Y
        x0 = r.x + mx + (uw - gw) / 2 + PITCH_X / 2
        y0 = r.y + my + (ud - gd) / 2 + PITCH_Y / 2
        n = 0
        base_no = dept_counts.get(r.dept, 0)
        for ri in range(rows):
            for ci in range(cols):
                if n >= need:
                    break
                out.append(Desk(
                    desk_id=f"F{level}-{r.dept[:3].upper()}-{base_no+n+1:02d}",
                    floor=level, room=r.rid, dept=r.dept,
                    x=round(x0 + ci * PITCH_X, 3),
                    y=round(y0 + ri * PITCH_Y, 3),
                    rot=round(float(0.0 if ri % 2 == 0 else np.pi), 4)))
                n += 1
        dept_counts[r.dept] = base_no + n
    return out


# ================================================================ 家具
def build_floor_furniture(level: int, rooms: list[Room],
                          desks: list[Desk]) -> list[Furn]:
    F: list[Furn] = []
    n = 0

    def add(ft, room, x, y, rot=0.0, w=0.0, d=0.0, h=0.0):
        nonlocal n
        n += 1
        F.append(Furn(f"F{level}-F{n:04d}", level, ft, room,
                      round(x, 3), round(y, 3), round(float(rot), 4), w, d, h))

    for r in rooms:
        if r.kind == "dept":
            for dk in [d for d in desks if d.room == r.rid]:
                add("desk", r.rid, dk.x, dk.y, dk.rot, DESK_W, DESK_D, 0.74)
                add("monitor", r.rid, dk.x, dk.y - DESK_D/2 + 0.1, dk.rot, .6, .06, .38)
                add("keyboard", r.rid, dk.x, dk.y + 0.12, dk.rot, .42, .16, .02)
                off = DESK_D / 2 + 0.45
                cx = dk.x - np.sin(dk.rot) * off
                cy = dk.y + np.cos(dk.rot) * off
                add("chair", r.rid, cx, cy, dk.rot + np.pi, .55, .55, .95)
                add("cabinet", r.rid, dk.x + DESK_W/2 + 0.25, dk.y, dk.rot, .4, .5, .6)
            add("plant", r.rid, r.x + 0.8, r.y + r.d - 0.8, 0, .5, .5, 1.3)
            add("plant", r.rid, r.x + r.w - 0.8, r.y + r.d - 0.8, 0, .5, .5, 1.3)
            add("whiteboard", r.rid, r.cx, r.y + 0.35, 0, min(2.6, r.w*.3), .1, 1.2)

        elif r.kind == "office":
            add("exec_desk", r.rid, r.cx, r.y + r.d - 1.9, 0, 2.4, 1.1, .78)
            add("exec_chair", r.rid, r.cx, r.y + r.d - 3.0, np.pi, .62, .62, 1.15)
            add("guest_chair", r.rid, r.cx - 1.1, r.y + r.d - 0.8, np.pi, .55, .55, .95)
            add("guest_chair", r.rid, r.cx + 1.1, r.y + r.d - 0.8, np.pi, .55, .55, .95)
            add("sofa", r.rid, r.cx + 1.5, r.y + 1.3, 0, 1.8, .85, .8)
            add("coffee_table", r.rid, r.cx + 1.5, r.y + 2.4, 0, 1.1, .6, .4)
            add("bookshelf", r.rid, r.x + 0.45, r.cy - 1.0, 0, .4, 2.2, 2.2)
            add("plant", r.rid, r.x + r.w - 0.7, r.y + 0.7, 0, .6, .6, 1.5)
            # 高管区里给两位高管留出工位，否则他们的座位点无家具承托
            for dk in [d for d in desks if d.room == r.rid]:
                add("desk", r.rid, dk.x, dk.y, dk.rot, DESK_W, DESK_D, 0.74)
                add("monitor", r.rid, dk.x, dk.y - DESK_D/2 + 0.1, dk.rot, .6, .06, .38)
                add("chair", r.rid, dk.x, dk.y + DESK_D/2 + 0.45, dk.rot + np.pi,
                    .55, .55, .95)

        elif r.kind == "meeting":
            big = r.w >= 10
            tw = min(r.w - 3.0, 5.2 if big else 2.6)
            td = 1.8 if big else 1.2
            add("table_oval" if big else "table_rect", r.rid, r.cx, r.cy, 0,
                tw, td, .75)
            nc = int(tw // 1.0)
            for i in range(nc):
                x = r.cx - tw/2 + 0.5 + i * 1.0
                add("meeting_chair", r.rid, x, r.cy - td/2 - 0.45, 0)
                add("meeting_chair", r.rid, x, r.cy + td/2 + 0.45, np.pi)
            add("screen", r.rid, r.cx, r.y + 0.35, 0, 2.6, .12, 1.5)
            add("plant", r.rid, r.x + r.w - 0.7, r.y + r.d - 0.7, 0, .6, .6, 1.5)

        elif r.rid == "lobby":
            add("reception_desk", r.rid, r.cx - 1.5, r.cy - 1.6, 0, 3.6, .9, 1.1)
            add("sofa", r.rid, r.x + 8.0, r.cy + 1.2, 0, 1.8, .85, .8)
            add("coffee_table", r.rid, r.x + 8.0, r.cy + 2.2, 0, 1.1, .6, .4)
            add("plant", r.rid, r.x + 0.8, r.y + 0.9, 0, .7, .7, 1.7)
            add("plant", r.rid, r.x + r.w - 0.8, r.y + 0.9, 0, .7, .7, 1.7)
            add("logo_wall", r.rid, r.x + 2.0, r.y + 0.15, 0, 3.0, .1, 1.4)

        elif r.rid == "cafe":
            add("counter", r.rid, r.cx, r.y + 0.75, 0, min(5.0, r.w-1.5), .65, .9)
            add("coffee_machine", r.rid, r.cx + 1.2, r.y + 0.75, 0, .5, .5, .55)
            add("fridge", r.rid, r.x + r.w - 1.0, r.y + 0.9, 0, .75, .7, 1.8)
            add("table_round", r.rid, r.cx - 1.5, r.cy, 0, 1.2, 1.2, .75)
            add("table_round", r.rid, r.cx + 1.0, r.cy + 2.6, 0, 1.2, 1.2, .75)
            add("sofa", r.rid, r.cx + 1.2, r.y + r.d - 2.2, np.pi, 1.8, .85, .8)
            add("coffee_table", r.rid, r.cx + 1.2, r.y + r.d - 3.3, 0, 1.1, .6, .4)
            for i, dy in enumerate([-2.4, -1.2, 0.4, 1.6]):
                add("guest_chair", r.rid, r.cx + 1.0, r.cy + dy, np.pi, .5, .5, .9)
            add("plant", r.rid, r.x + 0.8, r.y + 0.8, 0, .6, .6, 1.5)

        elif r.rid == "pantry" or r.rid.startswith("pantry"):
            add("counter", r.rid, r.cx, r.y + 0.75, 0, min(5.0, r.w-1.2), .65, .9)
            add("sink", r.rid, r.cx - 1.2, r.y + 0.75, 0, .8, .5, .06)
            add("fridge", r.rid, r.x + r.w - 0.9, r.y + 0.9, 0, .75, .7, 1.8)
            add("water_cooler", r.rid, r.x + 0.7, r.y + 0.75, 0, .4, .4, 1.1)
            add("bar_stool", r.rid, r.cx - 0.8, r.y + 1.9, 0, .4, .4, 1.05)
            add("bar_stool", r.rid, r.cx + 0.4, r.y + 1.9, 0, .4, .4, 1.05)
            add("table_round", r.rid, r.cx, r.y + 4.6, 0, 1.6, 1.6, .75)
            for i, dx in enumerate([-1.2, 0.0, 1.2]):
                add("guest_chair", r.rid, r.cx + dx, r.y + 6.0, np.pi, .5, .5, .9)

        elif r.rid.startswith("server"):
            for i in range(4):
                add("shelf", r.rid, r.x + 1.0 + i * 1.9, r.cy, 0, .6, 4.5, 2.0)

        elif r.rid.startswith("focus"):
            for i in range(3):
                add("phone_booth", r.rid, r.x + 1.0 + i * 1.8, r.cy + 1.6, 0, 1.1, 1.1, 2.2)
            add("table_round", r.rid, r.cx, r.cy - 2.2, 0, 1.4, 1.4, .75)

        elif r.rid.startswith("storage") or r.rid.startswith("print"):
            add("shelf", r.rid, r.x + 0.6, r.cy, 0, .5, min(r.d-1.5, 6.0), 2.0)
            add("printer", r.rid, r.x + r.w - 1.0, r.y + 1.2, 0, .7, .7, 1.1)
            add("shredder", r.rid, r.x + r.w - 1.0, r.y + 2.6, 0, .45, .45, .8)
            add("locker_bank", r.rid, r.x + r.w - 0.4, r.cy + 1.2, 0, .5, 3.2, 1.9)

        elif r.rid.startswith("security"):
            add("counter", r.rid, r.cx, r.y + r.d - 0.8, 0, min(3.5, r.w-1.0), .6, 1.05)
            add("tv", r.rid, r.cx, r.y + 0.25, 0, 1.6, .1, 1.0)

        elif r.rid == "boardroom":
            add("table_oval", r.rid, r.cx, r.cy, 0, min(r.w-2.5, 5.0), 1.8, .75)
            for i in range(4):
                x = r.cx - 1.9 + i * 1.25
                add("meeting_chair", r.rid, x, r.cy - 1.35, 0)
                add("meeting_chair", r.rid, x, r.cy + 1.35, np.pi)
            add("screen", r.rid, r.cx, r.y + 0.35, 0, 3.0, .12, 1.7)

        else:
            # 兜底：给个储物架，避免空房间
            add("shelf", r.rid, r.x + 0.6, r.cy, 0, .5, min(r.d-1.0, 4.0), 2.0)
    return F


# ================================================================ 路径
def floor_waypoints(level: int, rooms: list[Room]) -> list[dict]:
    """单层：横向主走廊骨架 + 各房间门节点（全部轴对齐）。

    单层没有电梯，走廊骨架就是唯一的连通通路，所有房间门节点垂直接到骨架上。
    """
    wp: list[dict] = []

    def add(wid, x, y):
        if any(abs(w["x"]-x) < 1e-6 and abs(w["y"]-y) < 1e-6 for w in wp):
            return
        wp.append({"id": wid, "x": round(x, 2), "y": round(y, 2), "links": []})

    cy = (CORR_Y0 + CORR_Y1) / 2
    north = [r for r in rooms if r.y > CORR_Y1]
    south = [r for r in rooms if r.y < CORR_Y0]

    xs = sorted({round(r.cx, 2) for r in north + south} | {3.0, 37.0})
    for i, x in enumerate(xs):
        add(f"c{i}", x, cy)
    for i in range(len(xs) - 1):
        a, b = f"c{i}", f"c{i+1}"
        next(w for w in wp if w["id"] == a)["links"].append(b)
        next(w for w in wp if w["id"] == b)["links"].append(a)

    for r in north:
        add(f"door_{r.rid}", r.cx, CORR_Y1 + 0.4)
        _link_spine(wp, f"door_{r.rid}", cy)
    for r in south:
        add(f"door_{r.rid}", r.cx, CORR_Y0 - 0.4)
        _link_spine(wp, f"door_{r.rid}", cy)
    return wp


def _link_spine(wp, nid, cy):
    """把门节点垂直接到最近的走廊骨架节点上（只连垂直，避免斜线）。"""
    node = next((w for w in wp if w["id"] == nid), None)
    if not node:
        return
    cands = [w for w in wp if abs(w["y"] - cy) < 1e-6 and w["id"] != nid]
    if not cands:
        return
    tgt = min(cands, key=lambda w: abs(w["x"] - node["x"]))
    if tgt["id"] not in node["links"]:
        node["links"].append(tgt["id"])
        tgt["links"].append(nid)


# ================================================================ 员工
def assign_staff(floors: dict[int, list[Room]], all_desks: list[Desk]) -> pd.DataFrame:
    """把员工分配到工位。优先复用已有的 1200 人行为数据做属性来源。"""
    src = []
    p = DATA / "link_employees.csv"
    if p.exists():
        src = pd.read_csv(p, encoding="utf-8-sig").to_dict("records")

    risk = {}
    rp = DATA / "link_employee_risk.csv"
    if rp.exists():
        rr = pd.read_csv(rp, encoding="utf-8-sig")
        risk = dict(zip(rr["employee_id"], zip(rr["risk_score"], rr["risk_level"])))

    rows, ei = [], 0
    for lv in sorted(floors):
        for dk in [d for d in all_desks if d.floor == lv]:
            rec = src[ei % len(src)] if src else {}
            ei += 1
            eid = f"E{ei:05d}"
            gender = str(rec.get("gender", "Male"))
            rs, rl = risk.get(rec.get("employee_id"), (None, None))
            rows.append({
                "employee_id": eid,
                "name": simulated_name(ei, gender),
                "gender": gender,
                "floor": lv,
                "floor_name": f"{lv}F",
                "department_key": dk.dept,
                "department_cn": DEPT[dk.dept]["cn"],
                # 职位从本部门岗位池里取，保证与部门一致
                "job_role": str(RNG.choice(ROLES_BY_DEPT[dk.dept])),
                "desk_id": dk.desk_id,
                "room": dk.room,
                "desk_x": dk.x, "desk_y": dk.y, "desk_rot": dk.rot,
                "risk_score": rs, "risk_level": rl,
            })
    return pd.DataFrame(rows)


# ================================================================ 绘图
def plot_floors(floors, desks_by_floor, furn_by_floor):
    fig, ax = plt.subplots(figsize=(17, 12))
    lv = sorted(floors)[0]
    ax.add_patch(Rectangle((0, 0), FW, FD, facecolor="#F7F9FC",
                           edgecolor="#1E2C3A", lw=2.6, zorder=1))
    # 走廊底色，让「主走廊」在平面上可读
    ax.add_patch(Rectangle((0, CORR_Y0), FW, CORR_Y1 - CORR_Y0,
                           facecolor="#E3EAF4", edgecolor="none", zorder=1.5))
    for r in floors[lv]:
        fc = r.color
        ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, facecolor=fc, alpha=.18,
                               edgecolor=fc, lw=1.5, zorder=2))
        ax.text(r.cx, r.cy, r.cn, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="#152535", zorder=9,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          alpha=.86, edgecolor=fc, lw=1))
    for d in desks_by_floor[lv]:
        ax.add_patch(Rectangle((d.x - DESK_W/2, d.y - DESK_D/2), DESK_W, DESK_D,
                               facecolor=DEPT[d.dept]["color"], alpha=.85,
                               edgecolor="white", lw=.3, zorder=4))
    for f in furn_by_floor[lv]:
        if f.ftype in ("desk", "chair", "monitor", "keyboard", "cabinet",
                       "plant", "meeting_chair"):
            continue
        ax.add_patch(Rectangle((f.x - 0.35, f.y - 0.35), .7, .7,
                               facecolor="#4A5568", alpha=.6, zorder=3))
    ax.set_xlim(-2, FW + 2)
    ax.set_ylim(-2, FD + 2)
    ax.set_aspect("equal")
    desk_count = sum(len(v) for v in desks_by_floor.values())
    ax.set_title(f"公司总部办公层 · {desk_count} 个工位 · "
                 f"35 × 26 m 单层平面" if (FW, FD) == (35, 26) else
                 f"公司总部办公层 · {desk_count} 个工位 · "
                 f"{FW:.0f} × {FD:.0f} m 单层平面",
                 fontsize=14, fontweight="bold")
    hs = [Patch(facecolor=d["color"], label=d["cn"]) for d in DEPT.values()]
    hs += [Patch(facecolor=ROOM_COLOR["meeting"], label="会议室"),
           Patch(facecolor=ROOM_COLOR["facility"], label="公共设施")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(.5, -.06),
              ncol=6, fontsize=10, framealpha=.94)
    plt.tight_layout()
    plt.savefig(FIGURES / "18_大楼各层平面图.png", dpi=140, bbox_inches="tight")
    plt.close()


def floor_title(lv=1):
    return "开放办公层 · 研发 / 市场 / 运营 / 客服 / 职能"


def plot_3d(floors, desks_by_floor, furn_by_floor):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    Z = [0]

    def cub(x0, y0, z0, sx, sy, sz):
        p = [(x0,y0,z0),(x0+sx,y0,z0),(x0+sx,y0+sy,z0),(x0,y0+sy,z0),
             (x0,y0,z0+sz),(x0+sx,y0,z0+sz),(x0+sx,y0+sy,z0+sz),(x0,y0+sy,z0+sz)]
        return [[p[0],p[1],p[2],p[3]],[p[4],p[5],p[6],p[7]],
                [p[0],p[1],p[5],p[4]],[p[2],p[3],p[7],p[6]],
                [p[1],p[2],p[6],p[5]],[p[0],p[3],p[7],p[4]]]

    def draw(f, c, a=1.0, ec="none", lw=0.0):
        Z[0] += 1
        ax.add_collection3d(Poly3DCollection(f, facecolor=c, edgecolor=ec,
                                             linewidths=lw, alpha=a, zorder=Z[0]))

    lv = sorted(floors)[0]
    draw(cub(0, 0, -SLAB, FW, FD, SLAB), "#C9D2E0", 1.0, "#8A96A6", .5)
    draw(cub(0, CORR_Y0, .01, FW, CORR_Y1 - CORR_Y0, .02), "#DCE5F0", .9)
    for r in floors[lv]:
        draw(cub(r.x, r.y, .02, r.w, r.d, .03), r.color, .30)
    for d in desks_by_floor[lv]:
        draw(cub(d.x-DESK_W/2, d.y-DESK_D/2, .02, DESK_W, DESK_D, .74),
             DEPT[d.dept]["color"], .95)
    for f in furn_by_floor[lv]:
        if f.ftype in ("desk", "monitor", "keyboard", "cabinet"):
            continue
        w = f.w or .5
        dd = f.d or .5
        draw(cub(f.x-w/2, f.y-dd/2, .02, w, dd, f.h or .5), "#5A6472", .9)
    # 玻璃幕墙
    draw(cub(-.12, 0, 0, .12, FD, FLOOR_H), "#8FC4E8", .12)
    draw(cub(FW, 0, 0, .12, FD, FLOOR_H), "#8FC4E8", .12)
    draw(cub(0, -.12, 0, FW, .12, FLOOR_H), "#9FB0C4", .12)
    draw(cub(0, FD, 0, FW, .12, FLOOR_H), "#9FB0C4", .12)

    ax.set_xlim(-6, FW + 6)
    ax.set_ylim(-8, FD + 8)
    ax.set_zlim(0, FLOOR_H + 4)
    ax.set_box_aspect((FW + 12, FD + 16, FLOOR_H + 6))
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("高度 (m)")
    desk_count = sum(len(v) for v in desks_by_floor.values())
    ax.set_title(f"公司总部办公层 3D — 单层 · {desk_count} 个工位",
                 fontsize=15, fontweight="bold")
    hs = [Patch(facecolor=d["color"], label=d["cn"]) for d in DEPT.values()]
    hs.append(Patch(facecolor=ROOM_COLOR["meeting"], label="会议室"))
    hs.append(Patch(facecolor=ROOM_COLOR["facility"], label="公共设施"))
    ax.legend(handles=hs, loc="upper left", fontsize=9, framealpha=.94)
    plt.tight_layout()
    plt.savefig(FIGURES / "19_大楼3D预览.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close()


# ================================================================ 主流程
def main():
    print("公司总部办公层布局（单层 · 42 人）")
    print("=" * 62)

    floors = build_all_rooms()
    lv = 1
    rooms = floors[lv]
    print(f"[1/6] 单层平面 {len(rooms)} 个房间")

    desks = fill_floor_desks(lv, rooms)
    desks_by_floor = {lv: desks}
    print(f"\n[2/6] 工位 {len(desks)} 个")
    print(f"      部门分布  {dict(Counter(d.dept for d in desks))}")

    furn = build_floor_furniture(lv, rooms, desks)
    furn_by_floor = {lv: furn}
    print(f"\n[3/6] 家具 {len(furn)} 件")

    wps = {lv: floor_waypoints(lv, rooms)}
    print(f"[4/6] 行走路径 {len(wps[lv])} 个节点")

    staff = assign_staff(floors, desks)
    print(f"\n[5/6] 员工 {len(staff)} 人（全部在同一层）")
    print("      " + str(dict(staff.groupby("department_cn").size())))

    payload = {
        "building": {
            "name": "公司总部办公层", "floors": N_FLOORS,
            "footprint": {"w": FW, "d": FD},
            "floor_height": FLOOR_H, "slab_thickness": SLAB,
            "wall_thickness": WALL, "door_width": DOOR,
            "total_area": FW * FD * N_FLOORS,
            "total_staff": len(staff),
            "corridor": {"y0": CORR_Y0, "y1": CORR_Y1},
        },
        "desk_module": {"w": DESK_W, "d": DESK_D, "px": PITCH_X, "py": PITCH_Y},
        "departments": [{"key": k, **v} for k, v in DEPT.items()],
        "floors": [
            {
                "level": lv,
                "name": f"{lv}F",
                "title": floor_title(lv),
                "rooms": [asdict(r) for r in rooms],
                "desks": [asdict(d) for d in desks],
                "furniture": [asdict(f) for f in furn],
                "waypoints": wps[lv],
            }
        ],
    }
    (DATA / "building_layout.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    staff.to_csv(DATA / "building_desks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(f) for f in furn]).to_csv(
        DATA / "building_furniture.csv", index=False, encoding="utf-8-sig")

    plot_floors(floors, desks_by_floor, furn_by_floor)
    plot_3d(floors, desks_by_floor, furn_by_floor)
    print(f"\n[6/6] 导出")

    print()
    print("=" * 62)
    print(f"建筑      {FW:.0f} × {FD:.0f} m × {N_FLOORS} 层 = {FW*FD*N_FLOORS:.0f} m²")
    print(f"层高      {FLOOR_H} m    单层无电梯，全平层动线")
    print(f"员工      {len(staff)} 人")
    print(f"工位      {len(desks)} 个    家具 {len(furn)} 件")
    print()
    print("输出:")
    for f in ["building_layout.json", "building_desks.csv", "building_furniture.csv"]:
        print(f"  {DATA/f}")
    print(f"  {FIGURES/'18_大楼各层平面图.png'}")
    print(f"  {FIGURES/'19_大楼3D预览.png'}")


if __name__ == "__main__":
    main()
