"""校验四层办公布局与员工档案是否使用同一组空间标识。

building_layout.json 是唯一的楼层、房间、工位坐标源；前端副本只能由它同步生成。
这个检查放在前端数据复制之后，避免改布局后出现员工、工位和房间错位。
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate() -> None:
    layout = load(ROOT / "data/building_layout.json")
    profiles = load(ROOT / "data/employee_profiles.json")
    output_layout = load(ROOT / "output/dashboard/building.json")
    output_profiles = load(ROOT / "output/dashboard/profiles.json")
    errors: list[str] = []

    floors = layout.get("floors", [])
    levels = [f.get("level") for f in floors]
    if len(levels) != len(set(levels)):
        errors.append("楼层 level 重复")

    desk_map: dict[str, dict] = {}
    room_map: dict[tuple[int, str], dict] = {}
    for floor in floors:
        level = floor.get("level")
        rooms = floor.get("rooms", [])
        room_ids = [r.get("rid") for r in rooms]
        if len(room_ids) != len(set(room_ids)):
            errors.append(f"{level}F 房间 rid 重复")
        for room in rooms:
            room_map[(level, room["rid"])] = room
        waypoint_ids = {w.get("id") for w in floor.get("waypoints", [])}
        for waypoint in floor.get("waypoints", []):
            missing = set(waypoint.get("links", [])) - waypoint_ids
            if missing:
                errors.append(f"{level}F 路径节点 {waypoint.get('id')} 缺少链接目标 {sorted(missing)}")
        for desk in floor.get("desks", []):
            desk_id = desk.get("desk_id")
            if desk_id in desk_map:
                errors.append(f"desk_id 重复：{desk_id}")
            desk_map[desk_id] = {**desk, "floor": level}
            if (level, desk.get("room")) not in room_map:
                errors.append(f"{desk_id} 指向不存在的房间 {desk.get('room')}")

    profile_ids = [p.get("employee_id") for p in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("employee_id 重复")
    if len(profiles) != len(desk_map):
        errors.append(f"员工数 {len(profiles)} 与工位数 {len(desk_map)} 不一致")

    for profile in profiles:
        eid = profile.get("employee_id")
        desk_id = profile.get("desk_id")
        desk = desk_map.get(desk_id)
        if desk is None:
            errors.append(f"{eid} 找不到工位 {desk_id}")
            continue
        if profile.get("floor") != desk["floor"]:
            errors.append(f"{eid} 楼层 {profile.get('floor')} 与工位楼层 {desk['floor']} 不一致")
        if profile.get("desk_room") != desk.get("room"):
            errors.append(f"{eid} 房间 {profile.get('desk_room')} 与工位房间 {desk.get('room')} 不一致")
        pos = profile.get("desk", {})
        for axis in ("x", "y"):
            if abs(float(pos.get(axis, 0)) - float(desk.get(axis, 0))) > 1e-6:
                errors.append(f"{eid} {axis} 坐标与 {desk_id} 不一致")

    if layout != output_layout:
        errors.append("output/dashboard/building.json 不是 data/building_layout.json 的同步副本")
    if profiles != output_profiles:
        errors.append("output/dashboard/profiles.json 不是 data/employee_profiles.json 的同步副本")

    if errors:
        raise SystemExit("布局一致性检查失败：\n- " + "\n- ".join(errors))
    print(f"[layout-check] OK: {len(floors)} floors / {len(desk_map)} desks / {len(profiles)} profiles")


if __name__ == "__main__":
    validate()
