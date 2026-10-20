# 资产清单 (BOM) — 办公楼总部室内场景

> 场景 `office-hq` · 单层 64×44m · 4 层
> 命名遵循 `company_3d/README.md` 规范：`<前缀>_<主体>_<变体>_v<主>.<次><状态>`
> 数量为**单层**用量；4 层复用，实例化而非复制源文件。

---

## 1. 建筑结构（P1 阶段）

| 资产 ID | 名称 | 数量/层 | 规格 | LOD | 优先级 |
|---|---|---|---|---|---|
| `mdl_floor-slab` | 楼板 | 1 | 64×44×0.3 m | 高 | P1 |
| `mdl_wall-partition` | 内隔墙模块 | ~40 段 | 厚 0.2 m, 高 3.2 m | 中 | P1 |
| `mdl_wall-curtain-glass` | 玻璃幕墙单元 | ~20 | 8×3.2 m 分格 | 高 | P1 |
| `mdl_mullion-alu` | 幕墙竖挺 | ~26 | 0.16×0.16×3.2 m | 中 | P1 |
| `mdl_column-concrete` | 结构柱 | 40 | 0.6×0.6×3.2 m | 低 | P1 |
| `mdl_ceiling-grid` | 吊顶格栅 | 1 套 | 600×600 模块 | 中 | P1 |
| `mdl_door-single` | 单开门 | ~14 | 0.9×2.1 m | 中 | P1 |
| `mdl_door-double-glass` | 玻璃双开门 | 2 | 1.8×2.4 m | 高 | P1 |
| `mdl_entry-turnstile` | 入口闸机 | 3 | 标准三辊闸 | 中 | P2 |

## 2. 工位系统（P2 阶段，核心资产）

| 资产 ID | 名称 | 数量/层 | 规格 | LOD | 优先级 |
|---|---|---|---|---|---|
| `mdl_desk-bench` | 工位桌板 | 351 | 1.4×0.7×0.04 m | 中 | P2 |
| `mdl_desk-leg` | 桌腿 | 1404 | 0.05×0.05×0.74 m | 低 | P2 |
| `mdl_monitor-24` | 显示器 24" | 351 | 0.62×0.38 m | 中 | P2 |
| `mdl_monitor-arm` | 显示器支架 | 351 | 0.18×0.14 m | 低 | P2 |
| `mdl_chair-task` | 办公椅 | 351 | 座 0.5×0.5, 背高 0.5 | 中 | P2 |
| `mdl_desk-drawer` | 活动柜 | 351 | 0.4×0.5×0.6 m | 低 | P3 |
| `mdl_desk-divider` | 工位隔板 | ~180 | 1.4×0.4 m | 低 | P3 |
| `mdl_cable-tray` | 线槽 | 1 套 | 桌面下走线 | 低 | P3 |

> 提示：351 工位 × 6 件 = 2106 个对象/层。务必用**实例化 (Linked Duplicate)**
> 而非复制，否则文件体积与渲染负担会失控。

## 3. 会议室（P2 阶段）

| 资产 ID | 名称 | 数量 | 规格 | LOD | 优先级 |
|---|---|---|---|---|---|
| `mdl_table-meeting-l` | 大会议桌 | 1 | 4.0×1.4 m, 16 座 | 高 | P2 |
| `mdl_table-meeting-m` | 中会议桌 | 2 | 2.6×1.2 m, 10 座 | 中 | P2 |
| `mdl_table-meeting-s` | 小会议桌 | 2 | 1.8×0.9 m, 6 座 | 中 | P2 |
| `mdl_chair-meeting` | 会议椅 | 48 | 标准 | 中 | P2 |
| `mdl_screen-display-65` | 会议显示屏 65" | 5 | 壁挂 | 中 | P2 |
| `mdl_whiteboard` | 白板 | 3 | 1.8×1.2 m | 低 | P2 |
| `mdl_conf-camera` | 视频会议摄像头 | 5 | 顶部安装 | 低 | P3 |

## 4. 前台 / 休息区 / 配套（P2–P3 阶段）

| 资产 ID | 名称 | 数量 | 规格 | LOD | 优先级 |
|---|---|---|---|---|---|
| `mdl_reception-desk` | 前台接待台 | 1 | 3.6×0.9×1.1 m | 高 | P2 |
| `mdl_sofa-lobby` | 大堂沙发 | 2 | 3 人位 | 中 | P2 |
| `mdl_coffee-table` | 茶几 | 2 | 1.2×0.6 m | 中 | P2 |
| `mdl_kitchen-counter` | 茶水间操作台 | 1 套 | 4.0×0.6×0.9 m | 中 | P2 |
| `mdl_sink-kitchen` | 水槽 | 2 | 标准双槽 | 中 | P3 |
| `mdl_fridge` | 冰箱 | 2 | 双门 | 中 | P2 |
| `mdl_coffee-machine` | 咖啡机 | 2 | 商用半自动 | 中 | P3 |
| `mdl_bar-stool` | 吧台椅 | 8 | 高脚 | 中 | P3 |
| `mdl_phone-booth` | 电话间隔音舱 | 4 | 1.0×1.0×2.2 m | 高 | P2 |
| `mdl_printer-floor` | 楼层打印机 | 2 | 落地式 | 中 | P2 |
| `mdl_shredder` | 碎纸机 | 1 | | 低 | P3 |
| `mdl_locker-bank` | 储物柜组 | 1 套 | 12 门 | 低 | P3 |
| `mdl_toilet-fixture` | 卫生间洁具 | 1 套 | 马桶/洗手台/隔断 | 中 | P2 |
| `mdl_plant-large` | 大型绿植 | ~12 | 1.5–2.2 m | 中 | P3 |
| `mdl_plant-desk` | 桌面小绿植 | ~60 | 0.2 m | 低 | P3 |
| `mdl_bin-office` | 垃圾桶 | ~30 | 标准 | 低 | P3 |
| `mdl_signage-wayfind` | 导视牌 | ~10 | 悬挂/立式 | 低 | P3 |

## 5. 人物角色（P3 阶段）

| 资产 ID | 名称 | 数量 | 规格 | LOD | 优先级 |
|---|---|---|---|---|---|
| `mdl_char-office-worker-m` | 男性办公角色 | 1 套 | 骨骼绑定 + 3 套服装 | 中 | P3 |
| `mdl_char-office-worker-f` | 女性办公角色 | 1 套 | 骨骼绑定 + 3 套服装 | 中 | P3 |
| `lib_anim_desk-sit` | 工位就坐循环 | 1 | 24 fps 循环 | — | P3 |
| `lib_anim_walk-office` | 室内行走 | 1 | 带根运动 | — | P3 |
| `lib_anim_meeting-talk` | 会议交谈 | 1 | | — | P3 |
| `lib_anim_coffee-stand` | 茶水间站立 | 1 | | — | P3 |

> 1164 人需大量人物。建议：**2 个基底角色 + 材质/配饰变体**，
> 配合实例化与 GPU 蒙皮，避免逐人建模。

## 6. 灯光（P4 阶段）

| 资产 ID | 名称 | 数量 | 规格 | 优先级 |
|---|---|---|---|---|
| `prs_light-daylight` | 日光系统 | 1 套 | 含太阳角度预设 | P4 |
| `mdl_luminaire-panel-600` | 面板灯 600×600 | ~340 | 嵌入式 LED | P4 |
| `mdl_luminaire-linear` | 线性灯带 | ~60 m | 走廊/会议区 | P4 |
| `mdl_luminaire-pendant` | 吊灯 | ~14 | 茶水间/前台 | P4 |
| `mdl_emergency-light` | 应急照明 | ~20 | | P4 |

## 7. 材质库（P1–P3 阶段）

| 资产 ID | 名称 | 应用于 | 优先级 |
|---|---|---|---|
| `mat_floor-carpet-gray` | 办公区地毯 | 部门工位区 | P1 |
| `mat_floor-vinyl-light` | 走廊 PVC | 主通道 | P1 |
| `mat_floor-wood-oak` | 木地板 | 会议室/前台 | P1 |
| `mat_wall-paint-white` | 白色乳胶漆 | 内隔墙 | P1 |
| `mat_wall-glass-clear` | 玻璃 | 幕墙/隔断 | P1 |
| `mat_ceiling-mineral` | 矿棉吊顶 | 吊顶 | P1 |
| `mat_metal-brushed-alu` | 拉丝铝 | 竖挺/五金 | P1 |
| `mat_fabric-partition` | 隔板布面 | 工位隔板 | P2 |
| `mat_stone-counter` | 人造石 | 茶水间台面 | P2 |
| `mat_ceramic-tile` | 瓷砖 | 卫生间 | P2 |

---

## 8. 资产总量估算

| 类别 | 资产种类 | 单层实例数 | 4 层实例数 |
|---|---|---|---|
| 建筑结构 | 9 | ~110 | ~440 |
| 工位系统 | 8 | ~2100 | ~8400 |
| 会议室 | 7 | ~70 | ~280 |
| 配套家具 | 17 | ~140 | ~560 |
| 人物角色 | 6 | ~291 | ~1164 |
| 灯光 | 5 | ~440 | ~1760 |
| 材质 | 10 | — | — |

> **实现方式**：建筑/家具用实例化；人物用「基底 + 变体」；
> 总计约 **12500 个实例**，源资产仅约 **62 个**。这是可控的规模——
> 真正的难点是家具与角色的**精细度**，不是数量。
