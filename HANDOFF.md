# 项目进度交接文档

> 写于 2026-10-16。用于换模型/换人继续开发。
> 项目路径：`/Users/pengjianghua/Desktop/线性回归预测`

---

## 一、项目目标

做一个**公司总部大楼的三维数字孪生**：4 层办公楼、员工有完整档案数据，
在 3D 场景里点击任意员工可以查看他的档案/行为/测评/离职风险；
人物要像游戏街道一样**自然地走动、开会、做自己的事**，整栋楼"活起来"。

最终用途：**嵌入员工离职风险预测系统**。

---

## 二、当前进度

| 模块 | 状态 |
|---|---|
| 数据分析管线（step1~step8） | ✅ 全部跑通 |
| 4 层楼布局数据（每层房间/工位/家具） | ✅ |
| 42 名员工完整档案（IBM 格式 + 背调 + 行为 + 测评 + 谈话 + 风险） | ✅ |
| Web 版 3D 大楼（`building.html`） | ✅ 可运行，60fps |
| 行为模拟引擎（`sim.js`，部门自主议程 + 寻路） | ✅ |
| 门/电梯动画 | ✅ |
| 家具细部建模 | ✅ |
| 点击人物查看档案 | ✅ |
| **Blender 4 层建筑** | 🟡 **3F 试点已完成**（静态环境 GLB 已导出并接入网页，其他楼层仍沿用程序化环境） |
| **人物形象提升** | ✅ **已完成初版（多部件人物 + 眼睛/领口/手部）** |
| **去掉倍速档，让走路自然** | ✅ **已完成（固定自然速度）** |
| **切楼层后改为"楼内顶部上帝视角"** | ✅ **已完成** |
| **3D 场景里写部门名称** | ✅ **已完成** |
| **独立员工档案库（搜索 / 部门名单）** | ✅ **已完成**（默认先筛选，再显示员工；三栏对齐） |
| **布局 / 档案坐标一致性检查** | ✅ **已完成**（`src/validate_building_consistency.py`） |
| **Three.js 颜色空间 / 色调映射** | ✅ **已完成初版**（保留实时阴影关闭，优先维持帧率） |
| **Blender 3F 静态环境试点** | ✅ **已完成**（`floor_3.glb`，人物仍由 `sim.js` 驱动） |

---

## 三、文件结构与关键文件

```
线性回归预测/
├── run.sh                       # 一键跑全流程（8 步）
├── README.md                    # 项目总说明
├── HANDOFF.md                   # 本文档
├── src/
│   ├── step1_prepare_data.py        # IBM 数据 + 模拟数据
│   ├── step2_train_model.py         # 特征工程/训练/评估
│   ├── step3_data_profile.py        # 数据体检
│   ├── step4_linear_regression.py   # 线性回归
│   ├── step5_build_behavior_link.py # 行为链路 5 张表
│   ├── step6_analyze_link.py        # 链路分析 + 风险评分
│   ├── step7_office_layout.py       # ★ 4 层大楼布局（房间/工位/家具/路径）
│   └── step8_employee_profiles.py   # ★ 42 人完整档案
├── data/
│   ├── building_layout.json      # ★ 大楼规格（3D 场景读这个）
│   ├── building_desks.csv        # 42 人 → 楼层 + 工位
│   ├── building_furniture.csv    # 家具清单
│   ├── employee_profiles.json    # ★ 42 人档案（点击查看的数据源）
│   └── link_*.csv                # 行为/质量/谈话/测评原始表
├── output/dashboard/
│   ├── building.html             # ★ Web 版数字孪生（3D + UI）
│   ├── sim.js                    # ★ 行为模拟引擎
│   ├── building.json             # = data/building_layout.json 的副本
│   ├── profiles.json             # = data/employee_profiles.json 的副本
│   ├── vendor/                   # three.js r160 + 本地 GLTFLoader（可离线）
│   └── assets/office/floor_*.glb # Blender 1F~4F 静态环境资产
├── company_3d/                   # Blender 侧
│   ├── build_scene.sh            # 旧版单层人物场景（兼容保留）
│   ├── build_floor_assets.sh     # ★ 从 building_layout 导出 1F~4F 环境 GLB
│   ├── 03_scripts/
│   │   ├── scr_build-company-scene_v02.00.py   # ★ 当前用的（按楼层导出）
│   │   └── scr_build-office-scene_v01.00.py    # 旧版（已废弃）
│   ├── 02_scenes/wip/            # .blend / .glb 产出
│   ├── 05_renders/preview/       # 渲染图
│   └── 06_docs/                  # 场景规格书 / 资产清单
└── output/figures/               # 19 张分析图 + 平面图 + 3D 预览
```

**数据流向**：
```
step7 (布局) ─┬─→ data/building_layout.json ──→ cp → output/dashboard/building.json
              └─→ data/building_desks.csv ──┐
                                            ↓
step8 (档案) ←───────────────────────────────┘
              └─→ data/employee_profiles.json ──→ cp → output/dashboard/profiles.json
                                                          ↓
                                            building.html + sim.js 读取渲染
```

---

## 四、怎么运行

```bash
cd /Users/pengjianghua/Desktop/线性回归预测

# 1) 全流程（数据 + 模型 + 布局 + 档案）
./run.sh                      # 会顺便起 http server 并占用终端

# 2) 只改 3D 场景时（推荐日常用这个）
python3 src/step7_office_layout.py      # 改布局/人数/家具后重跑
python3 src/step8_employee_profiles.py  # 档案跟着重跑
cp data/building_layout.json output/dashboard/building.json
cp data/employee_profiles.json output/dashboard/profiles.json
cd output/dashboard && python3 -m http.server 8777
# 打开 http://127.0.0.1:8777/building.html

# 3) Blender 侧（从唯一四层布局导出 1F~4F 静态 GLB）
./company_3d/build_floor_assets.sh
```

**改完一定 `Cmd+Shift+R` 强制刷新**，否则看到的是缓存旧版。

---

## 五、当前数据事实（改之前先看这里）

- 建筑：**34 × 24 m，4 层，层高 3.6 m**，总 3264 m²
- 核心筒：**x=14.5, y=13.0, w=5.0, d=3.5**（2 电梯 + 1 楼梯，贯通 4 层）
- 走廊：横向主走廊 **y = 10.0 ~ 13.0**
- 人数：**42 人** = 1F 10 / 2F 10 / 3F 14 / 4F 8
- 部门分布：
  - 1F 大堂层：客服中心 6 + 前台接待 4
  - 2F：研发部 10
  - 3F：市场部 7 + 运营部 7
  - 4F：财务部 4 + 人力与行政 2 + 高管办公区 2

**房间坐标定义在 `src/step7_office_layout.py` 的 `floor_rooms(level)` 函数里**，
格式 `Room(rid, cn, en, kind, x, y, w, d, color, door, dept, seats)`，
`door` 是门朝向（'n'/'s'/'e'/'w'，指朝走廊那一侧）。

---

## 六、Web 版核心实现（`building.html` + `sim.js`）

### 6.1 渲染架构（性能关键）

**全部用 InstancedMesh 实例化**，这是 60fps 的前提：

- **静态建筑 + 家具**：走 `push(mat, x,y,z, sx,sy,sz, ry, geo)` 累积到
  `batches`，最后 `finalizeBatches(group)` 按「材质+几何」合并成少量
  实例网格 → **整个建筑 + 749 件家具只有约 200 次 draw call**
- **人物**：9 个 InstancedMesh 承载全部 42 人
  （torso / head / hair / armU / armL / thigh / shin / foot / shadow）
- **门**：1 个 InstancedMesh（25 扇）
- **电梯**：2 个独立 Group（各含轿厢 + 双开门）

**几何只有 3 种**：`UNIT`(盒) / `CYLG`(柱) / `SPHG`(球)，通过矩阵缩放复用。

> ⚠️ **如果要继续加家具细节，务必继续用 `push()` 批量接口**，
> 不要写 `new THREE.Mesh` 逐个加 —— 那会把 draw call 从 200 拉到几千，直接卡死。

### 6.2 人物配色

- `mShirt` 材质用 `instanceColor` 上色（每实例一个颜色）
- 颜色计算在 `computeColors()`，写入 `PC`（Float32Array）
- 更新在 `putInst(..., isShirt=true)` 里逐实例写入
- **坑**：`DEPT_HEX` 必须存**数字**。`building.json` 里颜色是 CSS 字符串
  `"#4C7DFF"`，而 `Color.setHex()` 要数字，传字符串会得到 NaN → 位移后变 0
  → **所有人渲染成纯黑**。已在 fetch 处用
  `parseInt(String(d.color).replace('#',''), 16)` 转换。

### 6.3 行为模拟（`sim.js`）

**核心类**：`Sim`，输入 `building.json` + `profiles.json`。

- `FloorGraph`：每层走廊节点图，`path(from,to)` 是 BFS 最短路
- `buildDailyPlan(emp)`：给每人生成一天的日程块（确定性，用员工编号 + 部门 + 岗位做种）
  - 上班/迟到、下班/早退（由合成行为节奏决定，不读取离职风险）
  - 部门议程 `DEPT_AGENDA`（研发 10:00 站会、市场 11:00 例会、财务 16:00 结算…）
  - 随机离座：茶水间/打印/休息区；固定会议和午休优先，随机活动只填空档
  - 去同事工位讨论 `discuss`
  - **巡场走动 `stroll`**：去本层另一个房间转一圈（走廊里就有人的关键）
- `blockAtWithLead(a,t)`：**提前动身**（`a.lead * LEAD_MIN` 分钟），
  且**按人错峰**（`a.lead` = 0.5~1.9），否则全部门同时出发同时到达
- `seek(t)`：用固定 `0.05` 分钟 tick 推进，保证从不同时间点回放到同一时刻得到相同快照。
- `_step(dtMin)`：逐人推进
  - 平滑转身（`a.turn` 以 `dt*7` 速率插值，不瞬间扭头）
  - 加减速（`a.v` 向 `wantV` 过渡，离终点 1.6m 内线性减速）
  - **步态相位 `a.phase` 由实际位移累积**（步幅 0.78m），脚不打滑
  - 路径点跳过（`< 0.08m` 视为已到达）
- 会议室、茶水间、咖啡吧等活动按房间生成确定性分散站位，不再让所有人使用房间中心点。
- 跨楼层路径分为 `walk_to_elevator → wait_elevator → ride → walk_from_elevator → arrived`，目标层坐标不会混入原楼层路径。
- `company_3d/03_scripts/scr_build-company-scene_v02.00.py` 支持 `--layout data/building_layout.json --floor 3 --characters 0`，Blender 导出时从四层权威布局适配单层静态环境；网页角色继续由 `sim.js` 驱动，不与 GLB 人物叠加。
- 当前已用 Blender 5.2.2 导出 `company_3d/02_scenes/wip/floor_3_environment.blend/.glb`，并复制为网页资源 `output/dashboard/assets/office/floor_3.glb`；`building.html` 通过本地 `GLTFLoader` 载入，成功后隐藏 3F 程序化静态节点，保留部门标签、门、电梯与动态人物。
- Blender 的平面深度轴在 glTF 导入后方向相反，网页加载适配层对资产 Z 轴做一次翻转；员工/工位业务坐标仍只来自 `building_layout.json`，没有硬编码视觉偏移。
- `fillSnapshot()`：写入**预分配缓冲**（不产生新对象，避免 GC 抖动）

### 6.4 已修复的关键 bug（**改代码时别改回去**）

1. **走路死锁**：人已在工位上时路径是 `[工位→走廊→工位]`，
   绕回来又判不到"已到达" → `v=0 → 位移=0 → pathIdx 不前进 → 永远"行走中"却不动`。
   修法：**同房间直达不绕走廊**（`_repath` 里判断 `here === dest.room`），
   加跳过已到达路径点。
2. **跨层永不完成**：判断写成了 `pathIdx >= path.length`（永远不成立，
   应该是 `length - 1`），跨层的人朝另一个楼层的点一直走。
3. **材质引用失效**（Blender 侧）：材质在 `clear_scene()` 前创建，
   `read_factory_settings()` 清空 `bpy.data` 后引用失效。必须清场后再建。
4. **建筑分组与爆炸视图错位**：楼层间距只推了室内没推楼体外壳，
   顶层家具戳出屋顶。修法：每层楼板+幕墙也放进该层 group。
5. **接触阴影立在地上**：`CircleGeometry` 默认朝 +Z，需 `sg.rotateX(-Math.PI/2)`。
6. **切楼层时其他层人物没隐藏**：人物是全体共用一个实例网格，
   不能用 `group.visible`。修法：按楼层把实例缩放到 `1e-4`。
7. **Blender 构建极慢**：原实现对每个物体调 `bpy.ops.mesh.primitive_cube_add`，
   351 工位 × 8 部件 = 2800 次调用跑几分钟。改用 `from_pydata` + 实例化后 ~4 秒。
8. **Blender 相机拍空**：手填欧拉角没对准场景。用
   `direction.to_track_quat('-Z','Y').to_euler()` 精确对准。

---

## 七、后续任务与实现记录（按用户优先级）

> 本轮已完成 ③ 楼内顶部视角和 ④ 部门名称，并新增独立员工档案库。下面两节保留实现思路，便于后续维护；实际剩余重点是人物形象细化、自然走动和 Blender 四层建筑。

### ① 【最高优先级】人物形象提升

用户原话："最主要的还是这个人物"。当前人物是圆柱+球体的低多边形小人，
用户觉得"丑"。参考用户给的截图，现状问题：
- 头是球体但偏大，没有五官
- 身体是直筒圆柱，没有肩宽/腰身区分
- 手臂很细且位置偏高
- 坐着时比例奇怪

**改进方向**：
- 头：球体 + 简单的五官（眼睛两点、鼻/嘴可省）+ 发型区分男女
- 躯干：上宽下窄（用 `CylinderGeometry(0.17, 0.14, ...)` 或加肩部方块）
- 加**领口/袖口**颜色区分，让衣服有层次
- 腿：大腿粗、小腿细，过渡自然
- 坐姿时手臂自然放在桌面上（现在偏悬空）
- 考虑加**性别/体型差异**（profiles.json 里有 `Gender`）

> 代码位置：`building.html` 的 `initPeople()`（几何定义）+ `updatePeople()`（姿态）

### ② 去掉倍速档，让走路自然

用户原话："这个倍速是什么意思不需要这个倍速就他们自然的走"。

**已完成**：已移除 0.4×/2×/8× 倍速按钮，场景固定按 1 秒真实时间推进 1 秒模拟时间。
拖动时间轴时仍使用模拟器内部的分档步长快速追赶，松开后回到自然速度。
初次打开场景默认定位到 10:00，员工已经陆续到岗，便于直接查看楼内行为。

### ③ 切楼层后改为"楼内顶部上帝视角"

**已完成**：`setFloor(v)` 会把相机切到对应楼层的顶部俯视位置，关闭自动旋转；单层模式隐藏该层吊顶和灯具，拖动楼层间距时会同步更新相机高度。

用户原话："我切换楼层之后我的视角就是楼内顶部的时候我能够看到这层全部员工的行为
就像上帝视角不再是楼层外面看的视角"。

实现位置：
- 相机位置 ≈ `(W/2, (v-1)*(FH+gap) + 0.96*max(W,D), D/2)`（保证整层进入视野）
- 目标 ≈ `(W/2, (v-1)*(FH+gap) + 0.15, D/2)`（楼层中心）
- 单层模式隐藏 `ceilGroups[level]`，否则俯视会被吊顶挡住
- 切回全楼时恢复默认相机和 `autoRotate`

> 注意：吊顶在 `buildInteriors()` 里用
> `push(F.ceil, 17, FH - .12, 12, 34, .1, 24)` 推到各层 group，
> 要单独控制需改成独立的 `ceilGroups[level]`。

### ④ 3D 场景里写清部门名称

**已完成**：`buildDepartmentLabels(fg, floor)` 为每个 `kind === 'dept'` 房间创建带中文名称和英文副标题的 Canvas Sprite，标签挂在对应楼层组内，会随楼层筛选同步显示。

用户原话："每个部门都要写清楚名字"。

**方案**：用 Canvas 生成文字贴图 → `THREE.Sprite` 或贴在地面上的 `PlaneGeometry`，
放在每个部门房间中央，高度略高于地面（如 y=0.1 平铺，或 y=2.5 悬浮）。

**参考实现**（v0 版本用过 Canvas 做标签，可复用思路）：
```js
const cvs = document.createElement('canvas');
const ctx = cvs.getContext('2d');
cvs.width = 512; cvs.height = 128;
ctx.font = 'bold 56px "PingFang SC",sans-serif';
ctx.fillStyle = '#123'; ctx.textAlign='center'; ctx.textBaseline='middle';
ctx.fillText('研发部', 256, 64);
const tex = new THREE.CanvasTexture(cvs);
// 平铺地面：PlaneGeometry 旋转 -PI/2，材质 MeshBasicMaterial({map:tex, transparent:true})
```
标签数据来自 `building.json` 的 `floors[].rooms[]`（有 `cn` 字段和房间矩形）。
建议**只为 `kind === 'dept'` 的房间**加标签，避免会议室/卫生间标签太密。
部门配色可用 `departments[].color`。

### ⑤ Blender 4 层建筑

**当前状态**：脚本已经读取权威 `data/building_layout.json` 并按 `--floor` 导出单层。
3F 试点已生成 `floor_3_environment.blend/.glb`，网页已通过本地 `GLTFLoader` 接入；
其他楼层还没有分别导出和接入，仍使用程序化 Web 环境。

**剩余工作**：
1. 用相同命令导出 1F / 2F / 4F 静态环境，并完成资源按楼层加载与释放
2. 用房间角点、门口、电梯口三个控制点逐层核对 GLB 与业务坐标
3. 继续补足玻璃、材质和灯光，做 Blender 预览渲染图

**Blender 侧已有的可用实现**（在 `scr_build-company-scene_v02.00.py` 里）：
- 材质库 `build_materials()`
- 家具部件库 `f_desk()/f_chair_task()/f_sofa()/f_bookshelf()` 等（这是 v02 单层版的，
  和 Web 版不是同一套，可参考但要重写）
- 角色 `build_character()`：低多边形人形 + Armature 骨骼 + 走路动画
  （**注意**：这个用 `bpy.ops` 逐部件建，慢；Web 版的做法更好）
- 坑：`iter_fcurves()` 兼容 Blender 4.x/5.x 的分层 Action API

**Blender 环境**：`/Applications/Blender.app/Contents/MacOS/Blender`，版本 5.2.2 LTS。

---

## 八、运行环境

- Python 3.14.6（Anaconda `/opt/anaconda3`）
- 依赖：pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib statsmodels scipy
- Blender 5.2.2 LTS
- 浏览器需支持 ES module + WebGL2（three.js r160 已本地化在 `output/dashboard/vendor/`）
- 中文字体：matplotlib 用 `PingFang SC`（macOS 无 `SimHei`，用了会变方框）

---

## 九、数据口径（重要，别搞混）

| 数据 | 用途 |
|---|---|
| IBM 真实数据 1470 条 | **模型训练与全部指标**只用这个 |
| 模拟数据 1530 条 | 只用于大屏点云视觉体量，**不参与指标** |
| 42 名大楼员工档案 | IBM 格式字段按真实分布采样；行为/测评为业务逻辑合成 |

`employee_profiles.json` 每人的字段：
- `profile`：IBM 35 字段 + `background`（学历/院校/专业/证书/前雇主/入职日期/合同/背调结果）
- `behavior`：90 天聚合 + 近 7 天明细 + 工作质量
- `assessment`：能力/敬业度/满意度/压力/综合 + 潜力 + 绩效
- `talks`：谈话记录（类型/主题/情绪/是否需跟进）
- `risk`：评分 + 等级 + **可解释因子**（带方向和业务说明）

---

## 十、给下一个模型的建议顺序

1. **先跑起来确认现状**：`python3 -m http.server 8777` in `output/dashboard`，
   打开 `building.html`，强刷。确认 42 人、4 层、能点人看档案。
2. **已完成 ④ 部门名称**：部门标签随楼层组渲染，单层筛选自动同步。
3. **已完成 ③ 上帝视角**：单层模式隐藏吊顶和灯具，并把相机移到该层顶部。
4. **做 ② 自然走动**（去掉倍速，调速度）
5. **做 ① 人物形象**（最重要也最费时，建议最后专门做一轮）
6. **做 ⑤ Blender 4 层**（独立任务，可并行）

**改任何东西前请先读本文档第六节的「已修复的关键 bug」**，
那些坑都是实际踩过的，容易改回去。

---

## 十一、快速定位表（精确行号，改代码时直接跳）

> 行号基于 2026-10-16 的 `output/dashboard/building.html`（约 62 KB），
> 若之后有改动，用 grep 关键字定位。

| 要改的东西 | 关键字 | 位置 |
|---|---|---|
| 自然速度循环 | `simSpeed = 1` | `building.html` 脚本顶部 |
| 切楼层逻辑（含上帝视角） | `function setFloor` | `building.html:1191` |
| 部门名称标签 | `function buildDepartmentLabels` | `building.html:775` |
| 独立员工档案库 | `employee_archive.html` | `output/dashboard/employee_archive.html` |
| 吊顶单独隐藏 | `ceilGroups` | `building.html:201,720` |
| 吊顶（要能单独隐藏） | `push(F.ceil, 17, FH - .12` | `buildInteriors()` 内 |
| 人物几何定义 | `function initPeople` | `building.html`（搜 `PGEO.torso`） |
| 人物姿态动画 | `function updatePeople` | `building.html`（搜 `hipsY`） |
| 人物配色 | `function computeColors` | `building.html` |
| 点击拾取 | `function bindPicking` | `building.html`（搜 `ray.intersectObject`） |
| 静态家具批量接口 | `function push(` | `building.html`（搜 `batches`） |
| 工位细部 | `function buildWorkstation` | `building.html` |
| 其它家具细部 | `function buildFurniture` | `building.html` |
| 楼层分组创建 | `roofGroup = new THREE.Group()` | `init3D()` 内 |
| 模拟：日程生成 | `function buildDailyPlan` | `sim.js` |
| 模拟：部门议程 | `DEPT_AGENDA` | `sim.js` 顶部 |
| 模拟：步长分档 | `STEP_FAR` / `STEP_NEAR` | `sim.js` 的 `seek()` |
| 模拟：推进逻辑 | `_step(dtMin)` | `sim.js` |
| 模拟：路径重算 | `_repath(a, dest)` | `sim.js` |
| 模拟：快照缓冲 | `fillSnapshot` | `sim.js` |
| 房间坐标定义 | `def floor_rooms` | `src/step7_office_layout.py` |
| 部门岗位池 | `ROLES_BY_DEPT` | `src/step7_office_layout.py` |
| 家具摆位规则 | `def build_floor_furniture` | `src/step7_office_layout.py` |
| 档案生成 | `def build_profiles` | `src/step8_employee_profiles.py` |

### 调试钩子（`window.__bld`）

在浏览器控制台可直接调用，方便验证改动：

```js
__bld.sim()                  // 拿 Sim 实例
__bld.seek(600)              // 跳到 10:00
__bld.pause(true)            // 暂停
__bld.setFloor('3')          // 只看 3F
__bld.recolor()              // 按当前配色模式重算
__bld.focusEmployee('E00005')// 定位并选中某员工
__bld.look(px,py,pz, tx,ty,tz)  // 直接设相机
__bld.shot()                 // 同步场景并导出 PNG（dataURL）
__bld.perf()                 // {calls, triangles, geometries, textures}
__bld.stats()                // {people, sim:{各状态人数}, t, doors, elevators}
__bld.sim().agents           // 全部 agent 内部状态（位置/路径/速度）
```

**验证性能的标准**：`__bld.perf().calls` 应保持在 200 左右，
`triangles` 约 10 万，播放时 fps 显示 60。若 calls 飙升到几千，说明
有人绕过了 `push()` 批量接口用了 `new THREE.Mesh`。

**验证走路平滑的标准**：
```js
const s = __bld.sim(); s.seek(606);
const w = s.agents.find(a => a.state === 'walk');
const tr=[]; for (let k=0;k<16;k++){ s.seek(606+k*0.012); tr.push([w.x,w.y]); }
// 相邻采样点距离应稳定在 0.6~0.8 米，且无 0、无大跳
```

