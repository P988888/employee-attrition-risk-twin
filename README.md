# 员工离职风险预测 · 公司总部大楼数字孪生

> **2026-10 更新｜3D 办公层已切换为 NEXUS 28 人办公区**
>
> 主风险大屏「3D楼层」、独立 `building.html` 与员工档案库 `employee_archive.html`
> 三个前端入口已统一为同一批 28 人展示团队。28 人名单由 `output/dashboard/office_team.json`
> 指定；点击 3D 人物会联动右侧员工档案卡 / 员工档案抽屉。
>
> 3D 场景文件为 `output/dashboard/office-embed.html`（自包含，无 CDN 依赖）。场景背景为深色数字天空、
> 星点/悬浮粒子、地面数字网格与雷达扫描底座，3D 窗口带青色 HUD 边缘；人物会离座、
> 沿安全通道行走，并与白板 / 窗边 / 水吧的同事停留交谈。点击人物后的「当前状态」由 3D 场景
> 实时回传（专注工作 / 起身离座 / 前往白板 / 与同事交流中 / 返回工位），不再读取旧的静态模拟状态。
> 底层 IBM 原始数据仍为 1470 条，模型评估口径不受影响。
> 3D 场景天空素材来自 Poly Haven「Joburg Central Sunset」（CC0，作者 Dimitrios Savva / Greg Zaal）：
> https://polyhaven.com/a/sunset_jhbcentral


基于 **IBM HR Analytics 真实离职数据（1470 条）** 构建的完整分析框架，
并延伸到**公司总部大楼的数字孪生**：28 人办公层展示团队、每人一份完整档案，
**点击 3D 场景里的人物即可查看他的全部数据**。

```
数据清洗 → 特征工程 → 多模型对比 → 风险评分 → 行为链路
        → 办公层布局 → 28 人档案 → 可点击 3D 数字孪生
```

---

## 快速开始

```bash
./run.sh                     # 一键跑通全流程（8 步）
```

浏览器打开：

| 页面 | 内容 |
|---|---|
| **<http://127.0.0.1:8777/building.html>** | **公司大楼数字孪生（点人查看档案）** |
| **<http://127.0.0.1:8777/employee_archive.html>** | **独立员工档案库（搜索姓名 / 工号 / 部门，查看部门名单）** |
| <http://127.0.0.1:8777/index.html> | 离职风险分析大屏 |

分步执行：

```bash
python3 src/step1_prepare_data.py        # 数据准备 + 模拟数据补充
python3 src/step2_train_model.py         # 特征工程 + 训练 + 评估 + 导出大屏数据
python3 src/step3_data_profile.py        # 数据整理与体检
python3 src/step4_linear_regression.py   # 线性回归（收入回归 + 线性概率模型）
python3 src/step5_build_behavior_link.py # 办公场景 5 张关联表
python3 src/step6_analyze_link.py        # 行为链路分析 + 员工风险评分导出
python3 src/step7_office_layout.py       # 4 层大楼布局（42 个工位 + 电梯核心筒）
python3 src/step8_employee_profiles.py   # 42 人完整档案
python3 src/validate_building_consistency.py  # 校验员工 / 房间 / 工位坐标一致
python3 src/build_notebook.py            # 生成 Jupyter notebook
cd output/dashboard && python3 -m http.server 8777
```

Blender 5.2 已安装时，可从权威四层布局一次导出四层静态环境（当前网页先启用 3F 试点）：

```bash
./company_3d/build_floor_assets.sh
```

导出的 `floor_1.glb` ~ `floor_4.glb` 会复制到 `output/dashboard/assets/office/`，打开 `building.html` 后选择 3F 即可验证；网页人物仍由 `sim.js` 实时驱动。

---

## 公司总部大楼数字孪生

### 建筑

| 项 | 值 |
|---|---|
| 规模 | 34 × 24 m × **4 层** = 3264 m²，层高 3.6 m |
| 垂直交通 | **2 部电梯 + 1 部疏散楼梯**（核心筒居中，每层可达） |
| 员工 | **42 人**（1F 10 / 2F 10 / 3F 14 / 4F 8） |
| 工位 | 42 个 |
| 家具 | 359 件 |

### 楼层与部门（按标准企业总部规模）

| 楼层 | 部门 | 人数 |
|---|---|---|
| **1F 大堂层** | 前台接待 / 客服中心 | 10 |
| **2F** | 研发部（开放办公区 / 会议室 / 机房 / 茶水间） | 10 |
| **3F** | 市场部 7 + 运营部 7（会议室 / 茶水间 / 打印区） | 14 |
| **4F** | 财务部 4 + 人力资源与行政 2 + 高管办公区 2 | 8 |

### 每人一份完整档案

`data/employee_profiles.json`（42 条）—— **点击 3D 人物时读取的就是这份数据**：

| 板块 | 内容 |
|---|---|
| **profile** | IBM HR Analytics 格式的 **35 个字段**（Age/Gender/JobRole/MonthlyIncome/OverTime/JobSatisfaction… 字段名与取值域与 IBM 数据集一致）+ **背调信息**（学历 / 院校 / 专业 / 证书 / 前雇主数 / 入职日期 / 合同类型 / 紧急联系人 / 背调结果） |
| **behavior** | 近 **90 天** 考勤与工时聚合（日均工时 / 专注 / 会议 / 消息 / 任务、迟到、请假、专注趋势）+ **近 7 天明细** + 工作质量（交付达成率 / 返工 / 缺陷率 / 效率分 / 质量分） |
| **assessment** | 测评报告：能力 / 敬业度 / 满意度 / 压力 / 综合 + 潜力评估 + 绩效评级 |
| **talks** | 谈话记录（类型 / 主题 / 情绪倾向 / 是否需跟进 / 时长） |
| **risk** | 离职风险评分 + 等级 + **可解释因子**（每条带方向与业务说明） |

> **数据口径**：IBM 格式字段按真实数据的经验分布采样（不凭空捏造取值范围）；
> 行为与测评是按业务逻辑合成的模拟数据。没有真实数据的人，**一律生成模拟数据**。

### 交互（点人查看档案）

在 `building.html` 里：

- **拖拽**旋转、**滚轮**缩放
- **点击任意人物** → 右栏展开他的完整档案（含风险因子与谈话记录）
- 顶部**楼层页签**：全楼 / 1F / 2F / 3F / 4F
- **按部门 / 按风险**两种着色切换
- **楼层间距**滑块：分层爆炸视图，逐层查看
- 单层页签会自动切换到该层楼内顶部俯视，并隐藏吊顶与灯具
- 各部门办公区会显示中文部门名称和英文副标题
- 单层视角会降低墙体透明度，保持隐私边界的同时看清员工和家具
- 员工档案使用模拟中文姓名，并展示性别、学历、入职日期等字段

集成钩子（供预测系统嵌入调用）：

```js
window.__bld.focusEmployee('E00075')   // 定位到该员工所在楼层并选中，返回姓名
window.__bld.select('E00075')          // 仅选中
window.__bld.setFloor('3')             // 只看 3F
window.__bld.recolor('risk')           // 按风险着色
```

---

## 目录结构

```
线性回归预测/
├── run.sh                          # 一键运行（8 步）
├── data/
│   ├── ibm_hr_attrition_raw.csv    # IBM 官方原始数据（1470 条，35 字段）
│   ├── link_*.csv                  # 办公场景 5 张关联表（行为/质量/谈话/测评）
│   ├── building_layout.json        # 唯一四层空间源（房间/工位/家具/电梯/路径）
│   ├── building_desks.csv          # 42 人 → 楼层 + 工位
│   ├── building_furniture.csv      # 359 件家具
│   ├── employee_profiles.json      # ★ 42 人完整档案（点击查看的数据源）
│   └── employee_profiles.csv       # 扁平版（便于表格/建模）
├── src/
│   ├── step1~step4                 # 数据 / 建模 / 体检 / 线性回归
│   ├── step5~step6                 # 行为链路多源数据与分析
│   ├── step7_office_layout.py      # 4 层大楼布局
│   └── step8_employee_profiles.py  # 42 人档案生成
├── notebooks/线性回归_员工离职分析.ipynb
├── company_3d/                     # Blender 3D 场景（建筑/家具/人物）
│   └── build_floor_assets.sh       # 从 building_layout 导出 1F~4F GLB
└── output/
    ├── figures/                    # 19 张图表（含各层平面图、大楼 3D 预览）
    ├── models/best_model.pkl
    └── dashboard/
        ├── building.html           # ★ 大楼数字孪生（点人查看档案）
        ├── index.html              # 离职风险分析大屏
        ├── building.json / profiles.json
        ├── vendor/                 # three.js + GLTFLoader（本地，可离线）
        └── assets/office/floor_3.glb # Blender 3F 静态环境试点
```

---

## 建模说明



### 数据口径（重要）

| 用途 | 数据 | 说明 |
|---|---|---|
| 模型训练与全部指标 | **仅 IBM 真实数据 1470 条** | 避免模拟数据把成绩"注水" |
| 3D 大屏视觉体量 | 真实 1470 + 模拟 1530 = **3000 条** | 模拟数据只用于点云数量，不参与指标 |

模拟数据按「先模仿真实边际与层级分布、再按真实 logistic 效应方向注入标签」生成，
离职率与真实数据对齐（16.1% vs 16.1%）。

### 特征工程

在 22 个数值特征 + 7 个类别特征基础上，构造了 7 个业务含义明确的派生特征
（全部为比值/差值，不含标签信息，无泄漏）：

| 特征 | 含义 |
|---|---|
| `IncomePerLevel` | 薪酬相对职级水平 |
| `TenureRatio` | 当前司龄 / 总工龄 |
| `PromotionGap` | 距上次晋升年限占司龄比 |
| `ManagerStability` | 与现任上级共事年限占司龄比 |
| `SatisfactionScore` | 四项满意度综合分 |
| `CareerStagnation` | 职业停滞指数 |
| `CompanyHopRate` | 跳槽频率 |

### 模型与评估

- 5 个模型对比：逻辑回归（含/不含 SMOTE）、决策树、随机森林、梯度提升
- 分层抽样 8:2 划分，5 折交叉验证
- 不平衡处理：`class_weight` + SMOTE（SMOTE 置于 `imblearn.Pipeline` 中，只在训练折内执行，不污染验证折）
- 阈值选择：不固定 0.5，取 **F1 最优点**（HR 场景下漏报成本高于误报）

**实测结果（真实测试集 294 条）**

| 模型 | AUC | F1 | 准确率 | 精确率 | 召回率 | CV-AUC |
|---|---|---|---|---|---|---|
| **逻辑回归(无采样) ★** | **0.8025** | **0.5532** | 0.8571 | 0.5532 | 0.5532 | 0.8354±0.0232 |
| 逻辑回归 | 0.7985 | 0.5306 | 0.8435 | 0.5098 | 0.5532 | 0.8369±0.0188 |
| 随机森林 | 0.7965 | 0.5255 | 0.7789 | 0.4000 | 0.7660 | 0.8263±0.0308 |
| 梯度提升 | 0.7910 | 0.5060 | 0.8605 | 0.5833 | 0.4468 | 0.8263±0.0264 |
| 决策树 | 0.6945 | 0.4286 | 0.8367 | 0.4865 | 0.3830 | 0.6942±0.0516 |

> **需要如实说明的两点：**
> 1. 最优模型召回率仅 **0.55**，即约 45% 的实际离职者未被预警（漏报 21 人）。
>    这是 16% 离职率下的典型水平，若要提高召回需接受精确率下降。
> 2. 渲染出的重要性系数与阈值**跨次运行稳定**（该目标函数平坦、`random_state=42`），
>    但绝对值会因 sklearn 版本差异略有波动。

---

## 3D 可视化大屏

三个可切换的 3D 视图（Three.js r160，本地 vendor 引入，可离线运行）：

| 视图 | 内容 |
|---|---|
| **风险点云** | 年龄 × 收入 × 司龄 三维空间，每点颜色=风险概率、大小=风险高低，已离职者加白色光环；含坐标轴与环形网格 |
| **部门风险柱阵** | 部门 × 风险等级 的 3D 柱状矩阵，柱高=人数，带数值标注 |
| **压力地形** | 年龄 × 司龄 分箱后的平均风险高度图，含线框叠加 |

其余面板：5 张 KPI 卡、风险等级分布、部门风险画像、高危职位 Top 6、
ROC 曲线（5 模型对比）、核心指标、特征重要性 Top 10、模型横向对比、最高风险员工列表。

### 技术要点

- **`importmap` + ES module**：three r160 起官方移除了 `examples/js` 全局构建，
  必须用 ES module 导入 `OrbitControls`（`vendor/` 已本地化，避免 CDN 版本漂移导致
  `THREE.OrbitControls is not a constructor`）。
- **点云着色器**：自定义 `ShaderMaterial`，按风险概率动态设置 `gl_PointSize`，圆形柔光点。
- **中文字体**：matplotlib 侧使用 macOS 实际可用字体（PingFang SC 等），
  原代码中的 `SimHei` 在 macOS 上不存在会导致中文变方框。

---

## 输出物

**评估图表**（`output/figures/`）
1. `01_模型对比.png` — 各模型指标对比 + 交叉验证 AUC
2. `02_ROC_PR曲线.png` — ROC 与 PR 曲线
3. `03_混淆矩阵.png` — 5 模型混淆矩阵
4. `04_特征重要性.png` — 特征重要性 Top 15（真实 one-hot 展开名）
5. `05_风险分布.png` — 风险分布 / 等级 / 部门
6. `06_相关性分析.png` — 特征与离职的相关性

**模型产物**
- `output/models/best_model.pkl` — 训练好的完整 Pipeline（含预处理）
- `output/models/model_meta.json` — 全部模型指标与元信息
- `output/employee_risk_scores.csv` — 3000 条员工风险评分明细

---

## 环境要求

```
Python 3.10+  (实测 3.14.6 / Anaconda)
pandas, numpy, scikit-learn, imbalanced-learn
matplotlib, seaborn, joblib
```

浏览器需支持 ES module 与 WebGL（Chrome / Edge / Safari 现代版本均可）。

> 大屏需通过 HTTP 访问（`fetch('data.json')` 受 CORS 限制，直接双击打开
> `index.html` 会加载失败），请使用 `run.sh` 或手动起 `http.server`。

---

## DeepSeek 云端 AI 查询

默认使用 **DeepSeek 云端模型**，不再占用本机 GPU：

- 工具选择 / 多轮取数：`deepseek-chat`
- 最终综合判断：`deepseek-reasoner`（内部思考，不输出思维链）
- Key 只放在项目根目录 `.env`，该文件权限 600，并已加入 `.gitignore`；不写进源码
- DeepSeek 不可用时自动回退到 Ollama 云端模型，再回退本地模型

配置（复制 `.env.example` 为 `.env` 并填入 Key）：

```bash
cp .env.example .env
# 编辑 .env：DEEPSEEK_API_KEY=sk-...
python3 src/hr_server.py 8791
```

Agent 已接入全部本地数据接口：

| 工具 | 能力 |
|---|---|
| `get_employee_360` | 档案 + 风险因子 + 行为 + 质量 + 测评 + 谈话 + 干预建议 |
| `query_dataset` | 任意查询 `profiles/risk/behavior/quality/talks/surveys/desks`，支持筛选、分组、聚合、跨表关联 |
| `get_department_workforce` | 部门级跨表画像：风险、加班、离职率、工时、交付、质量、压力、满意度、谈话 |
| `get_model_info` | 最佳模型、AUC/Ap/阈值、验证集指标、候选模型 |
| `get_office_layout` | 28 人楼层 / 房间 / 工位 / 风险分布 |
| 其他 | 员工主档、部门统计、离职率、风险榜单、整体概览、数据目录 |

查询口径与前端一致，固定为 **28 人办公展示团队**；底层 IBM 1470 条训练数据不进入员工问答口径。

---

## 本地模型查询（Ollama，无需云端 Key）

不配置 DeepSeek 时，可以完全跑在本机：

```bash
ollama serve                 # 启动本地模型服务（11434 端口）
ollama pull qwen2.5:1.5b     # 轻量模型
./查数据.sh                   # 交互式查询终端
```

三种模式：

| 命令 | 说明 |
|---|---|
| `./查数据.sh` | Agent 模式，模型自行选择查询工具，可处理灵活问法 |
| `./查数据.sh --route` | 关键词路由，响应快，问法需在预设范围内 |
| `./查数据.sh --no-model` | 纯查表，不调用模型，结果 100% 来自 CSV |

单次查询与网页接口：

```bash
python3 src/hr_query.py "谢心怡有什么干预手段"
python3 src/hr_query.py --no-model "查看离职率"     # 不依赖模型
python3 src/hr_server.py 8791                      # 同时提供大屏与 /api/ask
```

Agent 可调用的工具：员工档案、干预建议、员工名单、部门统计、离职率、
风险榜单、整体概览。所有数字由 pandas 确定性计算，模型只负责选择工具与组织语言，
避免小模型编造统计量。

---

## 数据与使用声明

- **IBM HR Analytics Employee Attrition** 数据集为公开研究数据，原始 1470 条。
- 为演示大屏体量补充的模拟员工（姓名、工位、行为、谈话记录等）**全部为程序生成**，
  不对应任何真实个人。
- 风险分数是统计筛查信号，不是对个人的事实判断。本项目**不用于、也不应被用于**
  自动做出辞退、降薪、处罚等决定；任何人事动作都需要人工复核。
- 3D 场景天空素材来自 Poly Haven「Joburg Central Sunset」（CC0）。

## License

[MIT](LICENSE)

