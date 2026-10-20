# 命名规范速查（贴墙用）

## 公式

```
<前缀>_<主体>_<变体>_v<主>.<次><状态>
```

## 前缀

| 前缀 | 用途 | 示例 |
|---|---|---|
| `mdl_` | 模型 | `mdl_hero-chair_low_v03.00.fbx` |
| `tex_` | 贴图 | `tex_metal-panel_rough_v02.01.exr` |
| `mat_` | 材质 | `mat_concrete-wet_v01.02.blend` |
| `scn_` | 场景 | `scn_office-tower_day_v01.02.blend` |
| `rnd_` | 渲染 | `rnd_office-tower_hero_v01.04.png` |
| `prs_` | 预设 | `prs_archviz-interior_v01.00.blend` |
| `lib_` | 库资源 | `lib_hdri_studio-soft_v01.00.hdr` |
| `ref_` | 参考图 | `ref_office-tower_mood_v01.00.jpg` |
| `doc_` | 文档 | `doc_office-tower_shotlist_v01.00.md` |
| `scr_` | 脚本 | `scr_batch-export-v01.00.py` |

## 状态

`_wip` 制作中 → `_rvw` 待评审 → `_apr` 已通过（只读）→ `_fin` 已交付（只读）

## 版本

- 小改（材质/灯光/相机）：`v01.00` → `v01.01`
- 大改（结构/构图/重构）：`v01.03` → `v02.00`
- 永远两位补零。旧版本不删不覆盖。

## 五个禁止

1. 空格　2. 中文　3. `final` / `最终` / `new` / `修改版`
2. 日期当版本号　5. 覆盖 `_apr` / `_fin` 文件

## 记忆口诀

小写、前缀、下划线分隔；版本补零、状态收尾、旧版不删。
