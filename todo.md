# 待办 / 已知问题（Todo）

代码已整理并推送：https://github.com/22ABLE22/snooker-assist

**安全提醒**：上传时使用过的 PAT 出现在对话中，**请立即到 GitHub → Settings → Developer settings 撤销该 token 并新建**。

按优先级排列。

### 2h. ~~本地启动步骤繁琐~~ ✅ 已完成
- 新增 `启动悬浮窗.bat`、`启动截图分析.bat`（固定 PEMAE 路径，失败再试 conda activate）

---

## P0 — 影响正确性

### 1. ~~台面长宽比未从实测 felt 读取~~ ✅ 已完成（见 done.md）
- 每帧 `analyze()` 用 `felt_rect` 的 `fw/fh` 调用 `geometry.set_table_aspect()`
- `find_pot_paths` / `find_escape_paths*` / `plan_*` / 走廊挡球 / 角度计算均用实测 aspect
- 验证：demo 实测 `aspect=2.087`（理论 2.007）；`test_measured_aspect_override` 通过

### 1b. ~~吃库镜面是否「内移半径」~~ ✅ 已核查（几何正确）
- 脚本：`output/check_mirror.py`
- left/right 镜面 = `rx = r/aspect`；top/bottom = `ry = r`
- 入射角 = 反射角（物理坐标，差值 0）
- **说明**：实机球路偏离参考线，主因是游戏物理（加塞/库弹性/速度），不是镜面公式错误
- 可选增强见 P3-13：吃库角校准系数

### 2b. ~~击球推荐权重偏简单~~ ✅ 已完成（斯诺克逻辑评分）
- `config.POT_KIND_BASE` / `OBJ_TO_POCKET_WEIGHT` / `CUT_ANGLE_PENALTY` / 中袋罚分等
- `geometry.score_pot_plan` / `score_escape_plan` / `sort_plans`
- 直接易球 > 吃库进球 > 传球 > 翻袋；解球一库优于多库
- 单测 `test_snooker_score_prefers_easy_pot` 通过
- 可再调：走位/安全球（未建模）见 P3

### 2c. ~~中袋未限制与长库夹角~~ ✅ 已完成
- 规则：目标球→中袋的进球线与**长库**夹角 &lt; 15° 则否决（几乎平行长库打不进）
- `config.MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG = 15.0`
- `geometry.middle_pocket_rail_ok`；直接/吃库进球/翻袋/传球（B 进中袋）均过滤
- 单测：浅角 2.85° 被拒；正对中袋 90° 通过

### 2d. ~~传球未按斯诺克规则限制颜色~~ ✅ 已完成
- 规则：仅「打红球」时允许**红传红**；打黄/绿/棕/蓝/粉/黑时**禁止任何传球**（先碰目标色且只进该色）
- `find_pot_paths(..., other_colors=)`；`analyze` 传入球色列表
- 单测：`test_plant_only_red_to_red` 通过

### 2e. ~~多库解球中间段可误碰目标球~~ ✅ 已完成
- 问题：目标球不在 `other_balls` 里，第一库与第二库之间路径扫到目标仍算通过
- 修复：`path_blocked` 将白球路径分段；**非最后一段**把目标球加入障碍；最后一段才允许接触
- 走廊仍为两侧 2R 扫过区域
- 单测：`test_multicushion_midpath_hits_target` 通过

### 2f. ~~角袋贴库未外偏瞄准点~~ ✅ 已完成
- 规则：与某条袋角库边夹角 &lt; 15° 时，瞄准点从角点沿**另一条**库边向台内偏移 **1 个球半径**
- `corner_aim_point` / `effective_pot_target`；直接、吃库进球、翻袋、传球均使用
- 配置：`CORNER_POCKET_MIN_JAW_ANGLE_DEG=15`，`CORNER_AIM_OFFSET_IN_RADIUS=1.0`
- 单测：`test_corner_aim_offset` 通过

### 2g. ~~台面边界不够贴内沿~~ ✅ 已完成
- 结构：绿库 → 黑阴影一圈 → 绿台；真边界 = 绿库与黑影交界，**台内包含阴影**
- `detect_felt`：在绿库内找大块暗影（过滤黑球/UI），阴影外接框即台面矩形
- 无阴影时回退；`FELT_INSET_RATIO` 调至 0.008
- 配置：`FELT_SHADOW_MAX_V` / `FELT_SHADOW_MIN_AREA_RATIO` / `FELT_SHADOW_MAX_AREA_RATIO`
- 实机：felt≈(151,220,1278,663)，检出球正常

### 2. 悬浮窗与截图坐标系可能不完全对齐
- **位置**：`overlay.py` 用窗口几何贴合游戏；标注用 `felt_rect` 在截图像素坐标绘制
- **问题**：DPI、窗口边框、客户区偏移时，线会整体偏一点
- **建议**：统一「窗口客户区原点」；或只在 felt 矩形内绘制并减去窗口 chrome

### 3. 绿球 / 棕球仍可能误检
- **位置**：`detect.py` HSV 范围与台呢、Logo、球杆重叠
- **问题**：复杂局面偶发错色，目标选错会导致线路全错
- **建议**：增加「颜色置信度」并在 status 里提示；允许用户在 overlay 里点选目标球

---

## P1 — 实操体验

### 4. F8 一次分析可能偏慢
- **原因**：`find_pot_paths` 对「每目标 × 每袋 × 多库序列 × 每颗 B 传球」组合爆炸
- **建议**：先只搜当前目标球；传球限制 B 为最近 3–4 颗；库序列先 1 库再 2 库

### 5. 无「锁定上一帧球局、只改目标重算」
- **位置**：`overlay.py` F10 只改颜色，需再按 F8 重截
- **建议**：缓存最近 `img`，F10 后直接 `analyze(cached)`

### 6. 鼠标穿透后无法点击悬浮窗
- **现状**：靠热键；若热键被占用则难操作
- **建议**：加小控制条（不穿透）或托盘图标

### 7. 热键 F12 常被系统占用
- **现状**：已用 Esc 退出；启动日志有 warn
- **建议**：改 F7 或 Ctrl+Shift+8 作为退出

---

## P2 — 代码卫生

### 8. 死代码 / 冗余
- `analyze.py`：`mode=="escape_only"` 分支无入口
- `config.py`：`VIS_COLORS` 未被绘制使用
- `geometry.py`：`segment_blocks_ball` 仅为兼容旧接口
- `detect.py`：`_classify_pixel` 可能未被调用

### 9. `path_blocked` 中 plant 分支与 `find_pot_paths` 内联检查重复
- **建议**：统一走 `corridor_hits_any`，plant 只在一处生成并校验

### 10. README 落后于实现
- 未写 plant、走廊挡球、实测 aspect、110° 角、各向异性镜面、F8 综合模式
- **建议**：同步一版「当前行为」说明

### 11. 缺少 detect / capture 自动化测试
- 现仅有几何单测
- **建议**：对 `output/` 里 1–2 张真实截图做「球数 / 白球存在」回归

### 12. `output/` 会堆积截图
- **建议**：只保留最近 N 张，或按日期分子目录

---

## P3 — 增强（可选）

### 13. 加塞 / 高低杆 / 库弹性吃库角校准
- 当前：标准理想镜面（入射=反射）
- 真实游戏：加塞改变出射角；库有「反弹系数」；高速略偏；中杆/高杆也有影响
- **建议**：`config` 增加 `CUSHION_REBOUND_SCALE`（出射角偏离理想角的校正，或出射角 = f(入射角)）
- 需用几组实测吃库点标定，再写回配置

### 14. 实时循环（每 0.5s 自动分析）
- 需注意 CPU 与游戏焦点

### 15. 中袋袋口几何更真实（袋角、有效进球角）
- 现为中心点 + 圆形危险区

### 16. 用户点选：框选台面、点白球/目标
- 比纯自动更稳

---

## 当前推荐配置（config.py）

| 参数 | 值 | 说明 |
|------|-----|------|
| `POT_MIN_ANGLE_DEG` | 110 | 进球最小夹角 |
| `POT_KIND_BASE` | direct0 / bank30 / plant55 /翻袋70 | 方式难度 |
| `OBJ_TO_POCKET_WEIGHT` | 35 | 球离袋 |
| `CUT_ANGLE_PENALTY` | 0.18/度 | 切角 |
| `POT_MIDDLE_POCKET_PENALTY` | 15 | 中袋难度分 |
| `MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG` | 15 | 中袋与长库最小夹角 |
| `ESCAPE_CUSHION_PENALTY` | 40/库 | 解球库数 |
| `BALL_RADIUS_SCALE` | 1.1 | 自动半径缩放 |
| `FELT_INSET_RATIO` | 0.008 | 阴影交界后再微缩 |
| `FELT_SHADOW_MAX_V` | 55 | 暗影亮度阈值 |
| `TABLE_ASPECT` | ≈2.007 | 仅回退；优先 felt 实测 |

微调顺序建议：球径 → 球心偏移 → 台面内缩 → 夹角 → 袋口余量。
