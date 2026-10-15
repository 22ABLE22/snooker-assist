# 已完成（Done）

腾讯桌球 / 斯诺克外置解球辅助
仓库：https://github.com/22ABLE22/snooker-assist （public）

> 本文件只记录**已落地并验证过**的能力。
> 待办与已知问题见 [todo.md](todo.md)；球识别重做的细节见 [DETECT2_NOTES.md](DETECT2_NOTES.md)。

---

## 一、球体识别 v2（最近一次重大重做）

旧实现对 8 种颜色分别做 HSV 阈值切分，病根是阈值与台呢皮肤/光照强耦合、
同色球（绿球在绿台呢上）直接消失、球贴球时连通域粘连成一团。

**新方案改用与颜色无关的物理证据判球：**

- [x] **Lab 局部背景建模**：降采样中值估计台呢背景 → 感知色差图（抗光照/皮肤变化）
- [x] **双通道峰生成**
  - 色彩通道：圆盘−环形匹配滤波，**环形只统计台呢样像素**（贴库球不再被库边压制）
  - 明暗通道（Vstd）：球体必有高光渐变，是**唯一能发现同色球**的通道
- [x] **`disk_fill_ratio` 主判据**：球是实心圆盘（覆盖率 88.8~100%），
      库边只切过一侧（43.6~75.4%）→ 彻底排除库边/高光/球杆
- [x] **证据门（色差 OR Vstd）+ 置信度门限**：兼顾真实球与合成纯色球，
      排除台呢「腾讯桌球」水印弧线（误检 conf=0.43 vs 真球 0.90~0.99）
- [x] **实测袋心定位六袋** + 由袋心反推比赛区（不假设袋在矩形角上）
- [x] **半径两步标定**：匹配滤波粗估 → 距离变换精修（阈值经弦长法标定，误差 ≤0.02px）
- [x] **帧间零抖动**：12 帧离线 + 15 次实时连续抓取，球数/半径/颜色分布完全一致
- [x] **接口层兼容**：`detect.py` 保留旧的全部公开接口，`analyze/overlay/main` 零改动
- [x] 实测**满台局面**（22 颗：15 红 + 6 彩 + 白）比旧实现多找回 1 颗贴库红球

### 识别实测数据

| 场景 | 结果 |
|------|------|
| 离线 12 帧 | 19 颗 / r=19.77px / 颜色分布 **12/12 帧零波动** |
| 实时连续 15 次 | **15/15 次均 19 颗**，颜色分布完全一致 |
| 坐标合规 | **12/12 帧通过**（球心全在可及范围内） |
| 满台 22 颗 | 22 颗全对，颜色 0 错误，与旧实现逐球偏差 0.6~3.8px |
| 合成 demo | 5/5，与旧实现一致 |
| 速度 | 识别 0.6s + 几何 0.6s ≈ 1.2s/次 |

---

## 二、截图与窗口

- [x] 定位「腾讯桌球」窗口（标题精确匹配，排除编辑器/本工具误匹配）
- [x] PrintWindow / BitBlt / 置前后屏幕截 三路截取，按台面完整度自动选优
- [x] Per-Monitor DPI 感知，避免截残
- [x] `python src/main.py analyze|snap|demo|calibrate`

## 三、台面定位

- [x] 六袋（黑洞）定位台面 → 按袋半径内缩 → 台泥颜色**只向内**收边
- [x] 多色台泥皮肤：`FELT_CLOTH_MODE=auto` 在绿/蓝/红/紫/黄中自动选主色
- [x] 浅绿装饰库不再撑大外轮廓
- [x] **比赛区坐标系**：`felt_rect` = 可放球范围（四角即角袋中心），
      与 geometry 的库边镜面 `y=r` 前提一致；球心夹取保证不越界
- [x] 台面手动校准（F7 拖边/按钮微调 → 锁定，F6 清除）

## 四、几何与线路

- [x] 镜面反射解球：1–3 库；**球心镜面**，长短边各向异性（`rx=r/aspect`, `ry=r`）
- [x] **每帧使用台面实测 aspect**（`felt_w/felt_h`），`TABLE_ASPECT` 仅回退
- [x] 幽灵球 / 触点 / 走廊挡球均按实测 aspect 做物理距离
- [x] 进球类型：直接、白球吃库后进球、翻袋、传球（**仅打红时红传红**，打彩球禁传）
- [x] **斯诺克难度权重排序**：方式难度 + 球离袋 + 切角 + 中袋罚分 + 路径长 + 吃库数
- [x] 进球角 ≥ `POT_MIN_ANGLE_DEG`（当前 110°）
- [x] 中袋：进球线与长库夹角 ≥ 15°，否则否决
- [x] 角袋贴库：瞄准点沿另一库边外偏一个球半径
- [x] 袋口避让：吃库点与路径不得过近中袋/角袋
- [x] **路径走廊挡球**：线段两侧垂直扫过 2R，物理距离判定
- [x] 多库解球：库与库之间不得扫到目标球，仅最后一段可触碰（`path_blocked` 分段）
- [x] 镜面几何核查（`tools/check_mirror.py`）：内移半径、各向异性、入射=反射

## 五、界面

- [x] 置顶透明悬浮窗（tkinter Canvas，鼠标默认穿透）
- [x] 全局热键：F6 清锁定 / F7 校准 / F8 综合 / F9 进球 / F10 换目标 / F11 显隐 / Esc 退出
- [x] Win32 线程队列 RegisterHotKey + keyboard 库双保险
- [x] 台面锁定持久化到 `output/felt_lock.json`（存相对窗口比例，窗口移动仍可用）

## 六、工程与质量

- [x] `tests/test_geometry.py` 覆盖：反射、吃库角、挡球、贴身球、袋口风险、
      进球角、走廊、中袋终点、传球合法性、难度排序
- [x] `tests/test_detect.py` **识别回归测试**（216 项断言，基于 12 帧真实样本）：
      台面定位 / 球数 / 白球唯一 / 颜色分布与斯诺克约束 / 半径区间 /
      球心可及范围 / 接口层与引擎一致
- [x] `dev/verify_integration.py`：接口兼容 / 检测稳定性 / 坐标合规 / 几何可用 / 锁定模式
- [x] `dev/live_stability.py`：实时连续抓取稳定性
- [x] `dev/compare_frames.py`：新旧坐标系贴库合规性对比
- [x] `dev/bench.py` + `dev/deliver.py`：新旧方案并排评测与批量抓帧交付
- [x] 全部模块 import / 语法检查通过
- [x] 子代理独立审计识别模块，9 项缺陷全部修复（详见 DETECT2_NOTES.md）

## 七、项目结构整理

- [x] 目录化：`src/`（核心）`tests/`（单测）`dev/`（开发与验证）`tools/`（独立工具）
      `docs/`（文档与历史）`samples/`（参考样本）`output/`（运行产物）
- [x] 消除全部硬编码绝对路径（`E:\Users\Admin\...`），改为按 `__file__` 相对定位
- [x] bat 入口改用 `%~dp0`，仓库移动/改名均可用
- [x] 清理历史截图（output 从 133.7MB → 0.9MB）
- [x] 旧实现保留为 `dev/detect_legacy.py` 作基线对比，不参与运行时

---

## 使用速查

```powershell
# 悬浮窗 + 热键（推荐）
start_overlay.bat
# 或
D:\ProgramData\miniconda3\envs\PEMAE\python.exe src\overlay.py

# 命令行
python src\main.py snap                    # 截图分析
python src\main.py analyze <图片> --target red --both
python src\main.py demo                    # 合成局面演示

# 测试与验证
python tests\test_geometry.py              # 几何单测
python tests\test_detect.py                # 识别回归（216 项断言）
python dev\verify_integration.py           # 识别全链路验证
python dev\live_stability.py               # 实时稳定性（需游戏窗口可见）
```

| 热键 | 作用 |
|------|------|
| F6 | 清除台面锁定（恢复自动检测） |
| F7 | 校准台面并锁定（拖边/按钮） |
| F8 | 截图分析（优先进球，其次解球） |
| F9 | 只分析进球 |
| F10 | 切换目标球颜色 |
| F11 | 显示/隐藏标注 |
| Esc | 退出 |

## 架构

```text
src/
  config.py      全部可调参数（含目录常量）
  capture.py     截屏 / 找窗口
  detect2.py     识别引擎（Lab 局部背景 + 双通道物理证据）
  detect.py      识别接口层（对外保留旧接口，内部调用 detect2）
  geometry.py    镜面、走廊挡球、进球/解球搜索与评分
  analyze.py     分析管线 + 标注绘制
  overlay.py     置顶悬浮窗 + 热键
  main.py        CLI 入口
  hotkey.py      兼容入口（转发 overlay）
tests/
  test_geometry.py             几何与规则单测
  test_detect.py               识别回归（12 帧样本，216 项断言）
dev/
  bench.py / deliver.py        新旧评测、批量交付
  detect_legacy.py             旧 HSV 实现（基线）
  verify_*.py / live_stability.py / compare_*.py
  calib_*.py / measure_*.py    标定与测量脚本
tools/
  check_mirror.py              镜面几何独立核查
docs/
  DETECT2_NOTES.md             识别重做说明（思路/标定/实测）
  history/                     历史邮件等原始材料
samples/
  recognize_12frames/          12 帧真实样本 + 识别结果 + 打分表
  evidence/                    关键证据图（满台 22 颗、找回的贴库球等）
```
