# 已完成（Done）

腾讯桌球 / 斯诺克外置解球辅助  
仓库：https://github.com/22ABLE22/snooker-assist （public）

## 功能

### 截图与窗口
- [x] 定位「腾讯桌球」窗口（标题精确匹配，排除 MiMo/编辑器误匹配）
- [x] PrintWindow / BitBlt / 置前后屏幕截 三路截取，按台面完整度自动选优
- [x] Per-Monitor DPI 感知，避免截残
- [x] `python main.py analyze|snap|demo|calibrate`

### 视觉识别
- [x] 绿色台呢检测：**优先六袋（黑洞）定位台面**，再台泥颜色内侧收边；回退阴影/轮廓
- [x] **多色台泥皮肤**：`FELT_CLOTH_MODE=auto` 时在绿/蓝/红/紫/黄中自动选主色
- [x] 浅绿装饰库：不再被外轮廓撑大（袋口 + 内侧收边）
- [x] 连通域为主、Hough 为辅检球
- [x] HSV 分色：白/红/黄/绿/棕/蓝/粉/黑
- [x] 斯诺克规则约束：彩球各最多 1 颗，红球 ≤15
- [x] 球半径去异常后统一（MAD + 稳健均值 + 理论交叉校正）
- [x] 可配置强制球径 `BALL_RADIUS_PX`、缩放 `BALL_RADIUS_SCALE`、球心偏移 `BALL_CENTER_BIAS_*`
- [x] 亮部质心校正，削弱右下阴影拖偏
- [x] **黑球**：暗核距离变换求球心；贴库阴影环与袋口黑洞不误检

### 几何 / 线路
- [x] 镜面反射解球：1–3 库
- [x] **球心镜面**：边线内移球半径；长短边各向异性（`rx = r/aspect`, `ry = r`）
- [x] **每帧使用台面实测 aspect**（`felt_w/felt_h` → `set_table_aspect`），`TABLE_ASPECT` 仅回退
- [x] 幽灵球 / 触点 / 角度 / 走廊挡球均按实测 aspect 做物理距离
- [x] 幽灵球 / 触点按物理距离计算（各向异性）
- [x] 袋口：角袋=矩形四角，中袋=长边中点；展示半径可调
- [x] 进球优先于解球（F8：先进球再解球；F9：只要进球）
- [x] 进球类型：直接、白球吃库后进球、翻袋、传球（**仅打红时红传红**；打彩球禁传）
- [x] **斯诺克难度权重排序**（分越低越好）：方式难度 + 球离袋距离 + 切角是否够直 + 中袋罚分 + 路径长 + 吃库数；解球按「库数优先、路径次之」
- [x] 进球终点对准袋口中心
- [x] 进球角 ≥ `POT_MIN_ANGLE_DEG`（当前配置 110°）
- [x] **中袋：进球线与长库夹角 ≥ 15°**
- [x] **角袋贴库：瞄准点沿另一库边外偏一个球半径**（如贴左库打左上角 → 瞄 `(rx,0)`；横打右上角 → 瞄 `(1,ry)`）
- [x] 碰撞点折线转角校验（触点/目标处 ≥ 最小角）
- [x] 袋口避让：吃库点与路径不得过近中袋/角袋
- [x] **路径走廊挡球**：线段两侧垂直扫过 2R，物理距离判定
- [x] **多库解球**：库与库之间不得扫到**目标球**；仅最后一段可触碰目标（`path_blocked` 分段 allow_target）
- [x] `AnalysisResult.felt_aspect` / 状态栏显示实测 aspect
- [x] **镜面反射几何核查**（`output/check_mirror.py`）：边线内移半径、长短边各向异性、入射角=反射角，代码正确

### 界面
- [x] 标注图输出到 `output/`
- [x] 置顶透明悬浮窗（tkinter Canvas，鼠标默认穿透）
- [x] 全局热键：F6 清台面锁定 / F7 校准锁定 / F8 解球综合 / F9 进球 / F10 换目标 / F11 显隐 / Esc 退出
- [x] **台面手动校准**：F7 拖四边或按钮微调，确认后本盘锁定；袋口按矩形生成
- [x] Win32 线程队列 RegisterHotKey + keyboard 库双保险

### 质量
- [x] `test_geometry.py` 覆盖：反射、吃库角、挡球、贴身球、袋口风险、进球角、走廊、中袋终点
- [x] 几何单测全部通过
- [x] 全模块 import / 语法检查通过

---

## 使用速查

```powershell
cd E:\Users\Admin\SNOOKER
python overlay.py          # 推荐：悬浮窗+热键
python main.py snap        # 命令行截图分析
python main.py demo        # 合成局面演示
python test_geometry.py    # 几何单测
```

或双击：

- `start_overlay.bat` / `启动悬浮窗.bat` — 启动悬浮窗（失败会停住显示错误码）  
- `start_snap.bat` / `启动截图分析.bat` — 一次 `snap` 截图分析  

| 热键 | 作用 |
|------|------|
| F6 | 清除台面锁定 |
| F7 | 校准台面并锁定（拖边/按钮） |
| F8 | 截图分析（优先进球，其次解球） |
| F9 | 只分析进球 |
| F10 | 切换目标球颜色 |
| F11 | 显示/隐藏标注 |
| Esc | 退出 |

---

## 架构

```text
capture.py   截屏 / 找窗口
detect.py    台面+球识别
geometry.py  镜面、走廊挡球、进球/解球搜索
analyze.py   管线 + 画标注
overlay.py   置顶悬浮窗 + 热键
main.py      CLI
config.py    全部可调参数
```
