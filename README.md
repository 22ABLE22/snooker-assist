# 腾讯桌球 / 斯诺克解球辅助

对「腾讯桌球」等**俯视 2D 台球**界面做截图识别，用镜面反射与斯诺克规则搜索**进球 / 解球**线路，并用置顶悬浮窗叠加参考线。

> 仅作训练与瞄准参考。加塞、库弹性、力度等真实物理未完全建模；请遵守游戏条款。

## 功能

| 模块 | 说明 |
|------|------|
| 截屏 | 定位「腾讯桌球」窗口；PrintWindow / BitBlt / 屏幕截 三路选优；DPI 感知 |
| 识别 | 台面（同色库→黑影→台泥，**多色皮肤自动识别**）、球位置与颜色、统一球径 |
| 几何 | 球心镜面（长短边各向异性）、1–3 库解球、走廊挡球 |
| 进球 | 直接 / 吃库后进球 / 翻袋 / 红传红；中袋与角袋角度规则 |
| 排序 | 斯诺克难度分：方式、球离袋、切角、中袋、库数、路径长 |
| 界面 | CLI 分析；置顶透明悬浮窗 + 全局热键 |

## 快速开始

```powershell
# 依赖（Python 3.10+）
pip install -r requirements.txt

# 几何自检
python test_geometry.py

# 合成局面演示
python main.py demo

# 打开游戏后：悬浮窗 + 热键（推荐）
# 双击 启动悬浮窗.bat 即可（自动进 PEMAE 环境）
python overlay.py

# 或命令行截图分析
python main.py snap
# 双击 启动截图分析.bat
python main.py analyze "path\to\shot.png" --target red --both
```

## 悬浮窗热键

| 键 | 作用 |
|----|------|
| **F8** | 截图分析（优先进球，其次解球） |
| **F9** | 只分析进球 |
| **F10** | 切换目标球（红→黄→绿→棕→蓝→粉→黑） |
| **F11** | 显示 / 隐藏标注 |
| **Esc** | 退出 |

默认**鼠标穿透**，可直接在游戏内击球。

## 命令行

```powershell
python main.py snap --target blue          # 解/打蓝球
python main.py snap --pot                  # 只要进球线
python main.py snap --both --cushions 2    # 进球+解球，最多 2 库
python main.py snap --no-gui               # 不弹预览，只写 output/
python main.py calibrate                   # 标定说明
```

## 标注含义

- 彩色折线：推荐路径（#1 分最低为首选）
- 绿点：吃库点（球心路径）
- 品红圆：幽灵球 / 触球点
- 橘色圆：袋口（角袋=矩形四角，中袋=长边中点）
- 顶部状态：目标颜色、方案数、台面 aspect、首选难度分

## 关键规则（已实现）

1. **镜面**：库边向内平移球半径；`rx = r/aspect`，`ry = r`；每帧用实测 felt 长宽比  
2. **挡球**：路径向两侧扫过 2R 走廊；多库时中间段不得碰到目标球，仅最后一击可碰  
3. **进球角**：目标处夹角 ≥ `POT_MIN_ANGLE_DEG`（默认 110°）  
4. **中袋**：与长库夹角 ≥ 15°，否则不可打中袋  
5. **角袋贴库**：与某条袋角库边夹角 &lt; 15° 时，瞄准点沿另一库边外偏一个球半径  
6. **传球**：仅打红时允许**红传红**；打彩球禁止传球  
7. **排序**：直接易球优先于吃库/传球/翻袋；解球一库优于多库  
8. **台泥皮肤**：默认 `FELT_CLOTH_MODE="auto"`，在绿/蓝/红/紫/黄中自动选主色；也可在 `config.py` 写死  

> 红色台泥与红球同色，红球会明显变难，属已知限制。

## 配置（`config.py`）

可调项包括：球径 `BALL_RADIUS_PX` / `BALL_RADIUS_SCALE`、球心偏移、台面内缩、阴影阈值、各难度权重、中袋/角袋角度等。改完无需改代码，重启 `overlay.py` 即可。

## 项目结构

```text
main.py          CLI 入口
overlay.py       置顶悬浮窗 + 全局热键
capture.py       截屏 / 找窗口
detect.py        台面与球识别
geometry.py      镜面、走廊、进球/解球搜索与评分
analyze.py       分析管线与标注绘制
config.py        全部可调参数
test_geometry.py 几何与规则单测
hotkey.py        兼容入口（转发 overlay）
done.md          已完成能力
todo.md          已知问题与待办
```

## 测试

```powershell
python test_geometry.py
```

覆盖：反射与吃库角、挡球走廊、中袋/角袋规则、传球合法性、斯诺克难度排序等。

## 已知限制

- 不建模加塞 / 高低杆引起的吃库角变化  
- 中袋袋口几何为近似  
- 绿球/棕球在复杂 UI 下仍可能误检  
- 需要 Windows + 俯视「腾讯桌球」类画面  

## License

MIT
