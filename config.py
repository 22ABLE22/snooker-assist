# -*- coding: utf-8 -*-
"""桌球辅助解球 —— 全局配置"""

from pathlib import Path

# 输出目录
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# 默认截取区域（像素）。首次使用请用 main.py 的 calibrate 模式标定。
# 只覆盖台面（含库边），不要包含左右 UI 侧栏。
TABLE_REGION = {
    "left": 140,
    "top": 200,
    "width": 1680,
    "height": 840,
}

# 窗口标题关键词（精确标题「腾讯桌球」优先级更高）
WINDOW_TITLE_HINTS = ["腾讯桌球", "T-BALL"]

# 台面物理比例（国际标准斯诺克内沿约 3569 x 1778 mm）
TABLE_ASPECT = 3569.0 / 1778.0  # ≈ 2.007

# 几何坐标：x∈[0,1] 沿长边，y∈[0,1] 沿短边。
# 球半径 r 一律定义为「相对短边」；长边方向半径 = r / TABLE_ASPECT
# （否则左右库内移量会大约一倍，像用了直径）

# 球半径占台面短边比例的近似值（理论值，仅作参考/交叉校正）
# 标准球直径 52.5mm，短边 1778mm → 半径 ≈ 1.48%
BALL_RADIUS_RATIO = 0.0148

# ===== 球径与球心（可手动微调）=====
# 强制球半径（像素）。>0 时覆盖自动估计。请在标注图上量一颗球的半径填入
BALL_RADIUS_PX = 0
# 自动估计后的整体缩放。右下阴影使检测半径偏小时，可调 1.10 ~ 1.30
BALL_RADIUS_SCALE = 1.1
# 球心像素偏移（相对检测值）。阴影把中心拉向右下时，可试 (-0.5, -0.5)
BALL_CENTER_BIAS_X = 0.6
BALL_CENTER_BIAS_Y = 0.6

# 台面外接矩形再内缩比例（相对短边）。已按「绿库-黑影交界」取内沿后，仅留极小余量
FELT_INSET_RATIO = 0.002

# 黑阴影（库内沿一圈）检测
FELT_SHADOW_MAX_V = 55          # V 低于此视为暗/影
FELT_SHADOW_MIN_AREA_RATIO = 0.01  # 相对整图，过小的暗块（黑球等）丢弃
FELT_SHADOW_MAX_AREA_RATIO = 0.45

# 袋口标注半径（相对台面短边）。视觉约为球径的 1.8~2.2 倍
POCKET_DRAW_RADIUS_RATIO = 0.05

# 库边安全距离（球心到库边至少这个比例才认为合法路径不擦库）
CUSHION_MARGIN_RATIO = 0.012

# 袋口危险区（相对短边）。吃库点/路径过近会摔袋或撞袋角
# 中袋袋口更宽，留更大余量
POCKET_CLEARANCE_CORNER = 0.055
POCKET_CLEARANCE_MIDDLE = 0.085
# 路径线段经过袋口中心的危险半径（略小于吃库点余量，允许从袋口前掠过但不穿心）
POCKET_PASS_RADIUS_CORNER = 0.040
POCKET_PASS_RADIUS_MIDDLE = 0.055

# 路径搜索最大库数
MAX_CUSHIONS = 3

# 进球最小夹角（度）：目标球处「白球来向」与「去袋方向」夹角。
POT_MIN_ANGLE_DEG = 110.0

# ===== 斯诺克击球推荐权重（分越低越优先）=====
# 依据：先打容易进的球；库数少、球近袋、切角薄、中袋、多库/翻袋/传球都更难。

# 进球方式基础难度
POT_KIND_BASE = {
    "direct": 0.0,          # 直接进球（首选）
    "cue_bank_pot": 30.0,   # 白球吃库后进球
    "plant": 55.0,          # 传球/组合球
    "object_bank": 70.0,    # 目标球翻袋
}

# 解球：每多一库（斯诺克解球 1 库远优于 3 库）
ESCAPE_CUSHION_PENALTY = 40.0

# 路径总长（相对短边的物理长度）
PATH_LENGTH_WEIGHT = 10.0

# 目标球心 → 袋口中心距离（越近越好）
OBJ_TO_POCKET_WEIGHT = 35.0

# 切球角：目标处夹角每偏离 180° 一度的惩罚（越直越好）
CUT_ANGLE_PENALTY = 0.18

# 贴近合法阈值（POT_MIN_ANGLE_DEG）的风险附加
ANGLE_MARGIN_PENALTY = 0.40  # 每度「阈值余量不足」

# 中袋比角袋更难进（袋口几何、角度窗口更窄）
POT_MIDDLE_POCKET_PENALTY = 15.0

# 中袋：进球线与长库夹角（度）。过小（几乎平行长库）无法打进中袋
MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG = 15.0

# 角袋：与某一条袋角库边夹角过小时，瞄准点沿另一条库边外偏（球半径）
# 例：贴左库打左上角 → 瞄准点从角点沿上库右移一个半径
CORNER_POCKET_MIN_JAW_ANGLE_DEG = 15.0
CORNER_AIM_OFFSET_IN_RADIUS = 1.0  # 沿更开阔库边的偏移量（球半径倍数）

# 白球吃库段数（进球时每库额外小罚，与 KIND_BASE 叠加）
POT_CUSHION_PENALTY = 8.0

# 旧版简单排序（解球兜底）
RANK_CUSHION_WEIGHT = 120.0
RANK_LENGTH_WEIGHT = 1.0

# 颜色分类 HSV 范围（OpenCV H:0-179）
# 可按实际截图微调
COLOR_RANGES = {
    "cue": {  # 白球
        "h": [(0, 180)],
        "s": [(0, 50)],
        "v": [(180, 255)],
    },
    "red": {
        "h": [(0, 10), (170, 179)],
        "s": [(100, 255)],
        "v": [(80, 255)],
    },
    "yellow": {
        "h": [(18, 38)],
        "s": [(120, 255)],
        "v": [(120, 255)],
    },
    "green": {
        "h": [(40, 85)],
        "s": [(80, 255)],
        "v": [(80, 255)],
    },
    "brown": {
        "h": [(8, 22)],
        "s": [(100, 255)],
        "v": [(40, 140)],
    },
    "blue": {
        "h": [(95, 130)],
        "s": [(100, 255)],
        "v": [(80, 255)],
    },
    "pink": {
        "h": [(150, 175)],
        "s": [(40, 140)],
        "v": [(160, 255)],
    },
    "black": {
        "h": [(0, 180)],
        "s": [(0, 80)],
        "v": [(0, 50)],
    },
}

# 目标球可选颜色（解球时可指定）
TARGET_COLORS = ["red", "yellow", "green", "brown", "blue", "pink", "black"]

# 可视化颜色 BGR
VIS_COLORS = {
    "path": (0, 220, 255),       # 主路径黄
    "path_alt": (255, 160, 0),   # 备选路径青
    "ghost": (255, 0, 255),      # 幽灵球紫
    "cushion": (0, 255, 0),      # 库点绿
    "target": (0, 0, 255),       # 目标红
    "blocked": (80, 80, 255),    # 被挡淡红
    "pocket": (0, 140, 255),     # 袋口橙
}

# 截图保存前缀
SNAPSHOT_PREFIX = "shot"
