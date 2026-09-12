# -*- coding: utf-8 -*-
"""镜面反射解球几何（一库/多库）"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from config import (
    ANGLE_MARGIN_PENALTY,
    BALL_RADIUS_RATIO,
    CORNER_AIM_OFFSET_IN_RADIUS,
    CORNER_POCKET_MIN_JAW_ANGLE_DEG,
    CUT_ANGLE_PENALTY,
    CUSHION_MARGIN_RATIO,
    ESCAPE_CUSHION_PENALTY,
    MAX_CUSHIONS,
    MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG,
    OBJ_TO_POCKET_WEIGHT,
    PATH_LENGTH_WEIGHT,
    POCKET_CLEARANCE_CORNER,
    POCKET_CLEARANCE_MIDDLE,
    POCKET_PASS_RADIUS_CORNER,
    POCKET_PASS_RADIUS_MIDDLE,
    POT_CUSHION_PENALTY,
    POT_KIND_BASE,
    POT_MIDDLE_POCKET_PENALTY,
    POT_MIN_ANGLE_DEG,
    TABLE_ASPECT,
)

# 四条库边：名称与直线
# 台面归一化坐标 [0,1] x [0,1]
# top: y=0, bottom: y=1, left: x=0, right: x=1
CUSHIONS = ("top", "bottom", "left", "right")

# 本帧台面实测长宽比（felt 宽/高）。由 analyze 在每帧分析前写入；
# 为 None 时回退 config.TABLE_ASPECT。
_TABLE_ASPECT_OVERRIDE: float | None = None


def set_table_aspect(asp: float | None) -> None:
    """设置本帧台面实测 aspect = felt_w / felt_h。"""
    global _TABLE_ASPECT_OVERRIDE
    if asp is not None and 1.2 <= float(asp) <= 3.5:
        _TABLE_ASPECT_OVERRIDE = float(asp)
    else:
        _TABLE_ASPECT_OVERRIDE = None


def get_aspect(asp: float | None = None) -> float:
    """取有效 aspect：显式参数 > 本帧实测 > config 理论值。"""
    if asp is not None:
        a = float(asp)
        if 1.2 <= a <= 3.5:
            return a
    if _TABLE_ASPECT_OVERRIDE is not None:
        return _TABLE_ASPECT_OVERRIDE
    return float(TABLE_ASPECT)


def radii_xy(r: float, aspect: float | None = None) -> tuple[float, float]:
    """把「相对短边」的球半径 r 拆成归一化坐标系下的 (rx, ry)。

    x 沿长边、y 沿短边，故 rx = r / aspect，ry = r。
    """
    a = get_aspect(aspect)
    r = max(0.0, float(r))
    return (r / a, r)


@dataclass
class PathPlan:
    kind: str  # "direct" | "cue_bank" | "cue_bank_pot" | "object_bank" | "plant"
    cushions: list[str]
    points: list[tuple[float, float]]  # 归一化路径折点（含起点终点）
    aim_dir: tuple[float, float]  # 白球初始出杆方向（单位向量）
    impact_point: tuple[float, float]  # 白球第一触球点（幽灵球）
    length: float
    target_color: str
    target_index: int
    notes: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    # 推荐排序用
    target_pos: tuple[float, float] | None = None
    pocket_pos: tuple[float, float] | None = None
    score: float = 0.0  # 越小越优先

    @property
    def n_cushions(self) -> int:
        return len(self.cushions)

    @property
    def is_pot(self) -> bool:
        return self.kind in ("direct", "cue_bank_pot", "object_bank", "plant")


def _norm(p: tuple[float, float]) -> tuple[float, float]:
    x, y = p
    L = (x * x + y * y) ** 0.5
    if L < 1e-12:
        return (0.0, 0.0)
    return (x / L, y / L)


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _path_length(pts: list[tuple[float, float]]) -> float:
    return sum(_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def cushion_axis_value(cushion: str, r: float = 0.0, aspect: float | None = None) -> float:
    """球心反射线坐标：边线向内平移球半径（长短边分别用 rx/ry）。"""
    rx, ry = radii_xy(r, aspect)
    if cushion == "top":
        return ry
    if cushion == "bottom":
        return 1.0 - ry
    if cushion == "left":
        return rx
    if cushion == "right":
        return 1.0 - rx
    raise ValueError(cushion)


def reflect_point(
    p: tuple[float, float], cushion: str, r: float = 0.0, aspect: float | None = None
) -> tuple[float, float]:
    """关于「球心镜面」反射（边线内移 rx/ry）。"""
    x, y = p
    if cushion == "top":
        yb = cushion_axis_value("top", r, aspect)
        return (x, 2.0 * yb - y)
    if cushion == "bottom":
        yb = cushion_axis_value("bottom", r, aspect)
        return (x, 2.0 * yb - y)
    if cushion == "left":
        xb = cushion_axis_value("left", r, aspect)
        return (2.0 * xb - x, y)
    if cushion == "right":
        xb = cushion_axis_value("right", r, aspect)
        return (2.0 * xb - x, y)
    raise ValueError(cushion)


def reflect_seq(
    p: tuple[float, float],
    cushions: list[str],
    r: float = 0.0,
    aspect: float | None = None,
) -> tuple[float, float]:
    acc = p
    for c in reversed(cushions):
        acc = reflect_point(acc, c, r, aspect)
    return acc


def line_segment_hits_segment(p1, p2, q1, q2, eps=1e-9):
    """两线段是否相交（含端点邻域）。"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True
    return False


def segment_hits_circle(p1, p2, c, r, eps=1e-9) -> bool:
    """线段是否与圆相交（用于挡球检测）。圆心 c，半径 r。"""
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = c
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx * dx + dy * dy
    if a < eps:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 <= r * r
    b = 2 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * cc
    if disc < 0:
        return False
    sq = disc ** 0.5
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def segment_blocks_ball(
    p1: tuple[float, float],
    p2: tuple[float, float],
    ob: tuple[float, float],
    block_r: float,
    eps: float = 1e-12,
    aspect: float | None = None,
) -> bool:
    """线段 p1→p2 是否与障碍球碰撞圆相交（旧接口，block_r 为半宽）。

    请优先用 path_corridor_blocks()。
    """
    return path_corridor_blocks(p1, p2, ob, half_width=block_r, aspect=aspect, eps=eps)


def path_corridor_blocks(
    p1: tuple[float, float],
    p2: tuple[float, float],
    ob: tuple[float, float],
    r: float | None = None,
    half_width: float | None = None,
    aspect: float | None = None,
    eps: float = 1e-12,
) -> bool:
    """路径走廊挡球判定（物理距离）。

    把球心路径线段向两侧垂直平移 r（球半径）得到宽 2r 的扫过矩形；
    干扰球球心再带半径 r，故球心到路径的物理距离 ≤ 2r 即视为碰到。

    - 传 r=球半径（相对短边）时，走廊半宽 = 2*r
    - 也可直接传 half_width（相对短边的物理半宽）

    在长短边比例 aspect 的坐标系下计算垂直距离（x 先乘 aspect）。
    """
    if half_width is None:
        if r is None:
            return False
        half_width = 2.0 * r
    a = get_aspect(aspect)

    # 物理坐标（以短边为单位）
    x1, y1 = p1[0] * a, p1[1]
    x2, y2 = p2[0] * a, p2[1]
    cx, cy = ob[0] * a, ob[1]

    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < eps:
        # 退化为点：点到障碍球距离
        return ((x1 - cx) ** 2 + (y1 - cy) ** 2) < half_width * half_width

    # 障碍球心在线段上的投影参数
    t = ((cx - x1) * dx + (cy - y1) * dy) / L2
    # 贴身白球在起点且球在身后：不挡
    d0 = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
    if d0 < half_width and t <= 0.0:
        return False

    tc = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = x1 + tc * dx, y1 + tc * dy
    # 球心到路径的物理垂直距离
    dist = ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5
    return dist < half_width


def corridor_hits_any(
    p1: tuple[float, float],
    p2: tuple[float, float],
    obstacles: list[tuple[float, float]],
    r: float,
    aspect: float | None = None,
    skip: list[tuple[float, float]] | None = None,
    skip_tol: float | None = None,
) -> tuple[tuple[float, float] | None, str]:
    """扫过走廊是否碰到任一障碍球。返回 (障碍球心|None, 原因)。"""
    tol = skip_tol if skip_tol is not None else 0.3 * r
    for ob in obstacles:
        if skip:
            hit_skip = False
            for s in skip:
                if _dist(ob, s) < tol:
                    hit_skip = True
                    break
            if hit_skip:
                continue
        if path_corridor_blocks(p1, p2, ob, r=r, aspect=aspect):
            return ob, f"路径走廊碰到球 @({ob[0]:.3f},{ob[1]:.3f})"
    return None, ""


def fold_path(
    start: tuple[float, float],
    virtual_end: tuple[float, float],
    cushions: list[str],
    r: float = 0.0,
    aspect: float | None = None,
) -> list[tuple[float, float]]:
    """把 start→virtual_end 的直线按反射序折回真实台面折点。

    镜面为「球心反射线」（边线内移 rx/ry）。
    """
    pts = [start]
    end = virtual_end
    seg_a, seg_b = start, end
    for c in cushions:
        hit = intersect_line_with_cushion(seg_a, seg_b, c, r=r, aspect=aspect)
        if hit is None:
            return []
        pts.append(hit)
        seg_a = hit
        seg_b = reflect_point(seg_b, c, r, aspect)
    pts.append(seg_b)
    return pts


def intersect_line_with_cushion(
    p1, p2, cushion: str, eps=1e-9, r: float = 0.0, aspect: float | None = None
) -> tuple[float, float] | None:
    """求直线 p1-p2 与「球心镜面」的交点（前进方向）。"""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if cushion in ("top", "bottom"):
        yb = cushion_axis_value(cushion, r, aspect)
        if abs(dy) < eps:
            return None
        t = (yb - y1) / dy
        if t <= 1e-9 or t > 1 + 1e-6:
            return None
        x = x1 + t * dx
        if x < -0.05 or x > 1.05:
            return None
        return (x, yb)
    else:
        xb = cushion_axis_value(cushion, r, aspect)
        if abs(dx) < eps:
            return None
        t = (xb - x1) / dx
        if t <= 1e-9 or t > 1 + 1e-6:
            return None
        y = y1 + t * dy
        if y < -0.05 or y > 1.05:
            return None
        return (xb, y)


def _in_table(p, margin=0.0) -> bool:
    x, y = p
    return margin <= x <= 1 - margin and margin <= y <= 1 - margin


def _segments_valid(
    pts: list[tuple[float, float]], margin: float, r: float = 0.0, aspect: float | None = None
) -> bool:
    if len(pts) < 2:
        return False
    rx, ry = radii_xy(r, aspect)
    for i, p in enumerate(pts):
        if 0 < i < len(pts) - 1:
            x, y = p
            tol_x = max(0.02, rx + 0.01)
            tol_y = max(0.02, ry + 0.01)
            near = (
                abs(x - cushion_axis_value("left", r, aspect)) < tol_x
                or abs(x - cushion_axis_value("right", r, aspect)) < tol_x
                or abs(y - cushion_axis_value("top", r, aspect)) < tol_y
                or abs(y - cushion_axis_value("bottom", r, aspect)) < tol_y
            )
            if not near:
                return False
        else:
            if not _in_table(p, -0.02):
                return False
    for i in range(len(pts) - 1):
        if _dist(pts[i], pts[i + 1]) < 1e-4:
            return False
    return True


def _unit_offset(
    p: tuple[float, float],
    direction: tuple[float, float],
    dist_h: float,
    aspect: float | None = None,
) -> tuple[float, float]:
    """从 p 沿 direction 移动 dist_h（短边单位的物理距离），返回归一化坐标。

    direction 为归一化坐标系里的向量；物理上 x 方向要乘 aspect。
    """
    a = get_aspect(aspect)
    dx, dy = direction
    px, py = dx * a, dy
    L = (px * px + py * py) ** 0.5
    if L < 1e-12:
        return p
    px, py = px / L, py / L
    return (p[0] + (px * dist_h) / a, p[1] + py * dist_h)


def ghost_ball(
    target: tuple[float, float],
    pocket: tuple[float, float],
    r: float,
    aspect: float | None = None,
) -> tuple[float, float]:
    """幽灵球：目标球沿 target→pocket 运动时白球球心触点（物理间距 2R）。"""
    d = (pocket[0] - target[0], pocket[1] - target[1])
    # 从 target 向 pocket 的反方向退 2R
    return _unit_offset(target, (-d[0], -d[1]), 2.0 * r, aspect)


def pot_angle_ok(
    approach_from: tuple[float, float],
    target: tuple[float, float],
    pocket: tuple[float, float],
    aspect: float | None = None,
    min_deg: float | None = None,
) -> bool:
    """目标球心处：「目标→白球来向」与「目标→袋口」夹角 ≥ min_deg（默认100°）。"""
    if min_deg is None:
        min_deg = POT_MIN_ANGLE_DEG
    return _angle_between_deg(approach_from, target, pocket, aspect) >= min_deg - 1e-6


def middle_pocket_rail_angle_deg(
    target: tuple[float, float],
    pocket: tuple[float, float],
    aspect: float | None = None,
) -> float:
    """进球线与长库的夹角（度，0~90）。

    中袋在长库上；线几乎平行长库时角度≈0，无法打进。
    在物理坐标下：长库方向为 x 轴，角度 = asin(|vy|/|v|)。
    """
    import math

    a = get_aspect(aspect)
    vx = (pocket[0] - target[0]) * a
    vy = pocket[1] - target[1]
    L = math.hypot(vx, vy)
    if L < 1e-12:
        return 0.0
    return math.degrees(math.asin(min(1.0, abs(vy) / L)))


def middle_pocket_rail_ok(
    target: tuple[float, float],
    pocket: tuple[float, float],
    aspect: float | None = None,
    min_deg: float | None = None,
) -> bool:
    """中袋：进球线与长库夹角须 ≥ min_deg（默认 15°）。角袋恒通过。"""
    if min_deg is None:
        min_deg = MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG
    if not _is_middle_pocket_pos(pocket):
        return True
    return middle_pocket_rail_angle_deg(target, pocket, aspect) >= min_deg - 1e-6


def _is_corner_pocket(p: tuple[float, float]) -> bool:
    return (p[0] < 0.08 or p[0] > 0.92) and (p[1] < 0.08 or p[1] > 0.92)


def corner_jaw_angles_deg(
    target: tuple[float, float],
    pocket: tuple[float, float],
    aspect: float | None = None,
) -> tuple[float, float]:
    """角袋：进球线与两条袋角库边的夹角（度）。

    返回 (与长库夹角, 与短库夹角)，均在 0~90。
    长库沿 x，短库沿 y（物理坐标）。
    """
    import math

    a = get_aspect(aspect)
    vx = (pocket[0] - target[0]) * a
    vy = pocket[1] - target[1]
    L = math.hypot(vx, vy)
    if L < 1e-12:
        return 0.0, 0.0
    # 与水平长库：asin(|vy|/L)；与竖直短库：asin(|vx|/L)
    ang_long = math.degrees(math.asin(min(1.0, abs(vy) / L)))
    ang_short = math.degrees(math.asin(min(1.0, abs(vx) / L)))
    return ang_long, ang_short


def corner_aim_point(
    target: tuple[float, float],
    pocket: tuple[float, float],
    r: float,
    aspect: float | None = None,
    min_deg: float | None = None,
) -> tuple[float, float]:
    """角袋瞄准点：贴近某条库时，沿另一条库边向台内偏移一个球半径。

    例：
    - 目标在左侧、打左上角(0,0)：贴左库 → 瞄准 (rx, 0)（沿上库右移）
    - 横打右上角(1,0)：贴上库 → 瞄准 (1, ry)（沿右库下移）
    中袋不在此处理。
    """
    if min_deg is None:
        min_deg = CORNER_POCKET_MIN_JAW_ANGLE_DEG
    if not _is_corner_pocket(pocket):
        return pocket

    ang_long, ang_short = corner_jaw_angles_deg(target, pocket, aspect)
    rx, ry = radii_xy(r, aspect)
    off = CORNER_AIM_OFFSET_IN_RADIUS
    # 台内方向
    sx = 1.0 if pocket[0] < 0.5 else -1.0
    sy = 1.0 if pocket[1] < 0.5 else -1.0

    ax, ay = pocket[0], pocket[1]
    # 与短库夹角过小 → 几乎平行短库 → 沿长库偏移
    if ang_short < min_deg <= ang_long:
        ax = pocket[0] + sx * off * rx
    # 与长库夹角过小 → 几乎平行长库 → 沿短库偏移
    elif ang_long < min_deg <= ang_short:
        ay = pocket[1] + sy * off * ry
    # 两边都过小：同时偏一点
    elif ang_long < min_deg and ang_short < min_deg:
        ax = pocket[0] + sx * off * rx * 0.5
        ay = pocket[1] + sy * off * ry * 0.5
    return (ax, ay)


def effective_pot_target(
    target: tuple[float, float],
    pocket: tuple[float, float],
    r: float,
    aspect: float | None = None,
) -> tuple[float, float]:
    """进球几何应瞄准的「袋口点」（角袋可能外偏）。"""
    if _is_corner_pocket(pocket):
        return corner_aim_point(target, pocket, r, aspect)
    return pocket


def _angle_between_deg(
    a: tuple[float, float],
    vertex: tuple[float, float],
    b: tuple[float, float],
    aspect: float | None = None,
) -> float:
    """物理空间中 ∠(a-vertex-b)，单位：度。"""
    import math

    aa = get_aspect(aspect)
    ux = (a[0] - vertex[0]) * aa
    uy = a[1] - vertex[1]
    vx = (b[0] - vertex[0]) * aa
    vy = b[1] - vertex[1]
    lu = math.hypot(ux, uy)
    lv = math.hypot(vx, vy)
    if lu < 1e-12 or lv < 1e-12:
        return 0.0
    cosang = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
    return math.degrees(math.acos(cosang))


def contact_turn_ok(
    white_from: tuple[float, float],
    contact: tuple[float, float],
    object_dir_to: tuple[float, float],
    min_deg: float | None = None,
    aspect: float | None = None,
) -> bool:
    """碰撞点处路径转角 ≥ min_deg。"""
    if min_deg is None:
        min_deg = POT_MIN_ANGLE_DEG
    return _angle_between_deg(white_from, contact, object_dir_to, aspect) >= min_deg - 1e-6


def pot_polyline_angles_ok(
    pts: list[tuple[float, float]],
    ball_vertices: list[tuple[float, float]],
    min_deg: float | None = None,
    aspect: float | None = None,
) -> bool:
    """检查折线在指定球心顶点处的转角是否都 ≥ min_deg。"""
    if min_deg is None:
        min_deg = POT_MIN_ANGLE_DEG
    if len(pts) < 3:
        return True
    for v in ball_vertices:
        idx = None
        best = 1e9
        for i, p in enumerate(pts):
            d = _dist(p, v)
            if d < best:
                best = d
                idx = i
        if idx is None or idx <= 0 or idx >= len(pts) - 1:
            continue
        if best > 0.05:
            continue
        ang = _angle_between_deg(pts[idx - 1], pts[idx], pts[idx + 1], aspect)
        if ang < min_deg - 1e-6:
            return False
    return True


def plan_direct_cue_to_target(
    cue: tuple[float, float],
    target: tuple[float, float],
    r: float,
    aspect: float | None = None,
) -> PathPlan:
    # 触点：沿 cue→target 退 2R（各向异性）
    d = (target[0] - cue[0], target[1] - cue[1])
    impact = _unit_offset(target, (-d[0], -d[1]), 2.0 * r, aspect)
    aim = _norm(d)
    pts = [cue, target]
    return PathPlan(
        kind="direct",
        cushions=[],
        points=pts,
        aim_dir=aim,
        impact_point=impact,
        length=_path_length(pts),
        target_color="",
        target_index=-1,
        notes=["直线解球：白球直接打向目标球心方向"],
    )


def plan_cue_bank(
    cue: tuple[float, float],
    target: tuple[float, float],
    cushions: list[str],
    r: float,
    aspect: float | None = None,
) -> PathPlan | None:
    """白球吃库后撞击目标球（斯诺克解球主路径）。"""
    if not cushions or len(cushions) > MAX_CUSHIONS:
        return None
    virtual_target = reflect_seq(target, cushions, r, aspect)
    pts = fold_path(cue, virtual_target, cushions, r, aspect)
    if not pts or not _segments_valid(pts, CUSHION_MARGIN_RATIO, r=r, aspect=aspect):
        return None
    if _dist(pts[-1], target) > 0.03:
        if _dist(pts[-1], target) > 0.08:
            return None
        pts[-1] = target
    d = _norm((pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    final_dir = (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
    impact = _unit_offset(target, (-final_dir[0], -final_dir[1]), 2.0 * r, aspect)
    notes = [f"白球{'→'.join(cushions)}库后解到目标"]
    return PathPlan(
        kind="cue_bank",
        cushions=list(cushions),
        points=pts,
        aim_dir=d,
        impact_point=impact,
        length=_path_length(pts),
        target_color="",
        target_index=-1,
        notes=notes,
    )


def plan_object_bank(
    cue: tuple[float, float],
    target: tuple[float, float],
    pocket: tuple[float, float],
    cushions: list[str],
    r: float,
    aspect: float | None = None,
) -> PathPlan | None:
    """目标球吃库后入袋。镜面按球心内移 rx/ry。"""
    virtual_pocket = reflect_seq(pocket, cushions, r, aspect)
    obj_pts = fold_path(target, virtual_pocket, cushions, r, aspect)
    if not obj_pts or not _segments_valid(obj_pts, 0.0, r=r, aspect=aspect):
        return None
    if _dist(obj_pts[-1], pocket) < 0.08:
        obj_pts[-1] = pocket
    else:
        return None
    first_dir = (obj_pts[1][0] - obj_pts[0][0], obj_pts[1][1] - obj_pts[0][1])
    g = _unit_offset(target, (-first_dir[0], -first_dir[1]), 2.0 * r, aspect)
    if not pot_angle_ok(g, target, pocket, aspect):
        return None
    d = _norm((g[0] - cue[0], g[1] - cue[1]))
    pts = [cue, g] + obj_pts[1:]
    notes = [f"目标球{'→'.join(cushions)}库后入袋"]
    return PathPlan(
        kind="object_bank",
        cushions=list(cushions),
        points=pts,
        aim_dir=d,
        impact_point=g,
        length=_path_length(pts),
        target_color="",
        target_index=-1,
        notes=notes,
    )


def path_blocked(
    plan: PathPlan,
    cue: tuple[float, float],
    target: tuple[float, float],
    other_balls: list[tuple[float, float]],
    r: float,
    aspect: float | None = None,
) -> tuple[bool, str]:
    """检查路径是否被球挡住（走廊扫过，含目标球规则）。

    斯诺克解球/进球：
    - 白球**吃库过程中**（含库与库之间）不得碰到任何球，**包括目标球**
    - 只有**最后一段**去触碰目标球是允许的
    - 走廊：路径向两侧平移球半径，扫过区域内不得出现障碍球心（半宽 2R）
    """
    pts = plan.points

    def chain_segments(points: list[tuple[float, float]]) -> list[tuple[tuple, tuple]]:
        return [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    t_idx = None
    for i, p in enumerate(pts):
        if _dist(p, target) < 0.5 * r:
            t_idx = i
            break

    # (a, b, allow_target_contact)
    # allow_target_contact=True：该段允许走廊扫到目标球（最后一击）
    segs: list[tuple[tuple, tuple, bool]] = []

    if plan.kind == "direct":
        # 白球 → 幽灵球（触点），允许最终接触目标
        segs.append((cue, plan.impact_point, True))
        if t_idx is not None and t_idx < len(pts) - 1:
            for a, b in chain_segments(pts[t_idx:]):
                segs.append((a, b, True))  # 目标球出球段，目标自身不算障碍
    elif plan.kind == "cue_bank":
        # pts: cue, 库点…, target
        # 白球链：起点 + 各库点 + 幽灵球
        bounces = list(pts[1:-1])
        chain = [pts[0]] + bounces + [plan.impact_point]
        raw = chain_segments(chain)
        for i, (a, b) in enumerate(raw):
            segs.append((a, b, i == len(raw) - 1))
    elif plan.kind == "cue_bank_pot":
        # 白球：库… → 幽灵球；目标：目标 → 袋口
        g = plan.impact_point
        end_i = t_idx if t_idx is not None else max(2, len(pts) - 2)
        chain = list(pts[:max(2, end_i)])
        if not chain or _dist(chain[-1], g) > 1e-6:
            chain = list(chain) + [g]
        raw = chain_segments(chain)
        for i, (a, b) in enumerate(raw):
            segs.append((a, b, i == len(raw) - 1))
        if t_idx is not None:
            for a, b in chain_segments(pts[t_idx:]):
                segs.append((a, b, True))
    elif plan.kind == "object_bank":
        # 白球只走到幽灵球（不穿目标）；目标球整段吃库到袋
        segs.append((cue, plan.impact_point, True))
        obj_start = t_idx if t_idx is not None else 1
        for a, b in chain_segments(pts[obj_start:]):
            segs.append((a, b, True))
    elif plan.kind == "plant":
        # 白球 → gA（不穿 A）；A→gB；B→袋
        segs.append((cue, plan.impact_point, False))
        if len(pts) >= 6:
            segs.append((pts[2], pts[3], True))
            segs.append((pts[4], pts[5], True))
        elif t_idx is not None and t_idx < len(pts) - 1:
            for a, b in chain_segments(pts[t_idx:]):
                segs.append((a, b, True))
    else:
        for a, b in chain_segments(pts):
            segs.append((a, b, True))

    for a, b, allow_t in segs:
        if allow_t:
            # 最终触球段：其它球仍挡；目标球自身不挡
            ob, reason = corridor_hits_any(
                a, b, other_balls, r=r, skip=[target], skip_tol=0.3 * r, aspect=aspect
            )
        else:
            # 吃库/过渡段：目标球也算障碍
            obs = list(other_balls)
            if all(_dist(ob, target) > 0.3 * r for ob in obs):
                obs.append(target)
            ob, reason = corridor_hits_any(
                a, b, obs, r=r, skip=[], aspect=aspect
            )
        if ob is not None:
            return True, reason
    return False, ""


def default_pockets() -> list[tuple[float, float]]:
    """与 detect.build_pockets 一致：角袋=四角，中袋=长边中点。"""
    return [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]


def _inset_pocket(pk: tuple[float, float], eps: float = 0.02) -> tuple[float, float]:
    """展示用袋口在角/边中点；进球计算时略移入台面，避免反射退化。"""
    x = min(1.0 - eps, max(eps, pk[0] if 0.05 < pk[0] < 0.95 else (eps if pk[0] < 0.5 else 1.0 - eps)))
    y = min(1.0 - eps, max(eps, pk[1] if 0.05 < pk[1] < 0.95 else (eps if pk[1] < 0.5 else 1.0 - eps)))
    return (x, y)


def _is_middle_pocket(p: tuple[float, float]) -> bool:
    return abs(p[0] - 0.5) < 0.12 and (p[1] < 0.12 or p[1] > 0.88)


def _pocket_clearance(p: tuple[float, float]) -> float:
    return POCKET_CLEARANCE_MIDDLE if _is_middle_pocket(p) else POCKET_CLEARANCE_CORNER


def _pocket_pass_radius(p: tuple[float, float]) -> float:
    return POCKET_PASS_RADIUS_MIDDLE if _is_middle_pocket(p) else POCKET_PASS_RADIUS_CORNER


def point_dangerous_for_pocket(
    pt: tuple[float, float],
    pockets: list[tuple[float, float]] | None = None,
) -> bool:
    """吃库点是否太靠近袋口（易摔袋/撞袋角）。"""
    for pk in pockets or default_pockets():
        if _dist(pt, pk) < _pocket_clearance(pk):
            return True
    return False


def path_pocket_risk(
    plan: PathPlan,
    pockets: list[tuple[float, float]] | None = None,
) -> tuple[bool, str]:
    """检查路径是否有摔袋/撞袋角风险。

    1) 中间吃库点必须离各袋口足够远
    2) 白球路径线段不得穿过袋口危险圆心区
    3) 目标球终点本身可以靠近袋（解球碰球即可），最后一段触点不检查「靠近」
    """
    pockets = pockets if pockets is not None else default_pockets()
    pts = plan.points
    if not pts or len(pts) < 2:
        return False, ""

    # 中间折点 = 吃库点（首尾为白球/目标）
    bounce_pts = pts[1:-1] if len(pts) >= 3 else []
    for bp in bounce_pts:
        if point_dangerous_for_pocket(bp, pockets):
            return True, f"吃库点过近袋口 @({bp[0]:.3f},{bp[1]:.3f})"

    # 白球行进段（到 impact）
    if plan.kind == "direct":
        segs = [(pts[0], plan.impact_point)]
    elif plan.kind == "cue_bank":
        chain = list(pts[:-1]) + [plan.impact_point]
        segs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    elif plan.kind == "object_bank":
        segs = [(pts[0], plan.impact_point)]
    else:
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    for a, b in segs:
        for pk in pockets:
            if _segment_near_point(a, b, pk, _pocket_pass_radius(pk)):
                return True, f"路径穿过袋口 @({pk[0]:.3f},{pk[1]:.3f})"
    return False, ""


def _segment_near_point(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
    radius: float,
    eps: float = 1e-12,
) -> bool:
    x1, y1 = a
    x2, y2 = b
    cx, cy = p
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < eps:
        return _dist(a, p) < radius
    t = ((cx - x1) * dx + (cy - y1) * dy) / L2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    qx, qy = x1 + t * dx, y1 + t * dy
    return (qx - cx) ** 2 + (qy - cy) ** 2 < radius * radius


def _valid_cushion_sequences(max_n: int) -> list[tuple[str, ...]]:
    """枚举不相邻同库、不立即原路返回的库序列。"""
    seqs: list[tuple[str, ...]] = []
    for n in range(1, max_n + 1):
        for perm in permutations(CUSHIONS, n):
            ok = True
            for i in range(len(perm) - 1):
                if perm[i] == perm[i + 1]:
                    ok = False
                    break
                # 相反库立即返回通常无意义（先 top 再 bottom 在短路径可能有用，保留）
            if ok:
                seqs.append(perm)
    # 一库优先
    seqs.sort(key=lambda s: (len(s), s))
    return seqs


def find_escape_paths(
    cue: tuple[float, float],
    targets: list[tuple[float, float]],
    target_color: str,
    other_balls: list[tuple[float, float]],
    r: float,
    max_cushions: int | None = None,
    pockets: list[tuple[float, float]] | None = None,
    aspect: float | None = None,
) -> list[PathPlan]:
    """为指定目标球搜索解球路径（含直线、一库、多库）。

    aspect: 台面实测宽/高；None 时用本帧 set_table_aspect 或理论值。
    """
    max_n = max_cushions or MAX_CUSHIONS
    plans: list[PathPlan] = []
    asp = get_aspect(aspect)

    def obstacles_for(t: tuple[float, float]) -> list[tuple[float, float]]:
        out = []
        for ob in other_balls:
            if _dist(ob, t) < 0.5 * r:
                continue
            out.append(ob)
        return out

    # 直线
    for ti, t in enumerate(targets):
        p = plan_direct_cue_to_target(cue, t, r, asp)
        p.target_color = target_color
        p.target_index = ti
        blocked, reason = path_blocked(p, cue, t, obstacles_for(t), r, asp)
        if not blocked:
            risky, why = path_pocket_risk(p, pockets)
            if risky:
                blocked, reason = True, why
        p.blocked = blocked
        p.block_reason = reason
        if not blocked:
            plans.append(p)

    # 吃库
    for seq in _valid_cushion_sequences(max_n):
        for ti, t in enumerate(targets):
            p = plan_cue_bank(cue, t, list(seq), r, asp)
            if p is None:
                continue
            p.target_color = target_color
            p.target_index = ti
            blocked, reason = path_blocked(p, cue, t, obstacles_for(t), r, asp)
            if not blocked:
                risky, why = path_pocket_risk(p, pockets)
                if risky:
                    blocked, reason = True, why
            p.blocked = blocked
            p.block_reason = reason
            if not blocked:
                plans.append(p)

    return plans


def rank_score(plan: PathPlan) -> float:
    from config import RANK_CUSHION_WEIGHT, RANK_LENGTH_WEIGHT

    return plan.n_cushions * RANK_CUSHION_WEIGHT + plan.length * RANK_LENGTH_WEIGHT


def _phys_dist(
    a: tuple[float, float], b: tuple[float, float], aspect: float | None = None
) -> float:
    aa = get_aspect(aspect)
    dx = (a[0] - b[0]) * aa
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _is_middle_pocket_pos(p: tuple[float, float]) -> bool:
    return abs(p[0] - 0.5) < 0.15 and (p[1] < 0.08 or p[1] > 0.92)


def score_pot_plan(plan: PathPlan, aspect: float | None = None) -> float:
    """斯诺克进球难度分（越低越好）。

    综合：方式难度、目标离袋、切角是否够直、路径长度、中袋、吃库数。
    """
    t = plan.target_pos
    pk = plan.pocket_pos
    base = float(POT_KIND_BASE.get(plan.kind, 40.0))
    score = base

    if t is not None and pk is not None:
        # 球离袋：短边物理距离
        score += OBJ_TO_POCKET_WEIGHT * _phys_dist(t, pk, aspect)
        # 中袋更难
        if _is_middle_pocket_pos(pk):
            score += POT_MIDDLE_POCKET_PENALTY
        # 切角：目标处 (target→approach) 与 (target→pocket) 夹角，越接近180越易
        # approach 用 impact_point（幽灵球/触球来向）
        if plan.impact_point is not None:
            ang = _angle_between_deg(plan.impact_point, t, pk, aspect)
            # 偏离直线
            score += CUT_ANGLE_PENALTY * max(0.0, 180.0 - ang)
            # 贴近合法阈值的风险
            margin = ang - POT_MIN_ANGLE_DEG
            if margin < 15.0:
                score += ANGLE_MARGIN_PENALTY * max(0.0, 15.0 - margin)
        # 白球路径长度
        score += PATH_LENGTH_WEIGHT * plan.length
    else:
        score += PATH_LENGTH_WEIGHT * plan.length

    # 吃库进球再按库数小幅加罚
    if plan.kind == "cue_bank_pot":
        score += POT_CUSHION_PENALTY * plan.n_cushions

    return score


def score_escape_plan(plan: PathPlan, aspect: float | None = None) -> float:
    """斯诺克解球难度分（越低越好）：库数优先，其次路径短。"""
    score = ESCAPE_CUSHION_PENALTY * plan.n_cushions
    score += PATH_LENGTH_WEIGHT * plan.length
    return score


def score_plan(plan: PathPlan, aspect: float | None = None) -> float:
    if plan.is_pot:
        return score_pot_plan(plan, aspect)
    return score_escape_plan(plan, aspect)


def sort_plans(plans: list[PathPlan], aspect: float | None = None) -> list[PathPlan]:
    """按斯诺克击球逻辑排序并写回 plan.score。"""
    for p in plans:
        p.score = score_plan(p, aspect)
    plans.sort(key=lambda p: (0 if p.is_pot else 1, p.score, p.length))
    return plans


def find_escape_paths_ranked(
    cue: tuple[float, float],
    targets: list[tuple[float, float]],
    target_color: str,
    other_balls: list[tuple[float, float]],
    r: float,
    max_cushions: int | None = None,
    pockets: list[tuple[float, float]] | None = None,
    aspect: float | None = None,
) -> list[PathPlan]:
    plans = find_escape_paths(
        cue, targets, target_color, other_balls, r, max_cushions, pockets=pockets, aspect=aspect
    )
    asp = get_aspect(aspect)
    for p in plans:
        if p.target_index >= 0 and p.target_index < len(targets):
            p.target_pos = targets[p.target_index]
        p.score = score_escape_plan(p, asp)
    plans.sort(key=lambda p: (p.score, p.n_cushions, p.length))
    return plans


def find_pot_paths(
    cue: tuple[float, float],
    targets: list[tuple[float, float]],
    pockets: list[tuple[float, float]],
    target_color: str,
    other_balls: list[tuple[float, float]],
    r: float,
    include_object_bank: bool = True,
    max_cushions: int | None = None,
    aspect: float | None = None,
    other_colors: list[str] | None = None,
) -> list[PathPlan]:
    """进球方案：直接、白球吃库后进球、翻袋、传球(plant)。

    斯诺克传球规则：
    - 仅当「打红球」时允许红球传红球（两颗都是红）
    - 打彩球（黄/绿/棕/蓝/粉/黑）时不允许任何传球：必须先碰该彩球且只进该彩球
    - other_colors 与 other_balls 一一对应；缺省时打红仍允许 plant（保守：假设障碍里可能有红）
    """
    plans: list[PathPlan] = []
    max_n = max_cushions if max_cushions is not None else MAX_CUSHIONS
    all_obj = list(other_balls)
    asp = get_aspect(aspect)
    colors = list(other_colors) if other_colors is not None else [None] * len(all_obj)
    if len(colors) != len(all_obj):
        colors = [None] * len(all_obj)

    # 传球只在打红时搜索；彩球一律不传
    allow_plant = target_color == "red"

    for ti, t in enumerate(targets):
        obstacles = [ob for ob in all_obj if _dist(ob, t) >= 0.5 * r]

        for pi, pk in enumerate(pockets):
            # 中袋：进球线与长库夹角过小则不可进
            rail_ok = middle_pocket_rail_ok(t, pk, asp)
            # 角袋：贴库时瞄准点沿另一库边外偏一个球半径
            pk_aim = effective_pot_target(t, pk, r, asp)

            # ---- 1) 直接进球 ----
            g = ghost_ball(t, pk_aim, r, asp)
            if (
                rail_ok
                and pot_angle_ok(cue, t, pk_aim, asp)
                and contact_turn_ok(cue, g, t, aspect=asp)
                and pot_polyline_angles_ok([cue, g, t, pk_aim], [g, t], aspect=asp)
                and _in_table(g, -0.05)
            ):
                d = _norm((g[0] - cue[0], g[1] - cue[1]))
                pts = [cue, g, t, pk_aim]
                note_bag = f"袋口#{pi}"
                if _dist(pk_aim, pk) > 1e-6:
                    note_bag += "(角点外偏)"
                p = PathPlan(
                    kind="direct",
                    cushions=[],
                    points=pts,
                    aim_dir=d,
                    impact_point=g,
                    length=_path_length(pts),
                    target_color=target_color,
                    target_index=ti,
                    notes=["直接进球", note_bag],
                    target_pos=t,
                    pocket_pos=pk,
                )
                blocked, reason = path_blocked(p, cue, t, obstacles, r, asp)
                p.blocked = blocked
                p.block_reason = reason
                if not blocked:
                    plans.append(p)

            # ---- 2) 白球吃库后进球 ----
            if rail_ok and -0.02 <= g[0] <= 1.02 and -0.02 <= g[1] <= 1.02:
                for seq in _valid_cushion_sequences(max_n):
                    bank = plan_cue_bank(cue, g, list(seq), r, asp)
                    if bank is None:
                        continue
                    if not pot_angle_ok(g, t, pk_aim, asp):
                        continue
                    last_white = bank.points[-2] if len(bank.points) >= 2 else cue
                    if not contact_turn_ok(last_white, g, t, aspect=asp):
                        continue
                    cue_chain = list(bank.points[:-1]) + [g]
                    pts = cue_chain + [t, pk_aim]
                    if not pot_polyline_angles_ok(pts, [g, t], aspect=asp):
                        continue
                    p = PathPlan(
                        kind="cue_bank_pot",
                        cushions=list(seq),
                        points=pts,
                        aim_dir=bank.aim_dir,
                        impact_point=g,
                        length=_path_length(pts),
                        target_color=target_color,
                        target_index=ti,
                        notes=[f"吃库{'→'.join(seq)}后进球", f"袋口#{pi}"],
                        target_pos=t,
                        pocket_pos=pk,
                    )
                    blocked, reason = path_blocked(p, cue, t, obstacles, r, asp)
                    if not blocked:
                        for bp in p.points[1:-2]:
                            if point_dangerous_for_pocket(bp, pockets):
                                blocked, reason = True, "吃库点过近袋口"
                                break
                    p.blocked = blocked
                    p.block_reason = reason
                    if not blocked:
                        plans.append(p)

            # ---- 3) 目标球翻袋 ----
            if include_object_bank and rail_ok:
                for c in CUSHIONS:
                    p = plan_object_bank(cue, t, pk_aim, [c], r, asp)
                    if p is None:
                        continue
                    if len(p.points) >= 3:
                        if not pot_polyline_angles_ok(p.points, [p.impact_point, t], aspect=asp):
                            continue
                    p.target_index = ti
                    p.notes.append(f"袋口#{pi}")
                    p.target_pos = t
                    p.pocket_pos = pk
                    blocked, reason = path_blocked(p, cue, t, obstacles, r, asp)
                    if not blocked and p.cushions:
                        for bp in p.points[1:-1]:
                            if point_dangerous_for_pocket(bp, pockets) and _dist(bp, pk) < _pocket_clearance(pk):
                                blocked, reason = True, "翻袋吃库点过近袋口"
                                break
                    p.blocked = blocked
                    p.block_reason = reason
                    if not blocked:
                        plans.append(p)

            # ---- 4) 传球/组合球：白球→A→B→袋口 ----
            # 斯诺克：仅打红时允许「红传红」；打彩球禁止传球
            if not allow_plant:
                continue
            for B, B_col in zip(all_obj, colors):
                if _dist(B, t) < 1.5 * r:
                    continue
                # 仅红球可以作为第二颗（入袋球）
                if B_col is not None and B_col != "red":
                    continue
                if not middle_pocket_rail_ok(B, pk, asp):
                    continue
                pk_aim_B = effective_pot_target(B, pk, r, asp)
                gB = ghost_ball(B, pk_aim_B, r, asp)
                if not _in_table(gB, -0.05):
                    continue
                if not pot_angle_ok(cue, t, gB, asp):
                    continue
                if not pot_angle_ok(gB, B, pk_aim_B, asp):
                    continue
                gA = ghost_ball(t, gB, r, asp)
                if not _in_table(gA, -0.05):
                    continue
                if not contact_turn_ok(cue, gA, t, aspect=asp):
                    continue
                if not contact_turn_ok(t, gB, B, aspect=asp):
                    continue
                obs = [
                    ob
                    for ob in all_obj
                    if _dist(ob, t) >= 0.5 * r and _dist(ob, B) >= 0.5 * r
                ]
                blocked, reason = False, ""
                for a, b in ((cue, gA), (t, gB), (B, pk_aim_B)):
                    ob, why = corridor_hits_any(a, b, obs, r=r, skip=[], aspect=asp)
                    if ob is not None:
                        blocked, reason = True, why
                        break
                if blocked:
                    continue
                pts = [cue, gA, t, gB, B, pk_aim_B]
                if not pot_polyline_angles_ok(pts, [gA, t, gB, B], aspect=asp):
                    continue
                p = PathPlan(
                    kind="plant",
                    cushions=[],
                    points=pts,
                    aim_dir=_norm((gA[0] - cue[0], gA[1] - cue[1])),
                    impact_point=gA,
                    length=_path_length(pts),
                    target_color=target_color,
                    target_index=ti,
                    notes=["红传红/组合球", f"袋口#{pi}"],
                    # 评分用「入袋球」B 到袋口的距离
                    target_pos=B,
                    pocket_pos=pk,
                )
                p.blocked = False
                plans.append(p)

    # 斯诺克逻辑排序：直接易球优先，其次吃库/传球/翻袋；同型比难度分
    for p in plans:
        p.score = score_pot_plan(p, asp)
    plans.sort(key=lambda p: (p.score, p.n_cushions, p.length))
    return plans
