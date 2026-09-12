# -*- coding: utf-8 -*-
"""分析管线：图像 → 球局 → 解球方案"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import BALL_RADIUS_PX, BALL_RADIUS_RATIO, POCKET_DRAW_RADIUS_RATIO, TARGET_COLORS
from detect import Ball, TableState, analyze_frame, draw_detection_preview, get_cloth_name
from geometry import (
    PathPlan,
    find_escape_paths_ranked,
    find_pot_paths,
    rank_score,
    set_table_aspect,
    get_aspect,
    sort_plans,
)


@dataclass
class AnalysisResult:
    state: TableState
    plans: list[PathPlan]
    cue: tuple[float, float] | None
    ball_radius_norm: float
    mode: str
    target_color: str
    message: str = ""
    felt_aspect: float = 0.0

    def best(self) -> PathPlan | None:
        return self.plans[0] if self.plans else None


def _ball_radius_norm(state: TableState) -> float:
    """优先用已统一的球半径；支持 config 强制像素半径。"""
    _, _, fw, fh = state.felt_rect
    if BALL_RADIUS_PX > 0 and fh > 0:
        return BALL_RADIUS_PX / max(fh, 1)
    rs = [b.r for b in state.balls]
    if rs and fh > 0:
        med = float(np.median(rs))
        return max(BALL_RADIUS_RATIO * 0.5, min(BALL_RADIUS_RATIO * 2.5, med / max(fh, 1)))
    return BALL_RADIUS_RATIO


def _to_norm_balls(state: TableState) -> list[tuple[Ball, tuple[float, float]]]:
    _, _, fw, fh = state.felt_rect
    out = []
    for b in state.balls:
        out.append((b, (b.x / fw, b.y / fh)))
    return out


def analyze(
    img_bgr: np.ndarray,
    target_color: str | None = None,
    mode: str = "escape",
    max_plans: int = 8,
    max_cushions: int | None = None,
) -> AnalysisResult:
    """mode: escape(解球碰目标) | pot(进球) | both
    target_color: None 则自动选（优先红）
    """
    state = analyze_frame(img_bgr)
    if not state.ok():
        set_table_aspect(None)
        return AnalysisResult(
            state, [], None, BALL_RADIUS_RATIO, mode, target_color or "?",
            state.error or "未检测到台球桌台面",
        )

    # 本帧台面实测长宽比，供几何各向异性使用
    _, _, fw, fh = state.felt_rect
    felt_asp = (fw / fh) if fh > 0 else None
    set_table_aspect(felt_asp)
    asp = get_aspect(felt_asp)

    r = _ball_radius_norm(state)
    cue_b = state.cue_ball()
    if cue_b is None:
        return AnalysisResult(state, [], None, r, mode, target_color or "?", "未检测到白球", asp)

    cue = (cue_b.x / state.felt_rect[2], cue_b.y / state.felt_rect[3])
    if target_color is None:
        # 自动：有红选红
        reds = [b for b in state.balls if b.color == "red"]
        target_color = "red" if reds else next(
            (c for c in TARGET_COLORS if any(b.color == c for b in state.balls)), "red"
        )

    targets_b = [b for b in state.balls if b.color == target_color]
    if not targets_b:
        # 找不到指定色，退回任意非白球
        targets_b = [b for b in state.balls if b.color not in ("cue",)]
        if targets_b:
            target_color = targets_b[0].color
    if not targets_b:
        return AnalysisResult(state, [], cue, r, mode, target_color, "未检测到目标球", asp)

    target_pts = [(b.x / state.felt_rect[2], b.y / state.felt_rect[3]) for b in targets_b]
    # 障碍 = 全部非白球（含同色目标）；打某颗时由 geometry 剔除该颗自身
    others = [
        (b.x / state.felt_rect[2], b.y / state.felt_rect[3])
        for b in state.balls
        if b.color != "cue"
    ]
    # 与 others 对齐的颜色（用于斯诺克传球合法性）
    other_colors = [b.color for b in state.balls if b.color != "cue"]

    pot_plans: list[PathPlan] = []
    escape_plans: list[PathPlan] = []

    if mode in ("pot", "both"):
        pot_plans = find_pot_paths(
            cue,
            target_pts,
            state.pockets,
            target_color,
            others,
            r,
            include_object_bank=True,
            aspect=asp,
            other_colors=other_colors,
        )
    if mode in ("escape", "both", "escape_only"):
        escape_plans = find_escape_paths_ranked(
            cue,
            target_pts,
            target_color,
            others,
            r,
            max_cushions=max_cushions,
            pockets=state.pockets,
            aspect=asp,
        )

    # 优先：进球线路在前，其次解球线路；组内按斯诺克难度分
    if mode == "pot":
        plans = sort_plans(pot_plans, asp)
    elif mode == "escape":
        plans = sort_plans(escape_plans, asp)
    else:
        plans = sort_plans(pot_plans, asp) + sort_plans(escape_plans, asp)

    plans = plans[:max_plans]
    best = plans[0] if plans else None
    best_s = f"{best.score:.1f}" if best is not None else "-"
    msg = (
        f"目标={target_color} 检测球={len(state.balls)} "
        f"台泥={get_cloth_name()} "
        f"可行={len(plans)}(进球{len(pot_plans)}/解球{len(escape_plans)}) "
        f"aspect={asp:.3f} 首选分={best_s}"
    )
    return AnalysisResult(state, plans, cue, r, mode, target_color, msg, asp)


# ---------- 绘制 ----------

_COLOR_MAP = {
    "cue": (255, 255, 255),
    "red": (40, 40, 230),
    "yellow": (0, 220, 255),
    "green": (40, 180, 40),
    "brown": (30, 70, 140),
    "blue": (220, 120, 30),
    "pink": (200, 120, 255),
    "black": (30, 30, 30),
    "unknown": (160, 160, 160),
}


def draw_result(img_bgr: np.ndarray, result: AnalysisResult, top_k: int = 3) -> np.ndarray:
    out = draw_detection_preview(img_bgr, result.state)
    fx, fy, fw, fh = result.state.felt_rect

    def to_img(p):
        return (int(fx + p[0] * fw), int(fy + p[1] * fh))

    # 口袋（按台面短边比例，直径约为球的 1.8 倍）
    pocket_r = max(10, int(fh * POCKET_DRAW_RADIUS_RATIO))
    for p in result.state.pockets:
        cv2.circle(out, to_img(p), pocket_r, (0, 140, 255), 2)

    if not result.plans:
        cv2.putText(
            out,
            result.message or "No plan",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    colors = [(0, 230, 255), (255, 180, 0), (180, 0, 255)]
    for i, plan in enumerate(result.plans[:top_k]):
        col = colors[i % len(colors)]
        pts = [to_img(p) for p in plan.points]
        for a, b in zip(pts, pts[1:]):
            thickness = 3 if i == 0 else 2
            cv2.line(out, a, b, col, thickness, cv2.LINE_AA)
        # 库点
        if len(plan.cushions) > 0:
            for pt in pts[1:-1]:
                cv2.circle(out, pt, 7, (0, 255, 0), 2)
        # 幽灵球 / 触点
        imp = to_img(plan.impact_point)
        rr = max(6, int(fh * result.ball_radius_norm))
        cv2.circle(out, imp, rr, (255, 0, 255), 2)
        # 标注
        label = f"#{i+1} {plan.n_cushions}C {plan.kind}"
        cv2.putText(
            out,
            label,
            (pts[0][0] + 10, pts[0][1] - 10 - i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
            cv2.LINE_AA,
        )

    # 顶部信息
    info = f"{result.message}  r={result.ball_radius_norm:.4f}"
    cv2.putText(out, info, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # 图例
    y0 = fh + fy - 80
    for i, plan in enumerate(result.plans[:top_k]):
        col = colors[i % len(colors)]
        text = f"#{i+1} {plan.kind} sc={plan.score:.0f} len={plan.length:.2f}"
        cv2.putText(out, text, (20, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return out


def summarize(result: AnalysisResult, max_show: int = 5) -> str:
    lines = [result.message, ""]
    if not result.plans:
        lines.append("未找到可行解球线路。可尝试：")
        lines.append("1) 检查台面区域标定是否准确")
        lines.append("2) 降低遮挡阈值或增加最大库数")
        lines.append("3) 确认白球/目标球颜色识别正确")
        return "\n".join(lines)

    lines.append(f"按斯诺克难度排序的方案（目标: {result.target_color}，分越低越好）：")
    for i, p in enumerate(result.plans[:max_show]):
        cush = "直接" if not p.cushions else f"{p.n_cushions}库({'>'.join(p.cushions)})"
        tag = "进球" if p.is_pot else "解球"
        lines.append(
            f"  #{i+1} [{tag}/{p.kind}] {cush}  长={p.length:.3f}  "
            f"分={p.score:.1f}  杆向=({p.aim_dir[0]:.3f},{p.aim_dir[1]:.3f})"
        )
        if p.notes:
            lines.append(f"      {'; '.join(p.notes)}")
    lines.append("")
    lines.append("说明：折点为吃库点（球心路径）；品红圆为触球点。进球优先于解球。")
    return "\n".join(lines)
