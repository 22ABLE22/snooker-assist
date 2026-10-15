# -*- coding: utf-8 -*-
"""台面与球的视觉检测 —— 对外接口层（引擎：detect2.py）

本文件保留旧 `detect.py` 的**全部公开接口**，内部改用新的检测引擎
（`detect2.py`）。这样 `analyze.py` / `overlay.py` / `main.py` 无需改动。

为什么换引擎
------------
旧实现对 8 种颜色**分别做 HSV 阈值切分**，病根：
  * 阈值与台呢皮肤、光照强耦合，换皮肤即崩；
  * 同色球直接消失（绿球在绿台呢上、红球在红台呢上）；
  * 球贴球时连通域合并，一堆红球只能算出一颗。

新引擎改用与颜色无关的**物理证据**判球：
  * 明暗渐变（球体必有高光，台呢平坦）；
  * 盘内色差覆盖率（球是实心圆盘，库边只切过一侧）。
详见 detect2.py 顶部说明与 DETECT2_NOTES.md。

坐标系约定（务必与 geometry 保持一致）
--------------------------------------
`TableState.felt_rect` = **可放球的比赛区**（由实测袋心反推），不是绿色台呢外框。
  * 它的四角就是四个角袋中心；
  * `Ball.x/y` 相对 `felt_rect` 原点（因此 `felt_x + ball.x` 得图像坐标，
    `ball.x / felt_w` 得归一化坐标）；
  * 归一化后，球心恒落在 [r_norm, 1-r_norm] 内，贴库球恰好等于 r_norm，
    与 geometry 的库边镜面 `y = r` 一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import detect2 as _d2
from config import BALL_RADIUS_PX, POCKET_DRAW_RADIUS_RATIO

# 本帧识别到的台泥颜色名（旧接口需要；新引擎目前按绿色台呢定位）
_CLOTH_NAME = "green"


# ---------------------------------------------------------------- 数据类
@dataclass
class Ball:
    """与旧接口完全一致的球对象。

    x, y: 相对 TableState.felt_rect 原点的像素坐标
    r:    统一球半径（像素，经距离变换标定）
    score: 置信度 0~1（旧字段名，现由 detect2 的 conf 映射而来）
    """

    x: float
    y: float
    r: float
    color: str
    score: float = 1.0

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class TableState:
    """归一化到 [0,1] 比赛区坐标系的球局。原点左上，x 向右，y 向下。"""

    width: int
    height: int
    felt_rect: tuple[int, int, int, int]  # x,y,w,h in image px
    balls: list[Ball] = field(default_factory=list)
    pockets: list[tuple[float, float]] = field(default_factory=list)
    error: str = ""

    def ok(self) -> bool:
        _, _, w, h = self.felt_rect
        return w > 0 and h > 0 and not self.error

    @property
    def aspect(self) -> float:
        _, _, w, h = self.felt_rect
        return w / max(h, 1)

    def felt_px(self) -> tuple[int, int]:
        _, _, w, h = self.felt_rect
        return w, h

    def to_felt_px(self, nx: float, ny: float) -> tuple[float, float]:
        fx, fy, fw, fh = self.felt_rect
        return fx + nx * fw, fy + ny * fh

    def to_norm(self, ix: float, iy: float) -> tuple[float, float]:
        fx, fy, fw, fh = self.felt_rect
        return (ix - fx) / fw, (iy - fy) / fh

    def cue_ball(self) -> Ball | None:
        cues = [b for b in self.balls if b.color == "cue"]
        if not cues:
            return None
        return max(cues, key=lambda b: b.score)

    def targets(self, color: str | None = None) -> list[Ball]:
        if color is None or color == "red":
            reds = [b for b in self.balls if b.color == "red"]
            if reds:
                return reds
        if color:
            return [b for b in self.balls if b.color == color]
        reds = [b for b in self.balls if b.color == "red"]
        if reds:
            return reds
        return [b for b in self.balls if b.color not in ("cue", "black")]


def get_cloth_name() -> str:
    return _CLOTH_NAME


# ---------------------------------------------------------------- 台面定位
def detect_felt(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """检测**可放球的比赛区**矩形（图像坐标）。

    返回 play_rect（由实测袋心反推），而非绿色台呢外框 ——
    这样 overlay 的 F7 校准、analyze 的几何归一化用的是同一个框，
    避免「校准一个框、计算用另一个框」的错位。
    """
    global _CLOTH_NAME
    try:
        felt = _d2.find_felt(img_bgr)
    except Exception:
        return None
    if felt is None:
        return None
    fx, fy, fw, fh = felt
    inset = 3
    crop = img_bgr[fy + inset : fy + fh - inset, fx + inset : fx + fw - inset]
    if crop.size == 0:
        return None
    try:
        pockets = _d2.detect_pocket_centers(crop)
        pr = _d2.play_rect_from_pockets(pockets, crop.shape[1], crop.shape[0])
    except Exception:
        return None
    if pr == (0.0, 0.0, float(crop.shape[1]), float(crop.shape[0])):
        # 袋口检测失败，退回台呢框内缩（保守估计）
        pad = max(4, int(min(fw, fh) * 0.03))
        return (fx + pad, fy + pad, fw - 2 * pad, fh - 2 * pad)
    x = int(round(pr[0])) + fx + inset
    y = int(round(pr[1])) + fy + inset
    w = int(round(pr[2]))
    h = int(round(pr[3]))
    if w < 40 or h < 20:
        return None
    _CLOTH_NAME = "green"
    return (x, y, w, h)


def build_pockets(felt: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """归一化袋口：角袋=矩形四角，中袋=长边中点（与 geometry.default_pockets 一致）。"""
    return [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]


# ---------------------------------------------------------------- 主检测
def analyze_frame(
    img_bgr: np.ndarray,
    felt_override: tuple[int, int, int, int] | None = None,
) -> TableState:
    """检测球局。felt_override: 已锁定的比赛区矩形 (x,y,w,h)，锁定后不再自动测台面。"""
    h, w = img_bgr.shape[:2]
    debug: dict = {}

    if felt_override is not None:
        # 锁定模式：直接用调用方给的矩形作为坐标系
        st2 = _d2.analyze_frame2(img_bgr, felt_override=felt_override, debug=debug)
        felt_img = tuple(int(v) for v in felt_override)
    else:
        felt_img = detect_felt(img_bgr)
        if felt_img is None:
            return TableState(
                width=w, height=h, felt_rect=(0, 0, 0, 0), balls=[], pockets=[],
                error="未检测到台呢，请确认截图为俯视球台或手动校准台面",
            )
        st2 = _d2.analyze_frame2(img_bgr, felt_override=felt_img, debug=debug)

    if not st2.ok():
        return TableState(
            width=w, height=h, felt_rect=felt_img, balls=[], pockets=[],
            error=st2.error or "未检测到台面",
        )

    # detect2 的球坐标在 crop 内；先转图像坐标，再转 felt_rect 相对坐标
    cx0, cy0 = st2.crop_rect[0], st2.crop_rect[1]
    fx, fy, fw, fh = felt_img
    balls: list[Ball] = []
    for b in st2.balls:
        ix, iy = b.x + cx0, b.y + cy0
        balls.append(
            Ball(
                x=float(ix - fx),
                y=float(iy - fy),
                r=float(b.r),
                color=b.color,
                score=float(b.conf),
            )
        )

    # 半径：支持 config.BALL_RADIUS_PX 强制覆盖
    # 注意 BALL_RADIUS_SCALE / BALL_CENTER_BIAS_* 是旧引擎的补偿项，新引擎已不需要
    # （半径由距离变换标定到真值，球心由加权质心+限幅精修）。
    if BALL_RADIUS_PX > 0 and balls:
        for b in balls:
            b.r = float(BALL_RADIUS_PX)

    # 夹进球心可及范围，满足 geometry 的镜面前提（见函数说明）
    _clamp_balls_into_play_area(balls, felt_img)

    pockets = _pockets_from_debug(debug, felt_img)

    return TableState(
        width=w, height=h, felt_rect=felt_img, balls=balls, pockets=pockets,
    )


def _clamp_balls_into_play_area(balls: list[Ball], felt: tuple[int, int, int, int]) -> None:
    """把球心夹进「球心可及范围」，保证满足 geometry 的镜面前提。

    geometry 的库边镜面在 y = r 处，隐含前提是：
        球心归一化坐标恒落在 [r_norm, 1 - r_norm] 内，
        贴库球恰好等于 r_norm。
    实测由袋心反推的比赛区在短边方向偏大约 3px（袋心略在台呢内沿外侧），
    于是贴库球会「越界」约 3px。这在旧实现里同样存在且更严重（6.5px）。

    直接内缩 felt_rect 会连带把贴库球裁出检测范围（实测丢球），
    所以改为**只夹球心**：检测范围不变，仅把输出坐标夹进合法区间。
    夹取量 ≤3px，相对台面短边 0.5%，对瞄准角的影响 < 0.3°。
    """
    if not balls:
        return
    _, _, fw, fh = felt
    r = float(np.median([b.r for b in balls]))
    if r <= 0:
        return
    lo_x, hi_x = r, fw - r
    lo_y, hi_y = r, fh - r
    for b in balls:
        if hi_x > lo_x:
            b.x = float(min(max(b.x, lo_x), hi_x))
        if hi_y > lo_y:
            b.y = float(min(max(b.y, lo_y), hi_y))


def _pockets_from_debug(
    debug: dict, felt_img: tuple[int, int, int, int]
) -> list[tuple[float, float]]:
    """把实测袋心转成 felt_rect 归一化坐标。

    只保留角袋（中袋归一化后必然落在 [0,1] 之外，因为中袋袋心在库线上、
    而 felt_rect 的四角才是角袋中心）。中袋按长边中点补上，
    与 geometry.default_pockets() 完全一致 —— geometry 正是这样定义袋口的。
    """
    pr = debug.get("play_rect")
    pockets = debug.get("pockets")
    if not pr or not pockets:
        return build_pockets(felt_img)
    px, py, pw, ph = pr
    if pw <= 0 or ph <= 0:
        return build_pockets(felt_img)

    # 角袋：取四个最靠近 play_rect 角的实测袋心，夹到 [0,1]
    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    norm = [((cx - px) / pw, (cy - py) / ph) for cx, cy in pockets]
    out: list[tuple[float, float]] = []
    for tx, ty in corners:
        bx, by = min(norm, key=lambda n: (n[0] - tx) ** 2 + (n[1] - ty) ** 2)
        out.append((float(min(1.0, max(0.0, bx))), float(min(1.0, max(0.0, by)))))
    # 中袋：长边中点（geometry 的约定）
    out.extend([(0.5, 0.0), (0.5, 1.0)])
    return out


# ---------------------------------------------------------------- 绘制
_DRAW = {
    "cue": (255, 255, 255),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
    "brown": (0, 80, 160),
    "blue": (255, 80, 0),
    "pink": (255, 120, 220),
    "black": (40, 40, 40),
    "unknown": (180, 180, 180),
}


def draw_detection_preview(img_bgr: np.ndarray, state: TableState) -> np.ndarray:
    """在图上画出比赛区框、球与袋口（与旧接口一致）。"""
    out = img_bgr.copy()
    if not state.ok():
        cv2.putText(
            out, state.error or "no table", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return out
    fx, fy, fw, fh = state.felt_rect
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

    # 袋口
    pr = max(8, int(fh * POCKET_DRAW_RADIUS_RATIO))
    for nx, ny in state.pockets:
        px, py = int(fx + nx * fw), int(fy + ny * fh)
        cv2.circle(out, (px, py), pr, (0, 140, 255), 2, lineType=cv2.LINE_AA)

    for b in state.balls:
        px, py = int(fx + b.x), int(fy + b.y)
        col = _DRAW.get(b.color, (180, 180, 180))
        cv2.circle(out, (px, py), int(round(b.r)), col, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(
            out, b.color, (px + int(b.r), py - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ------------------------------------------------- 兼容旧版可能用到的名字
def detect_balls_in_felt(img_bgr: np.ndarray, felt: tuple[int, int, int, int]) -> list[Ball]:
    """兼容入口：按给定矩形检测球（坐标为矩形相对坐标）。"""
    st = analyze_frame(img_bgr, felt_override=felt)
    return st.balls


def normalize_ball_radii(balls: list[Ball], felt_h: int) -> list[Ball]:
    """兼容入口：新引擎已输出统一半径，此处仅做一次稳健统一。"""
    if len(balls) < 3:
        return balls
    med = float(np.median([b.r for b in balls]))
    if med > 0:
        for b in balls:
            b.r = med
    return balls
