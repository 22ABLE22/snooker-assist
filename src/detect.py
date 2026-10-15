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

    注意：袋心必须在**整图**上搜（detect_pocket_centers_region）。
    袋口位于库区、其中心通常落在台呢外框之外，若只在裁剪图内部搜，
    袋洞会被窗口边界切掉一部分，质心被系统性拉向台面内侧
    （实测偏内约 23px，叠加坐标系差共 33px ≈ 1.7 个球半径），
    而 geometry 完全依赖袋口坐标决定瞄准方向。
    """
    global _CLOTH_NAME
    try:
        felt = _d2.find_felt(img_bgr)
    except Exception:
        return None
    if felt is None:
        return None
    fx, fy, fw, fh = felt
    if fw < 120 or fh < 80:
        return None
    try:
        pockets_img = _d2.detect_pocket_centers_region(img_bgr, (fx, fy, fw, fh))
        # 袋心是图像坐标；转成以台呢框为原点的局部坐标再反推比赛区
        local = [(px - fx, py - fy) for px, py in pockets_img]
        pr = _d2.play_rect_from_pockets(local, fw, fh)
    except Exception:
        return None
    if pr == (0.0, 0.0, float(fw), float(fh)):
        # 袋口检测失败，退回台呢框内缩（保守估计）
        pad = max(4, int(min(fw, fh) * 0.03))
        return (fx + pad, fy + pad, fw - 2 * pad, fh - 2 * pad)
    x = int(round(pr[0])) + fx
    y = int(round(pr[1])) + fy
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


def analyze_frames(
    frames: list,
    felt_override: tuple[int, int, int, int] | None = None,
    *,
    min_votes: int | None = None,
    tol: float = 8.0,
) -> TableState:
    """多帧融合识别 —— 时间换鲁棒性。

    为什么需要它（现有架构的结构性弱点）
    ------------------------------------
    `analyze_frame` 是**开环**的：一次截图 → 一次判断。于是
      * 单帧噪声无法抑制（球心抖 ~0.5px，没有第二次观测去平掉）；
      * 瞬时干扰无法排除（球杆扫过、瞄准圈、特效只在个别帧出现，
        实测白球旁的灰色瞄准圈曾被判成粉球）。
    时间预算 8s、单帧约 1.1s，有充足余量做多次观测。

    做法
    ----
      1. 第 1 帧完整识别，取得球半径；
      2. 后续帧复用该半径（半径是渲染属性、跨帧恒定），跳过半径扫描，
         单帧耗时从 ~1.1s 降到 ~0.26s；
      3. 多帧结果按位置聚类投票：≥ min_votes 帧出现的才算球（干掉瞬时伪检）；
         球心按置信度加权平均（噪声按 √N 衰减）。

    参数
    ----
    frames: 图像列表（需保证同一球局，即球未移动）。
    min_votes: 至少几帧命中；默认 ceil(0.6 × 帧数)，至少 2。
    tol: 判定同一颗球的位置容差（px）。球局静止，可取较紧。
    """
    from fusion import fuse_detections  # 同目录

    imgs = [f for f in frames if f is not None and getattr(f, "size", 0)]
    if not imgs:
        return TableState(width=0, height=0, felt_rect=(0, 0, 0, 0), error="没有可用帧")

    h, w = imgs[0].shape[:2]
    if len(imgs) == 1:
        return analyze_frame(imgs[0], felt_override=felt_override)

    # 台面：各帧一致，用第 1 帧定下来即可
    if felt_override is not None:
        felt_img = tuple(int(v) for v in felt_override)
    else:
        felt_img = detect_felt(imgs[0])
        if felt_img is None:
            return TableState(
                width=w, height=h, felt_rect=(0, 0, 0, 0), balls=[], pockets=[],
                error="未检测到台呢",
            )

    per_frame: list[list[dict]] = []
    pockets_norm: list[tuple[float, float]] | None = None
    r_cache: float | None = None
    fx, fy = felt_img[0], felt_img[1]

    for i, img in enumerate(imgs):
        debug: dict = {}
        st2 = _d2.analyze_frame2(
            img, felt_override=felt_img, exp_r=r_cache, debug=debug
        )
        if not st2.ok():
            per_frame.append([])
            continue
        cx0, cy0 = st2.crop_rect[0], st2.crop_rect[1]
        dets = []
        for b in st2.balls:
            ix, iy = b.x + cx0, b.y + cy0
            dets.append(
                {
                    "x": float(ix - fx),
                    "y": float(iy - fy),
                    "r": float(b.r),
                    "color": b.color,
                    "conf": float(b.conf),
                }
            )
        per_frame.append(dets)
        if r_cache is None and dets:
            r_cache = float(np.median([d["r"] for d in dets]))
        if pockets_norm is None:
            pockets_norm = _pockets_from_debug(debug, felt_img)

    fused = fuse_detections(per_frame, min_votes=min_votes, tol=tol)
    balls = [
        Ball(
            x=float(d["x"]), y=float(d["y"]), r=float(d["r"]),
            color=d["color"], score=float(d["conf"]),
        )
        for d in fused
    ]

    if BALL_RADIUS_PX > 0 and balls:
        for b in balls:
            b.r = float(BALL_RADIUS_PX)
    _clamp_balls_into_play_area(balls, felt_img)

    return TableState(
        width=w, height=h, felt_rect=felt_img, balls=balls,
        pockets=pockets_norm or build_pockets(felt_img),
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
    """把实测袋心转成 felt_rect 归一化坐标（6 个袋全部用实测值）。

    角袋：实测袋心恰好落在 felt_rect 四角附近，归一化后夹到 [0,1]。
    中袋：实测袋心在长边中点、且**略微落在 felt_rect 之外**
          （袋口在库线上，而 felt_rect 是球心可及范围）。
          实测 y ≈ -0.028 / 1.028，若硬写成 0 / 1 会引入约 19px 偏差
          （≈1 个球半径），足以让瞄准方向明显偏斜。
          故这里保留实测值，允许略微超出 [0,1]。
    """
    pr = debug.get("play_rect")
    pockets = debug.get("pockets")
    if not pr or not pockets:
        return build_pockets(felt_img)
    px, py, pw, ph = pr
    if pw <= 0 or ph <= 0:
        return build_pockets(felt_img)

    norm = [((cx - px) / pw, (cy - py) / ph) for cx, cy in pockets]

    # 角袋：取四个最靠近四角的实测袋心，夹到 [0,1]
    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    corner_pts: list[tuple[float, float]] = []
    used: set[int] = set()
    for tx, ty in corners:
        order = sorted(range(len(norm)),
                       key=lambda i: (norm[i][0] - tx) ** 2 + (norm[i][1] - ty) ** 2)
        j = next((i for i in order if i not in used), order[0])
        used.add(j)
        corner_pts.append((float(min(1.0, max(0.0, norm[j][0]))),
                           float(min(1.0, max(0.0, norm[j][1])))))

    # 中袋：取剩余袋心中「最靠近长边中点」的两个，保留实测 y（允许超出 [0,1]）
    rest = [norm[i] for i in range(len(norm)) if i not in used]
    mids: list[tuple[float, float]] = []
    for ty in (0.0, 1.0):
        if not rest:
            mids.append((0.5, ty))
            continue
        j = min(range(len(rest)), key=lambda i: (rest[i][0] - 0.5) ** 2 + (rest[i][1] - ty) ** 2)
        mx, my = rest.pop(j)
        # 中袋 x 理论上在长边中点；y 保留实测（关键：不要硬写成 0/1）
        mids.append((float(min(0.9, max(0.1, mx))), float(my)))

    # 返回顺序与 build_pockets / geometry.default_pockets 一致：
    #   左上 中上 右上 左下 中下 右下
    return [
        corner_pts[0], mids[0], corner_pts[1],
        corner_pts[2], mids[1], corner_pts[3],
    ]


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
