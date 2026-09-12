# -*- coding: utf-8 -*-
"""球台与球的视觉检测"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    BALL_CENTER_BIAS_X,
    BALL_CENTER_BIAS_Y,
    BALL_RADIUS_PX,
    BALL_RADIUS_RATIO,
    BALL_RADIUS_SCALE,
    COLOR_RANGES,
    FELT_CLOTH_HSV,
    FELT_CLOTH_MODE,
    FELT_INSET_RATIO,
    FELT_SHADOW_MAX_AREA_RATIO,
    FELT_SHADOW_MAX_V,
    FELT_SHADOW_MIN_AREA_RATIO,
    TABLE_ASPECT,
)

# 本帧识别到的台泥颜色名
_CLOTH_NAME = "green"


@dataclass
class Ball:
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
    """归一化到 [0,1] 台面坐标系的球局。原点左上，x 向右，y 向下。"""

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
        # 默认：优先红球，否则所有非白非黑
        reds = [b for b in self.balls if b.color == "red"]
        if reds:
            return reds
        return [b for b in self.balls if b.color not in ("cue", "black")]


def _mask_hsv(hsv: np.ndarray, color: str) -> np.ndarray:
    spec = COLOR_RANGES[color]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h0, h1) in spec["h"]:
        s0, s1 = spec["s"][0]
        v0, v1 = spec["v"][0]
        lower = np.array([h0, s0, v0], dtype=np.uint8)
        upper = np.array([h1, s1, v1], dtype=np.uint8)
        # 跨 0 的红
        if h0 > h1:
            continue
        part = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, part)
    # 单独处理跨 0 红色
    if color == "red":
        for h0, h1 in spec["h"]:
            if h0 > 100:  # 170-179
                lower = np.array([h0, spec["s"][0][0], spec["v"][0][0]], np.uint8)
                upper = np.array([h1, spec["s"][0][1], spec["v"][0][1]], np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            else:
                lower = np.array([h0, spec["s"][0][0], spec["v"][0][0]], np.uint8)
                upper = np.array([h1, spec["s"][0][1], spec["v"][0][1]], np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def get_cloth_name() -> str:
    return _CLOTH_NAME


def _cloth_masks(hsv: np.ndarray, name: str) -> np.ndarray:
    (h0, s0, v0), (h1, s1, v1) = FELT_CLOTH_HSV[name]
    m = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    if name == "red":
        m2 = cv2.inRange(hsv, (168, s0, v0), (179, s1, v1))
        m = cv2.bitwise_or(m, m2)
    return m


def pick_cloth_mask(hsv: np.ndarray) -> tuple[str, np.ndarray]:
    """选台泥主色掩码。FELT_CLOTH_MODE=auto 时按「大矩形台面」评分。"""
    global _CLOTH_NAME
    mode = (FELT_CLOTH_MODE or "auto").lower()
    names = list(FELT_CLOTH_HSV.keys())
    if mode in FELT_CLOTH_HSV:
        _CLOTH_NAME = mode
        return mode, _cloth_masks(hsv, mode)

    img_area = float(hsv.shape[0] * hsv.shape[1])
    best_name, best_mask, best_score = "green", None, -1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for name in names:
        mask = _cloth_masks(hsv, name)
        ratio = float(mask.mean()) / 255.0
        if ratio < 0.06:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        def sc(c):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if w < 40 or h < 20:
                return -1.0
            if w * h < img_area * 0.10:
                return -1.0
            aspect = w / max(h, 1)
            if not (1.3 <= aspect <= 2.9):
                return -1.0
            return area / (1.0 + 2.0 * abs(np.log(aspect / TABLE_ASPECT)))

        cbest = max(contours, key=sc)
        s = sc(cbest)
        # 略偏好绿（默认皮肤）
        if name == "green":
            s *= 1.05
        if s > best_score:
            best_score, best_name, best_mask = s, name, mask

    if best_mask is None or best_score < 0:
        _CLOTH_NAME = "green"
        return "green", _cloth_masks(hsv, "green")
    _CLOTH_NAME = best_name
    return best_name, best_mask


def detect_felt(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """检测台面内沿矩形 (x,y,w,h)，支持多色台泥皮肤。

    结构（由外向内）：
      同色库边 → 黑阴影一圈 → 台泥（台面）
    真实边界 = 库与黑影交界；台内**包含**黑阴影及其内部台泥。
    """
    global _CLOTH_NAME
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w

    cloth_name, mask = pick_cloth_mask(hsv)
    cloth_ratio = float(mask.mean()) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cloth_ratio < 0.06:
        return None

    def score_cloth(c):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        return area / (1.0 + 2.0 * abs(np.log(aspect / TABLE_ASPECT)))

    best = max(contours, key=score_cloth)
    ox, oy, ow, oh = cv2.boundingRect(best)
    if ow * oh < img_area * 0.12:
        return None
    o_aspect = ow / max(oh, 1)
    if not (1.4 <= o_aspect <= 2.8):
        return None

    outer_cloth = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(outer_cloth, [best], -1, 255, -1)

    # 暗/阴影：整图低亮度，并限制在台面附近
    dark = (V < FELT_SHADOW_MAX_V).astype(np.uint8) * 255
    near_table = cv2.dilate(outer_cloth, kernel, iterations=4)
    dark = cv2.bitwise_and(dark, near_table)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    shadow = np.zeros(dark.shape, np.uint8)
    min_a = img_area * FELT_SHADOW_MIN_AREA_RATIO
    max_a = img_area * FELT_SHADOW_MAX_AREA_RATIO
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_a <= a <= max_a:
            shadow[labels == i] = 255

    felt = None
    if int(shadow.sum()) > 0:
        sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shadow_c = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, sk, iterations=2)
        s_contours, _ = cv2.findContours(shadow_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if s_contours:
            sc = max(s_contours, key=cv2.contourArea)
            sx, sy, sw, sh = cv2.boundingRect(sc)
            if sw >= ow * 0.45 and sh >= oh * 0.45 and sw <= ow * 1.05 and sh <= oh * 1.05:
                sa = sw / max(sh, 1)
                if 1.3 <= sa <= 2.9:
                    felt = (sx, sy, sw, sh)

    if felt is None:
        inset = max(4, int(min(ow, oh) * max(FELT_INSET_RATIO, 0.02)))
        if ow - 2 * inset < 40 or oh - 2 * inset < 20:
            return None
        felt = (ox + inset, oy + inset, ow - 2 * inset, oh - 2 * inset)

    fx, fy, fw, fh = felt
    inset = max(2, int(min(fw, fh) * FELT_INSET_RATIO))
    if fw - 2 * inset < 40 or fh - 2 * inset < 20:
        inset = 1
    fx, fy, fw, fh = fx + inset, fy + inset, fw - 2 * inset, fh - 2 * inset

    if fw * fh < img_area * 0.10:
        return None
    if not (1.35 <= fw / max(fh, 1) <= 2.8):
        return None
    roi = mask[fy : fy + fh, fx : fx + fw]
    roi_d = dark[fy : fy + fh, fx : fx + fw]
    if roi.size == 0:
        return None
    cloth_or_dark = float(np.maximum(roi, roi_d).mean()) / 255.0
    if cloth_or_dark < 0.50:
        return None
    return fx, fy, fw, fh


def _classify_pixel(hsv_patch: np.ndarray) -> tuple[str, float]:
    """用色块中心区域 HSV 均值做颜色分类。"""
    if hsv_patch.size == 0:
        return "unknown", 0.0
    # 取中心 50% 像素
    h, w = hsv_patch.shape[:2]
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    patch = hsv_patch[y0:y1, x0:x1].reshape(-1, 3)
    if len(patch) == 0:
        patch = hsv_patch.reshape(-1, 3)
    mean = patch.mean(axis=0)
    H, S, V = mean
    # 白：低饱和高亮
    if S < 55 and V > 170:
        return "cue", 0.95
    # 黑：低亮
    if V < 55:
        return "black", 0.9
    # 粉：偏红但中低饱和高亮
    if (150 <= H <= 179 or 0 <= H <= 10) and 35 <= S <= 150 and V > 150:
        return "pink", 0.75
    # 红
    if (H <= 12 or H >= 168) and S >= 90 and V >= 70:
        return "red", 0.9
    # 黄
    if 16 <= H <= 40 and S >= 100 and V >= 100:
        return "yellow", 0.9
    # 绿（与台呢区分：球更亮/更饱和且有高光）
    if 40 <= H <= 90 and S >= 70 and V >= 90:
        # 球心高光区更亮
        return "green", 0.7
    # 蓝
    if 90 <= H <= 135 and S >= 70 and V >= 60:
        return "blue", 0.9
    # 棕
    if 5 <= H <= 25 and S >= 80 and 30 <= V <= 130:
        return "brown", 0.85
    return "unknown", 0.2


def detect_balls_in_felt(img_bgr: np.ndarray, felt: tuple[int, int, int, int]) -> list[Ball]:
    """以连通域为主、Hough 为辅检测球（更适配俯视 2D 游戏画面）。"""
    fx, fy, fw, fh = felt
    if fw <= 0 or fh <= 0:
        return []
    roi = img_bgr[fy : fy + fh, fx : fx + fw]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    min_r = max(4, int(fh * 0.012))
    max_r = max(min_r + 3, int(fh * 0.040))

    balls = _blob_balls(roi, hsv, min_r, max_r)

    # Hough 仅作少量补充（参数更严，避免 UI 噪声）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2.0,
        param1=80,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None:
        for c in np.uint16(np.around(circles))[0]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            if x < r or y < r or x + r >= fw or y + r >= fh:
                continue
            # 若已有邻近球则跳过
            near = False
            for b in balls:
                if (b.x - x) ** 2 + (b.y - y) ** 2 < (min_r * 1.5) ** 2:
                    near = True
                    break
            if near:
                continue
            pad = max(1, r // 3)
            y0, y1 = max(0, y - r - pad), min(fh, y + r + pad)
            x0, x1 = max(0, x - r - pad), min(fw, x + r + pad)
            patch = hsv[y0:y1, x0:x1]
            local = np.zeros(patch.shape[:2], np.uint8)
            cv2.circle(local, (x - x0, y - y0), max(1, r - 1), 255, -1)
            mask = local > 0
            if mask.sum() < 8:
                continue
            px = patch[mask]
            H = float(np.median(px[:, 0]))
            S = float(np.median(px[:, 1]))
            V = float(np.median(px[:, 2]))
            highlight = float((px[:, 2] > 200).mean())
            color, score = _classify_median(H, S, V, highlight)
            balls.append(Ball(x=float(x), y=float(y), r=float(r), color=color, score=score * 0.9))

    balls = _nms_balls(balls, min_dist=min_r * 1.35)
    balls = _filter_snooker_unique(balls)
    balls = normalize_ball_radii(balls, fh)
    return balls


def normalize_ball_radii(balls: list[Ball], felt_h: int) -> list[Ball]:
    """统一球半径：去掉异常检测尺寸后取稳健均值，赋给每一颗球。

    可消除误检/漏检带来的大小不一，使碰撞半径与标注更一致。
    """
    if len(balls) < 3 or felt_h <= 0:
        return balls

    rs = np.array([b.r for b in balls], dtype=np.float64)
    med = float(np.median(rs))
    if med <= 0:
        return balls

    # 用绝对中位差（MAD）去异常
    mad = float(np.median(np.abs(rs - med)))
    # 若几乎无离散，用相对阈值
    if mad < 0.5:
        lo, hi = med * 0.70, med * 1.40
    else:
        # 1.5 * 1.4826 * MAD ≈ 稳健 2σ；再放宽一点
        lo, hi = med - 3.5 * mad, med + 3.5 * mad
        lo = max(lo, med * 0.55)
        hi = min(hi, med * 1.80)

    keep_mask = (rs >= lo) & (rs <= hi)
    if keep_mask.sum() < max(3, len(balls) // 3):
        # 过滤过严则退回中位数
        uniform = med
        kept = balls
    else:
        uniform = float(rs[keep_mask].mean())
        kept = [b for b, k in zip(balls, keep_mask) if k]
        # 极端离群且颜色 unknown 的已在别处过滤；这里只丢弃尺寸离谱的
        dropped = [b for b, k in zip(balls, keep_mask) if not k]
        # 尺寸异常但颜色高置信的彩球保留位置、只统一半径
        for b in dropped:
            if b.score >= 0.8 and b.color in (
                "cue", "red", "yellow", "green", "brown", "blue", "pink", "black"
            ):
                kept.append(b)

    # 与理论比例交叉校正（防止整体偏大/偏小）
    theory = felt_h * 0.0148
    if theory > 0 and BALL_RADIUS_PX <= 0:
        # 若稳健均值偏离理论过多，向理论靠拢一点
        if uniform > theory * 1.8 or uniform < theory * 0.50:
            uniform = 0.5 * uniform + 0.5 * theory

    # 用户缩放 / 强制像素半径
    if BALL_RADIUS_PX > 0:
        uniform = float(BALL_RADIUS_PX)
    else:
        uniform = uniform * float(BALL_RADIUS_SCALE)

    for b in kept:
        b.r = float(uniform)
    return kept


def _near_pocket(cx: float, cy: float, fw: int, fh: int, r: float) -> bool:
    """袋口暗洞易被误检为黑/棕球。"""
    if fw <= 0 or fh <= 0:
        return False
    nx, ny = cx / fw, cy / fh
    # 归一化下球半径
    rr = max(r / fw, r / fh)
    for px, py in ((0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)):
        if (nx - px) ** 2 + (ny - py) ** 2 < (1.8 * rr + 0.015) ** 2:
            return True
    return False


def _refine_dark_ball_center(
    hsv_roi: np.ndarray, cx: float, cy: float, r: float
) -> tuple[float, float]:
    """黑球球心：取最暗核的距离变换中心，避免阴影/袋口拖偏。"""
    fh, fw = hsv_roi.shape[:2]
    rad = max(2.0, float(r) * 1.8)
    y0, y1 = max(0, int(cy - rad)), min(fh, int(cy + rad + 1))
    x0, x1 = max(0, int(cx - rad)), min(fw, int(cx + rad + 1))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        return cx, cy
    sub = hsv_roi[y0:y1, x0:x1]
    V = sub[:, :, 2]
    # 仅最暗核（球体），阴影略亮会被滤掉
    core = (V < 38).astype(np.uint8)
    if int(core.sum()) < 8:
        core = (V < 48).astype(np.uint8)
    local = np.zeros(core.shape, np.uint8)
    cv2.circle(
        local,
        (int(round(cx)) - x0, int(round(cy)) - y0),
        max(2, int(r * 0.95)),
        255,
        -1,
    )
    core = cv2.bitwise_and(core, local)
    if int(core.sum()) < 6:
        return cx, cy
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    _, maxv, _, maxloc = cv2.minMaxLoc(dist)
    if maxv < 1.0:
        return cx, cy
    ys, xs = np.nonzero(core)
    w = dist[ys, xs]
    if float(w.sum()) < 1e-6:
        return float(x0 + maxloc[0]), float(y0 + maxloc[1])
    nx = float(np.average(xs, weights=w)) + x0
    ny = float(np.average(ys, weights=w)) + y0
    return nx, ny


def _refine_ball_center(
    hsv_roi: np.ndarray, cx: float, cy: float, r: float, color: str = ""
) -> tuple[float, float]:
    """球心校正。黑球走暗核距离变换；其余用较亮像素质心。"""
    if color == "black":
        return _refine_dark_ball_center(hsv_roi, cx, cy, r)
    h, w = hsv_roi.shape[:2]
    y0, y1 = max(0, int(cy - r - 2)), min(h, int(cy + r + 3))
    x0, x1 = max(0, int(cx - r - 2)), min(w, int(cx + r + 3))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        return cx, cy
    patch = hsv_roi[y0:y1, x0:x1]
    local = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(local, (int(round(cx)) - x0, int(round(cy)) - y0), max(1, int(r)), 255, -1)
    mask = local > 0
    if int(mask.sum()) < 10:
        return cx, cy
    ys, xs = np.nonzero(mask)
    px = patch[mask]
    v = px[:, 2].astype(np.float64)
    v_med = float(np.median(v))
    # 只保留明显比阴影亮的部分（阴影偏暗且在右下）
    keep = v >= max(18.0, v_med * 0.70)
    if int(keep.sum()) < 10:
        keep = v >= max(12.0, v_med * 0.55)
    if int(keep.sum()) < 8:
        return cx, cy
    wx = xs[keep].astype(np.float64)
    wy = ys[keep].astype(np.float64)
    ww = v[keep]
    nx = x0 + float(np.average(wx, weights=ww))
    ny = y0 + float(np.average(wy, weights=ww))
    return nx + BALL_CENTER_BIAS_X, ny + BALL_CENTER_BIAS_Y


def _blob_balls(roi, hsv, min_r, max_r) -> list[Ball]:
    """按颜色掩码找近似圆形连通域。"""
    fh, fw = roi.shape[:2]
    balls: list[Ball] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 台呢参考色：取全图 HSV 中位数
    med = np.median(hsv.reshape(-1, 3), axis=0)

    candidates = [
        "cue",
        "red",
        "yellow",
        "green",
        "brown",
        "blue",
        "pink",
        "black",
    ]
    for color in candidates:
        mask = _color_mask(hsv, color)
        # 与台呢接近的绿色球：额外要求更亮/更饱和且接近圆形
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < max(12, min_r * min_r * 1.2):
                continue
            if area > (max_r * 2) ** 2:
                continue
            if h < min_r * 0.8 or w < min_r * 0.8:
                continue
            aspect = w / max(h, 1)
            if not (0.70 <= aspect <= 1.43):
                continue
            # 填充率：面积 / 外接矩形
            fill = area / max(w * h, 1)
            if fill < 0.55:
                continue
            # 圆度：面积 / (π r^2) with r = (w+h)/4
            r = (w + h) / 4.0
            if r < min_r * 0.65 or r > max_r * 1.25:
                continue
            circularity = area / (np.pi * r * r + 1e-6)
            if circularity < 0.62:
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            # 用圆内像素复核颜色
            pad = 1
            y0, y1 = max(0, int(cy) - int(r) - pad), min(fh, int(cy) + int(r) + pad)
            x0, x1 = max(0, int(cx) - int(r) - pad), min(fw, int(cx) + int(r) + pad)
            patch = hsv[y0:y1, x0:x1]
            local = np.zeros(patch.shape[:2], np.uint8)
            cv2.circle(local, (int(cx) - x0, int(cy) - y0), max(1, int(r) - 1), 255, -1)
            m = local > 0
            if m.sum() < 8:
                continue
            px = patch[m]
            H = float(np.median(px[:, 0]))
            S = float(np.median(px[:, 1]))
            V = float(np.median(px[:, 2]))
            highlight = float((px[:, 2] > 200).mean())
            # 绿球与台呢/球杆/Logo 区分：必须很圆，且与台呢有明显差异或有高光
            if color == "green":
                if circularity < 0.78 or not (0.78 <= aspect <= 1.28):
                    continue
                dS = abs(S - med[1])
                dV = abs(V - med[2])
                if highlight < 0.04 and dS < 40 and dV < 35:
                    continue
                # 真实球体有明暗渐变；纯色 Logo 块 V 标准差很小
                v_std = float(np.std(px[:, 2]))
                if v_std < 8 and highlight < 0.05:
                    continue
            if color == "black":
                if circularity < 0.68:
                    continue
            # 黑球：排除贴库阴影环（沿边暗条易被当成球）
            if color == "black":
                m = max(r * 1.3, min_r * 1.2)
                if cx < m or cy < m or cx > fw - m or cy > fh - m:
                    continue
                if circularity < 0.70:
                    continue
            # 袋口黑洞：黑/棕误检
            if color in ("black", "brown") and _near_pocket(cx, cy, fw, fh, r):
                continue

            clf, score = _classify_median(H, S, V, highlight)
            # 掩码颜色与复核不一致时，以复核为准，但保留高圆度候选
            if clf == "unknown":
                clf = color
                score = 0.55
            # 阴影补偿：用球体较亮部分重估球心
            cx, cy = _refine_ball_center(hsv, cx, cy, r, color=clf)
            balls.append(Ball(x=cx, y=cy, r=float(r), color=clf, score=score))

    balls = _filter_size_consistent(balls, min_r, max_r)
    # 贴库边/角落的噪声：球心距库边应至少约 0.6 半径
    edge_keep = []
    for b in balls:
        if b.x < b.r * 0.55 or b.y < b.r * 0.55 or b.x > fw - b.r * 0.55 or b.y > fh - b.r * 0.55:
            # 允许贴库但不能完全在角落外
            if b.x < b.r * 0.3 or b.y < b.r * 0.3 or b.x > fw - b.r * 0.3 or b.y > fh - b.r * 0.3:
                continue
        edge_keep.append(b)
    balls = _filter_snooker_unique(edge_keep)
    return balls


def _filter_snooker_unique(balls: list[Ball]) -> list[Ball]:
    """斯诺克规则：彩球各最多 1 颗，白球 1 颗，红球最多 15 颗。同色保留最高分。"""
    by_color: dict[str, list[Ball]] = {}
    for b in balls:
        by_color.setdefault(b.color, []).append(b)
    kept: list[Ball] = []
    for color, group in by_color.items():
        group = sorted(group, key=lambda x: -x.score)
        if color == "red":
            # 红球可多颗，但过滤明显边角噪声
            for b in group[:15]:
                kept.append(b)
        elif color == "unknown":
            continue
        else:
            kept.append(group[0])
    return kept


def _filter_size_consistent(balls: list[Ball], min_r: float, max_r: float) -> list[Ball]:
    """球半径应彼此接近；去掉明显偏大/偏小的误检。"""
    if len(balls) < 3:
        return balls
    rs = np.array([b.r for b in balls], dtype=np.float32)
    med = float(np.median(rs))
    if med < 1:
        return balls
    kept = []
    for b in balls:
        # 半径在 [0.55, 1.55] 倍中位数内
        if 0.55 * med <= b.r <= 1.55 * med:
            kept.append(b)
    return kept


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if color == "cue":
        return ((S < 55) & (V > 170)).astype(np.uint8) * 255
    if color == "red":
        return (((H <= 12) | (H >= 165)) & (S >= 90) & (V >= 55)).astype(np.uint8) * 255
    if color == "yellow":
        return ((H >= 16) & (H <= 42) & (S >= 100) & (V >= 90)).astype(np.uint8) * 255
    if color == "green":
        # 腾讯桌球绿球偏青：高饱和 H~70-100；排除 Logo 黄绿
        return ((H >= 70) & (H <= 105) & (S >= 150) & (V >= 100)).astype(np.uint8) * 255
    if color == "brown":
        return ((H >= 5) & (H <= 28) & (S >= 55) & (V >= 25) & (V <= 150)).astype(np.uint8) * 255
    if color == "blue":
        return ((H >= 90) & (H <= 140) & (S >= 60) & (V >= 45)).astype(np.uint8) * 255
    if color == "pink":
        return (
            (((H >= 150) | (H <= 12)) & (S >= 30) & (S <= 155) & (V >= 150))
        ).astype(np.uint8) * 255
    if color == "black":
        return ((V < 55)).astype(np.uint8) * 255
    return np.zeros(hsv.shape[:2], np.uint8)


def _classify_median(H: float, S: float, V: float, highlight: float) -> tuple[str, float]:
    if S < 50 and V > 165:
        return "cue", 0.95 + highlight * 0.05
    if V < 60 and S < 100:
        return "black", 0.9
    # 粉：偏红/品红但中低饱和、偏亮
    if (145 <= H <= 179 or H <= 12) and 25 <= S <= 150 and V > 150:
        return "pink", 0.8
    # 棕：暗橙棕，V 中低
    if 5 <= H <= 28 and S >= 50 and 35 <= V <= 145:
        # 与红区分：红通常更亮更饱和
        if V <= 120 or S <= 180:
            return "brown", 0.85
    if (H <= 14 or H >= 165) and S >= 90 and V >= 70:
        return "red", 0.92
    if 16 <= H <= 42 and S >= 100 and V >= 100:
        return "yellow", 0.9
    if 40 <= H <= 95 and S >= 60 and V >= 85:
        # 真球有高光，均匀色块多半是 Logo/UI
        if S >= 160 and highlight >= 0.02:
            return "green", 0.75 + highlight * 0.2
        return "green", 0.5
    if 90 <= H <= 140 and S >= 60 and V >= 50:
        return "blue", 0.9
    if 5 <= H <= 28 and S >= 70 and 30 <= V <= 140:
        return "brown", 0.8
    return "unknown", 0.2


def _fallback_blob_balls(roi, hsv, min_r, max_r) -> list[Ball]:
    """Hough 失败时用颜色 blob 兜底。"""
    balls: list[Ball] = []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 非台呢：用亮度/饱和度排除大面积绿
    hsv_f = hsv.astype(np.float32)
    # 候选：高光圆斑
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 不太可靠，改用按颜色掩码
    for color in COLOR_RANGES:
        mask = _mask_hsv(hsv, color)
        if color == "green":
            # 绿球：仅保留高亮且圆
            bright = (hsv[:, :, 2] > 140).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, bright)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < (min_r * min_r * 2) or area > (max_r * max_r * 5):
                continue
            if h < min_r or w < min_r:
                continue
            aspect = w / max(h, 1)
            if not 0.55 <= aspect <= 1.8:
                continue
            cx, cy = centroids[i]
            r = (w + h) / 4
            if r < min_r * 0.7 or r > max_r * 1.3:
                continue
            balls.append(Ball(x=float(cx), y=float(cy), r=float(r), color=color, score=0.6))
    return _nms_balls(balls, min_dist=min_r * 1.4)


def _nms_balls(balls: list[Ball], min_dist: float) -> list[Ball]:
    balls = sorted(balls, key=lambda b: -b.score)
    kept: list[Ball] = []
    for b in balls:
        ok = True
        for k in kept:
            if (b.x - k.x) ** 2 + (b.y - k.y) ** 2 < min_dist**2:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


def build_pockets(felt: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """归一化袋口：角袋=矩形四角，中袋=长边中点。"""
    return [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]


def analyze_frame(img_bgr: np.ndarray) -> TableState:
    felt = detect_felt(img_bgr)
    h, w = img_bgr.shape[:2]
    if felt is None:
        # 标记失败：felt_rect 全 0
        return TableState(width=w, height=h, felt_rect=(0, 0, 0, 0), balls=[], pockets=[], error="未检测到台呢，请确认截图为俯视球台")
    balls = detect_balls_in_felt(img_bgr, felt)
    # 合理球数过滤：斯诺克最多 22 颗，检测过多视为误检
    if len(balls) > 30:
        balls = sorted(balls, key=lambda b: -b.score)[:30]
    state = TableState(
        width=w,
        height=h,
        felt_rect=felt,
        balls=balls,
        pockets=build_pockets(felt),
    )
    return state


def draw_detection_preview(img_bgr: np.ndarray, state: TableState) -> np.ndarray:
    out = img_bgr.copy()
    if not state.ok():
        cv2.putText(out, state.error or "no table", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    fx, fy, fw, fh = state.felt_rect
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
    for b in state.balls:
        px, py = state.to_felt_px(b.x / 1.0, b.y / 1.0) if False else (fx + b.x, fy + b.y)
        # balls 存的是 felt-roi 像素坐标
        px, py = fx + b.x, fy + b.y
        color = {
            "cue": (255, 255, 255),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 200, 0),
            "brown": (0, 80, 160),
            "blue": (255, 80, 0),
            "pink": (255, 120, 220),
            "black": (40, 40, 40),
        }.get(b.color, (180, 180, 180))
        cv2.circle(out, (int(px), int(py)), int(b.r), color, 2)
        cv2.putText(
            out,
            b.color,
            (int(px) + int(b.r), int(py) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
