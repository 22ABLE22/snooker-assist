# -*- coding: utf-8 -*-
"""游戏自带瞄准延长线检测（借鉴 billiard-assistant，适配 PC 桌面截图）。

在台面 ROI 内找「白线」像素 → PCA 拟合方向 → 从白球心延长至库边。
不依赖 minicap/模板。
"""

from __future__ import annotations

import cv2
import numpy as np


def white_guide_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """近白、高亮像素（游戏辅助线/瞄准线）。"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((V >= 180) & (S <= 95)) | ((V >= 150) & (H >= 90) & (H <= 135) & (S <= 80))
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    return mask


def fit_line_pca(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    """对 2D 点做 PCA，返回 (dx, dy) 单位方向；失败 None。"""
    if pts is None or len(pts) < 10:
        return None
    data = pts.astype(np.float64)
    mean = data.mean(axis=0)
    data0 = data - mean
    # 2x2 协方差
    cov = np.cov(data0, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    dx, dy = float(v[0]), float(v[1])
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-9:
        return None
    return dx / n, dy / n


def pick_longest_component(mask: np.ndarray, min_area: int = 40) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    best_i, best_s = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ext = max(w, h)
        thick = max(1, min(w, h))
        elong = ext / thick
        s = elong * min(ext, 500) + area * 0.005
        if s > best_s:
            best_s, best_i = s, i
    if best_i < 1:
        return None
    return (labels == best_i).astype(np.uint8)


def ray_to_table(p0x: float, p0y: float, dx: float, dy: float, fw: float, fh: float):
    """从点沿 (dx,dy) 到台面 [0,fw]x[0,fh] 边界的交点。"""
    best_t = None
    best_p = None
    eps = 1e-9
    if abs(dx) > eps:
        t = (fw - p0x) / dx if dx > 0 else (0.0 - p0x) / dx
        if t > 0:
            y = p0y + t * dy
            if -1e-6 <= y <= fh + 1e-6:
                if best_t is None or t < best_t:
                    best_t, best_p = t, (p0x + t * dx, p0y + t * dy)
    if abs(dy) > eps:
        t = (fh - p0y) / dy if dy > 0 else (0.0 - p0y) / dy
        if t > 0:
            x = p0x + t * dx
            if -1e-6 <= x <= fw + 1e-6:
                if best_t is None or t < best_t:
                    best_t, best_p = t, (p0x + t * dx, p0y + t * dy)
    return best_p


def detect_aim_line(
    img_bgr: np.ndarray,
    felt: tuple[int, int, int, int],
    cue: tuple[float, float] | None = None,
    margin: float = 0.04,
) -> list[tuple[float, float]]:
    """返回台面归一化 ROI 坐标下的延长线 [(x1,y1),(x2,y2)]，无则空表。"""
    fx, fy, fw, fh = [int(v) for v in felt]
    if fw < 80 or fh < 40:
        return []
    roi = img_bgr[fy : fy + fh, fx : fx + fw]
    if roi.size == 0:
        return []

    # 内缩，去掉库边/袋口噪声
    mx, my = int(fw * margin), int(fh * margin)
    inner = roi[my : fh - my, mx : fw - mx]
    mask = white_guide_mask(inner)
    comp = pick_longest_component(mask, min_area=max(30, int(fw * fh * 0.00002)))
    if comp is None:
        return []

    ys, xs = np.nonzero(comp)
    pts = np.stack([xs + mx, ys + my], axis=1).astype(np.float64)
    dirn = fit_line_pca(pts)
    if dirn is None:
        return []
    dx, dy = dirn

    # 起点：白线点集中离白球最近的一端
    if cue is not None:
        # cue 为窗口像素 → 换算到 felt ROI
        cx, cy = cue
    else:
        # 用白线质心的一端
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    # 选投影后靠近 cue 的一端作为线的“近端”
    proj = pts @ np.array([dx, dy])
    # 近端：投影最小或最大，取更靠近 cue 的那个
    i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
    p0, p1 = pts[i0], pts[i1]
    # 从近端沿方向指向远端
    if (p1[0] - p0[0]) * dx + (p1[1] - p0[1]) * dy < 0:
        p0, p1 = p1, p0
        dx, dy = -dx, -dy

    # 若 cue 在附近，用 cue 作起点
    d0 = (p0[0] - cx) ** 2 + (p0[1] - cy) ** 2
    d1 = (p1[0] - cx) ** 2 + (p1[1] - cy) ** 2
    if d1 < d0:
        # 反转
        dx, dy = -dx, -dy
        start = p1
    else:
        start = p0

    hit = ray_to_table(float(start[0]), float(start[1]), dx, dy, float(fw), float(fh))
    if hit is None:
        # 直接用方向延长
        L = max(fw, fh)
        hit = (start[0] + dx * L, start[1] + dy * L)

    return [(float(start[0]), float(start[1])), (float(hit[0]), float(hit[1]))]


def detect_aim_line_window(
    img_bgr: np.ndarray,
    felt: tuple[int, int, int, int],
    cue: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """同上，但返回窗口/截图像素坐标。"""
    fx, fy, fw, fh = [int(v) for v in felt]
    pts = detect_aim_line(img_bgr, felt, cue=cue)
    out = []
    for x, y in pts:
        out.append((fx + x, fy + y))
    return out
