# -*- coding: utf-8 -*-
"""用引擎自身的 Lab 距离图（resp）裁决绿球真实圆心。

绿球色相与台呢相同，色相掩码看不见它；
但它与台呢的**亮度**差异会体现在 Lab 距离 d 上，
而这正是引擎自己用来找球的特征。

本脚本取引擎 debug 里的 resp 图，对绿球邻域做阈值+圆拟合，
得到独立于「响应加权质心」的圆心估计，与报告值对比。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import detect  # noqa: E402
import detect2 as D2  # noqa: E402


def fit_circle(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    return None if r2 <= 0 else (float(cx), float(cy), float(np.sqrt(r2)))


def circle_from_d(d, cx0, cy0, r_guess):
    """在 (cx0,cy0) 邻域用 d 的阈值化区域做圆拟合。"""
    R = int(r_guess * 1.8)
    x0, y0 = max(0, int(cx0) - R), max(0, int(cy0) - R)
    x1, y1 = min(d.shape[1], int(cx0) + R + 1), min(d.shape[0], int(cy0) + R + 1)
    sub = d[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    # 以邻域内的高分位为球内水平
    inner = float(np.percentile(sub, 85))
    thr = max(12.0, inner * 0.5)
    m = (sub >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 50:
            continue
        gx, gy = cents[i][0] + x0, cents[i][1] + y0
        dd = (gx - cx0) ** 2 + (gy - cy0) ** 2
        if best is None or dd < best[0]:
            best = (dd, gx, gy, a, i)
    if best is None:
        return None
    _dd, gx, gy, a, i = best
    ys, xs = np.nonzero(lab == i)
    pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    f = fit_circle(pts) if len(pts) >= 20 else None
    return dict(c=(gx, gy), area=a, fit=f, thr=thr, inner=inner)


info = None
try:
    import capture
    info = capture.find_window_rect()
except Exception:
    pass

frames = []
if info:
    im = capture.print_window_bgr(info["hwnd"])
    if im is not None:
        frames.append(("实时帧", im))
for nm in ("full_rack.png", "live_v2.png", "raw_0.png"):
    p = ROOT / "samples" / "evidence" / nm
    if p.exists():
        frames.append((nm, cv2.imread(str(p))))

for tag, img in frames:
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    felt = st.felt_rect
    r_rep = float(np.median([b.r for b in st.balls]))
    r_raw = r_rep / config.BALL_RADIUS_FIX_SCALE
    dbg: dict = {}
    st2 = D2.analyze_frame2(img, felt_override=felt, debug=dbg)
    d = dbg["resp"]
    ox, oy = st2.crop_rect[0], st2.crop_rect[1]
    fx, fy = felt[0], felt[1]

    print(f"\n{'='*80}")
    print(f"{tag}  报告r={r_rep:.2f}  引擎原始r={r_raw:.2f}")
    print("=" * 80)
    print(f"  {'颜色':>7} {'报告球心':>17} {'d拟合球心':>17} {'偏移':>6} "
          f"{'d拟合r':>7} {'r/原始':>7}")
    print("  " + "-" * 72)
    rows = []
    for b in st2.balls:
        # st2 球坐标是 crop 系；换算到图像系
        gx, gy = b.x + ox, b.y + oy
        res = circle_from_d(d, b.x, b.y, r_raw)
        if not res or not res["fit"]:
            print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f})  拟合失败")
            continue
        ncx, ncy, nr = res["fit"]
        # 拟合结果是 crop 系
        gncx, gncy = ncx + ox, ncy + oy
        off = float(np.hypot(gncx - gx, gncy - gy))
        rows.append((b.color, off, nr))
        flag = "  <<<" if off > 3 else ""
        print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f}) ({gncx:>7.1f},{gncy:>7.1f}) "
              f"{off:>6.2f} {nr:>7.2f} {nr/r_raw:>7.4f}{flag}")
    if rows:
        offs = np.array([t[1] for t in rows])
        nrs = np.array([t[2] for t in rows])
        print("  " + "-" * 72)
        print(f"  球心偏移: 中位={np.median(offs):.2f}  max={offs.max():.2f}")
        print(f"  d拟合半径/引擎原始: 中位={np.median(nrs)/r_raw:.4f}")
