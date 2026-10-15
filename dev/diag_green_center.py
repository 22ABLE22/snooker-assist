# -*- coding: utf-8 -*-
"""诊断绿球真实圆心：用「整球」阈值（含暗边缘）做圆拟合。

先前用 inner*0.5 只圈到球的亮核，圆心被高光拉偏。
改为低阈值，把球的**整个渲染区（含暗边缘）**都纳入，再拟合圆。
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


def fit_on_d(d, cx0, cy0, r_guess, thr):
    R = int(r_guess * 1.7)
    x0, y0 = max(0, int(cx0) - R), max(0, int(cy0) - R)
    x1, y1 = min(d.shape[1], int(cx0) + R + 1), min(d.shape[0], int(cy0) + R + 1)
    sub = d[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    m = (sub >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 60:
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
    return dict(c=(gx, gy), area=a, fit=f)


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
for nm in ("full_rack.png", "live_v2.png"):
    p = ROOT / "samples" / "evidence" / nm
    if p.exists():
        frames.append((nm, cv2.imread(str(p))))

print("=" * 84)
print("绿球圆心：不同阈值下用「整球」区域做圆拟合")
print("=" * 84)

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
    print(f"\n--- {tag}  报告r={r_rep:.2f}  引擎原始r={r_raw:.2f} ---")
    print(f"  {'颜色':>7} {'报告球心':>17} " +
          " ".join(f"{'thr='+str(t):>16}" for t in (12, 20, 30)))
    print("  " + "-" * 78)
    for b in st2.balls:
        gx, gy = b.x + ox, b.y + oy
        cells = []
        for thr in (12, 20, 30):
            res = fit_on_d(d, b.x, b.y, r_raw, float(thr))
            if res and res["fit"]:
                ncx, ncy, nr = res["fit"]
                gncx, gncy = ncx + ox, ncy + oy
                off = float(np.hypot(gncx - gx, gncy - gy))
                cells.append(f"{off:>6.1f}/{nr:>6.1f}")
            else:
                cells.append(f"{'失败':>16}")
        print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f}) " + " ".join(f"{c:>16}" for c in cells))
    print("  （每格为「偏移px / 拟合半径px」）")
