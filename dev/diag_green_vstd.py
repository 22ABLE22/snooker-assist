# -*- coding: utf-8 -*-
"""用亮度纹理（Vstd）独立测绿球真实圆心 —— 检验「绿球歪了」。

绿球的色相与台呢相同，颜色判据看不见它；
但它有**高光与明暗渐变**，局部亮度标准差（Vstd）远高于台呢
（此前实测：真球 Vstd 34~58，台呢 ≈0.4）。

本脚本用 Vstd 做独立定位，与引擎报告的绿球球心对比，
量化偏移量与方向。
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


def vstd_map(img, r):
    """局部亮度标准差图（与引擎 brightness_std_map 同思路的独立实现）。"""
    V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    k = max(3, int(r * 0.9))
    mu = cv2.blur(V, (k, k))
    mu2 = cv2.blur(V * V, (k, k))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))


def fit_circle(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    return None if r2 <= 0 else (float(cx), float(cy), float(np.sqrt(r2)))


def center_by_vstd(vs, cx0, cy0, rmax, frac=0.5):
    """在给定点附近，用「Vstd 超过阈值」的区域的质心/圆拟合求圆心。"""
    R = int(rmax * 1.6)
    x0, y0 = max(0, int(cx0) - R), max(0, int(cy0) - R)
    x1, y1 = min(vs.shape[1], int(cx0) + R), min(vs.shape[0], int(cy0) + R)
    sub = vs[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    thr = float(np.percentile(sub, 90)) * frac
    m = (sub >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    # 取与 (cx0,cy0) 最近、且面积合理的连通域
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 30:
            continue
        gx = cents[i][0] + x0
        gy = cents[i][1] + y0
        d = (gx - cx0) ** 2 + (gy - cy0) ** 2
        if best is None or d < best[0]:
            best = (d, gx, gy, a, i)
    if best is None:
        return None
    _d, gx, gy, a, i = best
    # 圆拟合
    ys, xs = np.nonzero(lab == i)
    pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    f = fit_circle(pts) if len(pts) >= 20 else None
    return dict(cx=gx, cy=gy, area=a, fit=f, thr=thr)


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
    fx, fy = st.felt_rect[0], st.felt_rect[1]
    r_rep = float(np.median([b.r for b in st.balls]))
    r_raw = r_rep / config.BALL_RADIUS_FIX_SCALE
    vs = vstd_map(img, r_raw)
    print(f"\n{'='*78}")
    print(f"{tag}  报告r={r_rep:.2f}")
    print("=" * 78)
    print(f"  {'颜色':>7} {'报告球心':>17} {'Vstd球心':>17} {'偏移':>7} {'Vstd值':>7}")
    print("  " + "-" * 60)
    for b in st.balls:
        gx, gy = b.x + fx, b.y + fy
        res = center_by_vstd(vs, gx, gy, r_raw)
        if not res:
            print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f})  无法定位")
            continue
        vc = vs[int(gy), int(gx)]
        off = float(np.hypot(res["cx"] - gx, res["cy"] - gy))
        flag = "  <<< 偏移大" if off > 3 else ""
        print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f}) ({res['cx']:>7.1f},{res['cy']:>7.1f}) "
              f"{off:>7.2f} {vc:>7.1f}{flag}")
    # 绿球重点
    grn = [b for b in st.balls if b.color == "green"]
    if grn:
        b = grn[0]
        gx, gy = b.x + fx, b.y + fy
        print(f"\n  绿球重点：报告=({gx:.1f},{gy:.1f})")
        res = center_by_vstd(vs, gx, gy, r_raw)
        if res:
            print(f"    Vstd 定位=({res['cx']:.1f},{res['cy']:.1f})  "
                  f"偏移={np.hypot(res['cx']-gx,res['cy']-gy):.2f}px  面积={res['area']}")
            print(f"    该处 Vstd={vs[int(gy),int(gx)]:.1f}")
        # 邻域 Vstd 峰值位置
        R = int(r_raw * 2.0)
        x0, y0 = max(0, int(gx) - R), max(0, int(gy) - R)
        x1, y1 = min(vs.shape[1], int(gx) + R), min(vs.shape[0], int(gy) + R)
        sub = vs[y0:y1, x0:x1]
        jj = np.unravel_index(int(np.argmax(sub)), sub.shape)
        px, py = x0 + jj[1], y0 + jj[0]
        print(f"    邻域 Vstd 峰值={sub.max():.1f} @({px},{py})  "
              f"距报告球心 {np.hypot(px-gx,py-gy):.2f}px")
