# -*- coding: utf-8 -*-
"""诊断绿球：为何它的球心/半径特别不准。

绿球颜色与台呢同为绿色，只在明暗上有差异，
是全部球里最难的。用户反馈「绿球歪了」。

本脚本对绿球做：
  1. 打印其邻域的「非台呢」掩码 ASCII，看它是否成形；
  2. 用不同「台呢判据」测它的边缘，看结论是否稳定；
  3. 与红球（易检测）对照。
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


def fit_circle(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    return None if r2 <= 0 else (float(cx), float(cy), float(np.sqrt(r2)))


def edge_pts(mask, cx, cy, rmax, n=720):
    H, W = mask.shape
    out = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        dx, dy = np.cos(th), np.sin(th)
        prev = None
        for k in range(2, rmax):
            px, py = int(round(cx + dx * k)), int(round(cy + dy * k))
            if not (0 <= px < W and 0 <= py < H):
                break
            cur = bool(mask[py, px])
            if prev is False and cur:
                out.append((cx + dx * (k - .5), cy + dy * (k - .5)))
                break
            prev = cur
    return np.array(out) if out else np.empty((0, 2))


def best_fit(mask, cx0, cy0, rmax, iters=4):
    cx, cy, r = float(cx0), float(cy0), None
    for _ in range(iters):
        p = edge_pts(mask, cx, cy, rmax)
        if len(p) < 30:
            return None
        d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        med = float(np.median(d))
        sel = (d > med * 0.85) & (d < med * 1.15)
        if sel.sum() < 20:
            return None
        f = fit_circle(p[sel])
        if f is None:
            return None
        cx, cy, r = f
    return cx, cy, r


def masks_of(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        # 台呢判据（mask=1 表示台呢）
        "G1 严": ((Hc >= 40) & (Hc <= 88) & (Sc > 55) & (Vc > 40)),
        "G2 中": ((Hc >= 35) & (Hc <= 95) & (Sc > 35)),
        "G3 宽": ((Hc >= 30) & (Hc <= 100) & (Sc > 25)),
    }


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

for tag, img in frames:
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    fx, fy, fw, fh = st.felt_rect
    r_rep = float(np.median([b.r for b in st.balls]))
    r_raw = r_rep / config.BALL_RADIUS_FIX_SCALE
    print(f"\n{'='*80}")
    print(f"{tag}  {img.shape[1]}x{img.shape[0]}  报告r={r_rep:.2f} 引擎原始r={r_raw:.2f}")
    print("=" * 80)

    grn = [b for b in st.balls if b.color == "green"]
    red = [b for b in st.balls if b.color == "red"]
    picks = (grn[:1] + red[:1])
    ms = masks_of(img)

    for b in picks:
        gx, gy = b.x + fx, b.y + fy
        print(f"\n  --- {b.color} 报告球心=({gx:.1f},{gy:.1f}) ---")
        # 邻域非台呢 ASCII（用 G2）
        R = int(r_raw * 1.8)
        x0, y0 = int(gx) - R, int(gy) - R
        x1, y1 = int(gx) + R, int(gy) + R
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.shape[1], x1), min(img.shape[0], y1)
        sub = ms["G2 中"][y0c:y1c, x0c:x1c]
        # 下采样打印
        step = max(1, sub.shape[0] // 26)
        print(f"    非台呢掩码（'.'=台呢  '#'=非台呢，步长{step}）:")
        for j in range(0, sub.shape[0], step):
            line = "".join("#" if sub[j, i] else "." for i in range(0, sub.shape[1], step))
            print(f"      {line}")
        # 各判据下的拟合
        print(f"    {'判据':>6} {'拟合球心':>18} {'偏移':>7} {'拟合r':>7} {'r/原始':>7}")
        for mname, m in ms.items():
            f = best_fit(m, gx, gy, int(r_raw * 2.2))
            if f is None:
                print(f"    {mname:>6} {'拟合失败':>18}")
                continue
            ncx, ncy, nr = f
            off = float(np.hypot(ncx - gx, ncy - gy))
            print(f"    {mname:>6} ({ncx:>8.1f},{ncy:>8.1f}) {off:>7.2f} "
                  f"{nr:>7.2f} {nr/r_raw:>7.4f}")
