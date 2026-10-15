# -*- coding: utf-8 -*-
"""用「边缘点圆拟合」同时求真实圆心与半径（含稳健剔除）。

为什么换方法
-----------
之前是「以检测到的球心为原点，做径向剖面量半径」。
但球心本身可能有偏差（阴影会把质心拉偏），
圆心一偏，径向剖面就被拉成椭圆，量出的半径不可靠
—— 这正对应反馈里的「绿球歪了、圈不准」。

新方法（循环精化）
  1. 以当前球心为中心，逐角度找球边缘点；
  2. 对边缘点做**稳健最小二乘圆拟合**（剔除离群点），
     同时得到**精化圆心**与**半径**；
  3. 用新圆心重复 1~2，直至收敛。

这既修圆心（解决「歪」），又给出一致的半径（解决「偏小」）。
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


def fit_circle(pts: np.ndarray):
    """最小二乘圆拟合。pts: Nx2 -> (cx, cy, r)"""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def edge_points(green, cx, cy, rmax, n_ang=720):
    """从 (cx,cy) 向外逐角度找「台呢由 False 变 True」的边缘点。"""
    H, W = green.shape
    pts = []
    for th in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        dx, dy = np.cos(th), np.sin(th)
        prev = None
        for k in range(2, rmax):
            px = int(round(cx + dx * k))
            py = int(round(cy + dy * k))
            if not (0 <= px < W and 0 <= py < H):
                break
            cur = bool(green[py, px])
            if prev is False and cur:
                pts.append((cx + dx * (k - 0.5), cy + dy * (k - 0.5)))
                break
            prev = cur
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


def robust_fit(green, cx0, cy0, rmax, iters=3, keep=0.55):
    """循环精化：每轮用「半径处于中位附近」的点做拟合。"""
    cx, cy = float(cx0), float(cy0)
    r = None
    for _ in range(iters):
        pts = edge_points(green, cx, cy, rmax)
        if len(pts) < 30:
            return None
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        med = float(np.median(d))
        # 只保留靠近中位的点（剔除被邻球/库边拉长的离群）
        lo, hi = med * (1 - (1 - keep)), med * (1 + (1 - keep))
        sel = (d >= lo) & (d <= hi)
        if sel.sum() < 20:
            return None
        f = fit_circle(pts[sel])
        if f is None:
            return None
        cx, cy, r = f
    return cx, cy, r, int(sel.sum())


def analyze_frame(img, tag):
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        return None
    fx, fy, fw, fh = st.felt_rect
    r_rep = float(np.median([b.r for b in st.balls]))
    r_raw = r_rep / config.BALL_RADIUS_FIX_SCALE
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc = hsv[:, :, 0], hsv[:, :, 1]
    green = ((Hc >= 35) & (Hc <= 95) & (Sc > 35))

    print(f"\n{'='*82}")
    print(f"{tag}  {img.shape[1]}x{img.shape[0]}  台面 {fw}x{fh}  球数={len(st.balls)}")
    print(f"  报告 r={r_rep:.2f}   引擎原始 r={r_raw:.2f}   当前系数={config.BALL_RADIUS_FIX_SCALE}")
    print("=" * 82)
    print(f"  {'颜色':>7} {'报告球心':>17} {'拟合球心':>17} {'偏移':>7} "
          f"{'报告r':>7} {'拟合r':>7} {'拟合/原始':>9}")
    print("  " + "-" * 84)

    rows = []
    for b in st.balls:
        gx, gy = b.x + fx, b.y + fy
        out = robust_fit(green, gx, gy, int(r_raw * 2.2))
        if not out:
            print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f})  拟合失败")
            continue
        ncx, ncy, nr, ns = out
        off = float(np.hypot(ncx - gx, ncy - gy))
        rows.append((b.color, gx, gy, ncx, ncy, off, b.r, nr))
        print(f"  {b.color:>7} ({gx:>7.1f},{gy:>7.1f}) ({ncx:>7.1f},{ncy:>7.1f}) "
              f"{off:>7.2f} {b.r:>7.2f} {nr:>7.2f} {nr/r_raw:>9.4f}")

    if not rows:
        return None
    offs = np.array([t[5] for t in rows])
    nrs = np.array([t[7] for t in rows])
    print("  " + "-" * 84)
    print(f"  球心偏移: 中位={np.median(offs):.2f}px  均值={offs.mean():.2f}  "
          f"max={offs.max():.2f}")
    print(f"  拟合半径: 中位={np.median(nrs):.2f}px  "
          f"比值(拟合/引擎原始)={np.median(nrs)/r_raw:.4f}")
    # 颜色分组，看绿球是否异常
    bycol = {}
    for c, _a, _b, _cc, _dd, o, rr, nr in rows:
        bycol.setdefault(c, []).append((o, nr))
    print(f"  分颜色（偏移 / 拟合半径）:")
    for c in sorted(bycol):
        v = bycol[c]
        print(f"    {c:<7} 偏移中位={np.median([t[0] for t in v]):>5.2f}px  "
              f"拟合r中位={np.median([t[1] for t in v]):>6.2f}")
    return dict(tag=tag, r_raw=r_raw, ratio=float(np.median(nrs) / r_raw),
                off=float(np.median(offs)), rows=rows)


results = []
for nm in ("raw_0.png", "live_v2.png", "full_rack.png", "diag_now.png"):
    p = ROOT / "samples" / "evidence" / nm
    if p.exists():
        r = analyze_frame(cv2.imread(str(p)), nm)
        if r:
            results.append(r)

# 实时帧
try:
    import capture
    info = capture.find_window_rect()
    if info:
        img = capture.print_window_bgr(info["hwnd"])
        if img is not None:
            r = analyze_frame(img, "实时帧")
            if r:
                results.append(r)
except Exception as e:
    print("实时帧失败:", e)

print()
print("=" * 82)
print("汇总：拟合半径 / 引擎原始半径")
print("=" * 82)
for r in results:
    print(f"  {r['tag']:>16}  原始r={r['r_raw']:>6.2f}  比值={r['ratio']:.4f}  "
          f"球心偏移中位={r['off']:.2f}px")
if results:
    rr = np.array([r["ratio"] for r in results])
    print()
    print(f"  比值中位 = {np.median(rr):.4f}   范围=[{rr.min():.4f}, {rr.max():.4f}]")
    print(f"  当前使用系数 = {config.BALL_RADIUS_FIX_SCALE}")
    print(f"  >>> 建议系数 = {np.median(rr):.4f}")
