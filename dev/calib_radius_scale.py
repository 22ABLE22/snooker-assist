# -*- coding: utf-8 -*-
"""球半径标定（阴影稳健 + 亚像素）。

原理
----
球带**方向性阴影**：阴影只在部分角度把视觉边缘撑大，绝不会让其变小。
故取边缘分布的**低分位数**（p10）即为球体真实边界；
中位数会被阴影污染而偏大。

两个独立真值来源
  1. 球体轮廓的 p10（跨球取中位）
  2. 游戏自绘**幽灵球空心圈**的外沿（用户指出：圈=真实球大小）

本脚本给出亚像素精度的测量与推荐的修正系数。
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


def make_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc = hsv[:, :, 0], hsv[:, :, 1]
    return ((Hc >= 35) & (Hc <= 95) & (Sc > 35))


def make_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    return ((S < 95) & (V > 135))


def edge_radii(mask, cx, cy, rmax, n_ang=1440):
    """逐角度求「mask 由 False 变 True」的半径，带线性亚像素插值。"""
    H, W = mask.shape
    out = np.full(n_ang, np.nan)
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n_ang, endpoint=False)):
        dx, dy = np.cos(th), np.sin(th)
        prev = None                              # 上一采样点的布尔
        for k in range(2, rmax):
            px = int(round(cx + dx * k))
            py = int(round(cy + dy * k))
            if not (0 <= px < W and 0 <= py < H):
                break
            cur = bool(mask[py, px])
            if prev is False and cur is True:
                # 在 k-1 与 k 之间插值到「恰好跨过」的位置
                out[i] = k - 0.5
                break
            prev = cur
    return out


def ring_edges(img, cx, cy, rmax, n_ang=1440):
    """幽灵圈：每角度记录白描边的起止半径（用于取内/外沿）。"""
    W_ = make_white(img)
    H, W = W_.shape
    ins = np.full(n_ang, np.nan)
    outs = np.full(n_ang, np.nan)
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n_ang, endpoint=False)):
        dx, dy = np.cos(th), np.sin(th)
        first = last = None
        for k in range(2, rmax):
            px = int(round(cx + dx * k))
            py = int(round(cy + dy * k))
            if not (0 <= px < W and 0 <= py < H):
                break
            if W_[py, px]:
                if first is None:
                    first = k
                last = k
            elif first is not None:
                break
        if first is not None and last is not None:
            ins[i] = first - 0.5
            outs[i] = last + 0.5
    return ins, outs


print("=" * 82)
print("球半径标定（阴影稳健：低分位数 + 亚像素）")
print("=" * 82)

FRAMES = [("raw_0.png", 168.3, 412.1),      # 含幽灵圈
          ("live_v2.png", None, None),
          ("full_rack.png", None, None),
          ("diag_now.png", None, None)]

summary = []
for nm, rcx, rcy in FRAMES:
    p = ROOT / "samples" / "evidence" / nm
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    fx, fy, fw, fh = st.felt_rect
    # 还原「引擎原始半径」= 修正前的值
    r_eng = float(np.median([b.r for b in st.balls])) / config.BALL_RADIUS_FIX_SCALE

    green = make_green(img)
    per_ball = []
    for b in st.balls:
        gx, gy = b.x + fx, b.y + fy
        e = edge_radii(green, gx, gy, int(r_eng * 2.2))
        v = e[~np.isnan(e)]
        if len(v) < 400:
            continue
        per_ball.append((float(np.percentile(v, 10)),
                         float(np.percentile(v, 50))))
    if not per_ball:
        continue
    p10 = float(np.median([t[0] for t in per_ball]))
    p50 = float(np.median([t[1] for t in per_ball]))
    print(f"\n--- {nm}  台面 {fw}x{fh}  引擎原始 r={r_eng:.3f} ---")
    print(f"  球轮廓 p10（阴影稳健）= {p10:.3f}   /引擎 = {p10/r_eng:.4f}")
    print(f"  球轮廓 p50（含阴影）  = {p50:.3f}   /引擎 = {p50/r_eng:.4f}")
    # 阴影造成的额外外扩
    print(f"  阴影外扩量 = {p50-p10:.2f}px "
          f"（{(p50-p10)/r_eng*100:.1f}% 引擎半径）")

    if rcx is not None:
        ins, outs = ring_edges(img, rcx, rcy, int(r_eng * 2.2))
        iv = ins[~np.isnan(ins)]
        ov = outs[~np.isnan(outs)]
        print(f"  幽灵圈: 内沿={np.median(iv):.3f}  "
              f"外沿={np.median(ov):.3f}  描边宽={np.median(ov)-np.median(iv):.2f}")
        print(f"    外沿/引擎 = {np.median(ov)/r_eng:.4f}")
        print(f"    >>> 外沿({np.median(ov):.2f}) vs 球 p10({p10:.2f}) "
              f"差 {np.median(ov)-p10:+.2f}px")
        summary.append((nm, r_eng, p10, p50, r_eng and np.median(ov)))
    else:
        summary.append((nm, r_eng, p10, p50, None))

print()
print("=" * 82)
print("汇总")
print("=" * 82)
print(f"  {'样本':>15} {'引擎r':>7} {'球p10':>7} {'球p50':>7} {'圈外沿':>7} "
      f"{'p10/引擎':>9} {'圈/引擎':>8}")
for nm, re_, p10, p50, ro in summary:
    rs = f"{ro:>7.2f}" if ro else f"{'-':>7}"
    rt = f"{ro/re_:>8.4f}" if ro else f"{'-':>8}"
    print(f"  {nm:>15} {re_:>7.2f} {p10:>7.2f} {p50:>7.2f} {rs} "
          f"{p10/re_:>9.4f} {rt}")

p10s = np.array([t[2] / t[1] for t in summary])
rings = np.array([t[4] / t[1] for t in summary if t[4]])
print()
print(f"  球 p10 / 引擎 : 中位={np.median(p10s):.4f}  "
      f"范围=[{p10s.min():.4f}, {p10s.max():.4f}]")
if len(rings):
    print(f"  圈外沿 / 引擎 : 中位={np.median(rings):.4f}")
comb = np.concatenate([p10s, rings]) if len(rings) else p10s
print()
print(f"  >>> 推荐修正系数 = {np.median(comb):.4f}")
print(f"      （由 {len(p10s)} 个球轮廓测量"
      + (f" + {len(rings)} 个幽灵圈测量" if len(rings) else "")
      + " 取中位）")
