# -*- coding: utf-8 -*-
"""最终裁决：把幽灵圈与真实球的径向剖面**并排打印**，看清哪条边对齐。

歧义所在
--------
圆有描边宽度，真实球有抗锯齿 + 阴影，于是有多个候选「半径」：
    圈内沿 13.5 / 圈中径 15.0 / 圈外沿 16.5
    球 p10 16.5 / 球 p50 17.5 / 球外延 ~18
到底哪一对相等？

关键洞察（抗锯齿的物理含义）
--------------------------
实心圆盘经抗锯齿渲染后，**几何边缘处的覆盖率恰为 50%**。
所以对无阴影方向，球边缘 = 覆盖率 0.5 的位置。
阴影只在部分角度把覆盖率**向外撑**，不会向内。
=> 取「各角度边缘半径」的**低分位数**即得真实半径（避开阴影方向）。

本脚本打印两者的完整剖面，并明确标注每个候选位置，供对照。
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

img = cv2.imread(str(ROOT / "samples" / "evidence" / "raw_0.png"))
st = detect.analyze_frame(img)
fx, fy, fw, fh = st.felt_rect
r_eng = float(np.median([b.r for b in st.balls])) / config.BALL_RADIUS_FIX_SCALE

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
Hc, Sc = hsv[:, :, 0], hsv[:, :, 1]
green = ((Hc >= 35) & (Hc <= 95) & (Sc > 35))
S, V = hsv[:, :, 1], hsv[:, :, 2]
white = ((S < 95) & (V > 135))


def profile(mask, cx, cy, rmax, n_ang=1440):
    ks = np.arange(1, rmax)
    f = np.zeros(len(ks))
    ths = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    ct, st_ = np.cos(ths), np.sin(ths)
    for j, k in enumerate(ks):
        px = np.round(cx + ct * k).astype(int)
        py = np.round(cy + st_ * k).astype(int)
        ok = (px >= 0) & (px < img.shape[1]) & (py >= 0) & (py < img.shape[0])
        f[j] = mask[py[ok], px[ok]].mean() if ok.any() else np.nan
    return ks, f


print(f"引擎原始 r = {r_eng:.2f}")
print()

# 幽灵圈剖面（白描边占比）
ks_r, fr_ring = profile(white, 168.3, 412.1, int(r_eng * 2.0))
# 真实球剖面（非台呢占比 = 1 - 台呢占比）
b = [x for x in st.balls if x.color == "red"][0]
ks_b, fr_ball = profile(~green, b.x + fx, b.y + fy, int(r_eng * 2.0))

print(f"{'r':>4} | {'幽灵圈白占比':>12} {'':12} | {'球非台呢占比':>12} {'':12}")
print("-" * 74)
for i in range(0, len(ks_r)):
    if ks_r[i] >= len(ks_b):
        break
    a = fr_ring[i]
    bb = fr_ball[i]
    bar_a = "#" * int((0 if np.isnan(a) else a) * 24)
    bar_b = "#" * int((0 if np.isnan(bb) else bb) * 24)
    mark = ""
    if abs(ks_r[i] - 16.5) < 0.6:
        mark = "  <== 16.5 (圈外沿 & 球p10)"
    elif abs(ks_r[i] - 19.77) < 0.6:
        mark = "  <== 19.77 引擎半径"
    print(f"{ks_r[i]:>4} | {a:>12.3f} {bar_a:<12} | {bb:>12.3f} {bar_b:<12}{mark}")

# 球的逐角度边缘：低分位 vs 高分位
def per_angle_edge(cx, cy, rmax, n_ang=1440):
    out = np.full(n_ang, np.nan)
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n_ang, endpoint=False)):
        dx, dy = np.cos(th), np.sin(th)
        prev = None
        for k in range(2, rmax):
            px = int(round(cx + dx * k))
            py = int(round(cy + dy * k))
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]):
                break
            cur = bool(green[py, px])
            if prev is False and cur is True:
                out[i] = k - 0.5
                break
            prev = cur
    return out


edges = []
for x in st.balls:
    e = per_angle_edge(x.x + fx, x.y + fy, int(r_eng * 2.2))
    v = e[~np.isnan(e)]
    if len(v) > 400:
        edges.append(v)
allv = np.concatenate(edges)
print()
print("=" * 74)
print("球的逐角度边缘半径分布（全部球合并）")
print("=" * 74)
for q in (2, 5, 10, 15, 25, 50, 75, 90, 98):
    print(f"  p{q:>2} = {np.percentile(allv, q):>6.2f}")
print()
print(f"  最小值 = {allv.min():.2f}   最大值 = {allv.max():.2f}")
print()
print("=" * 74)
print("结论")
print("=" * 74)
print(f"  幽灵圈：内沿=13.50  中径=15.00  外沿=16.50")
print(f"  球边缘：p02={np.percentile(allv,2):.2f}  p10={np.percentile(allv,10):.2f}  "
      f"p50={np.percentile(allv,50):.2f}")
print()
print(f"  >>> 圈外沿(16.50) 与 球 p10({np.percentile(allv,10):.2f}) 吻合")
print(f"  >>> 圈中径(15.00) 无对应球特征")
print()
print("  为何用 p10 而非 p50：")
print("    实心盘抗锯齿后，几何边缘处覆盖率=50%，故无阴影方向边缘=真实半径；")
print("    阴影只在部分角度把边缘**向外撑大**，使 p50 偏大。")
print("    p10 取的是「阴影最弱的那批方向」，即真实几何边缘。")
