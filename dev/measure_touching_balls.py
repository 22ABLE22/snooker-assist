# -*- coding: utf-8 -*-
"""用「相贴球」直接测球直径 —— 不依赖任何颜色阈值。

原理（这是最硬的几何约束）
------------------------
两颗球相贴时，球心距**恒等于 2R**，与渲染、阴影、抗锯齿全都无关。
球心本身是亚像素精度测出来的，所以这是最干净的 R 测量。

用来裁决最后一个歧义：
    幽灵圈 内沿 13.50 / 中径 15.00 / 外沿 16.50
    R 到底等于哪一个？
        min 球心距 ≈ 33.0 -> R = 16.5（圈外沿）
        min 球心距 ≈ 30.0 -> R = 15.0（圈中径）
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

print("=" * 80)
print("用相贴球测直径（球心距 = 2R，与渲染无关）")
print("=" * 80)

cands = []
for nm in ("full_rack.png", "live_v2.png", "raw_0.png", "diag_now.png",
           "f8_sim.png", "final_delivery.png"):
    p = ROOT / "samples" / "evidence" / nm
    if p.exists():
        cands.append((nm, p))
for p in sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:4]:
    cands.append((p.name, p))

all_min = []
for nm, p in cands:
    img = cv2.imread(str(p))
    st = detect.analyze_frame(img)
    if not st.ok() or len(st.balls) < 3:
        continue
    r_fix = float(np.median([b.r for b in st.balls]))
    fh = st.felt_rect[3]
    # 引擎原始半径（修正前）
    r_eng = r_fix / config.BALL_RADIUS_FIX_SCALE

    pts = np.array([[b.x, b.y] for b in st.balls])
    cols = [b.color for b in st.balls]
    n = len(pts)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(*(pts[i] - pts[j])))
            pairs.append((d, cols[i], cols[j]))
    pairs.sort()
    if not pairs:
        continue
    dmin = pairs[0][0]
    all_min.append((nm, dmin, r_eng, fh))
    print(f"\n--- {nm}  球数={n} ---")
    print(f"  最小球心距（前 5）:")
    for d, c1, c2 in pairs[:5]:
        print(f"    {d:>7.2f}px  ({c1}-{c2})  -> 若为相贴则 R = {d/2:>5.2f}")
    print(f"  引擎原始 R = {r_eng:.2f}（2R = {2*r_eng:.2f}）")
    print(f"  修正后   R = {r_fix:.2f}（2R = {2*r_fix:.2f}）")

print()
print("=" * 80)
print("汇总：最小球心距 vs 2R")
print("=" * 80)
print(f"  {'样本':>18} {'最小球心距':>10} {'/2':>7} {'引擎2R':>8} {'修正后2R':>9}")
for nm, dmin, r_eng, fh in all_min:
    print(f"  {nm:>18} {dmin:>10.2f} {dmin/2:>7.2f} {2*r_eng:>8.2f} "
          f"{2*r_eng*config.BALL_RADIUS_FIX_SCALE:>9.2f}")

if all_min:
    dmins = np.array([t[1] for t in all_min])
    print()
    print(f"  最小球心距总体最小 = {dmins.min():.2f}  -> R_min = {dmins.min()/2:.2f}")
    print(f"  最小值的中位         = {np.median(dmins):.2f}  -> R = {np.median(dmins)/2:.2f}")
    print()
    print("  裁决：")
    print(f"    幽灵圈 内沿/中径/外沿 = 13.50 / 15.00 / 16.50")
    for lbl, R in (("圈内沿", 13.50), ("圈中径", 15.00), ("圈外沿", 16.50)):
        print(f"      {lbl} {R:>5.2f} -> 2R = {2*R:>6.2f}")
    print(f"    实测最小球心距 = {dmins.min():.2f}")
    best = min((("圈内沿", 13.50), ("圈中径", 15.00), ("圈外沿", 16.50)),
               key=lambda t: abs(2 * t[1] - dmins.min()))
    print(f"    >>> 最接近的是「{best[0]}」(2R={2*best[1]:.2f})")
    print()
    print("  注意：只有**真正相贴**的球对才能用。若样本中没有相贴球，")
    print("        最小球心距会偏大，此裁决不成立。需人工确认该对是否相贴。")
