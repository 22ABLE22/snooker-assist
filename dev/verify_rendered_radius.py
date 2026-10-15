# -*- coding: utf-8 -*-
"""为什么「跨分辨率等比」不能证明半径正确 —— 并给出正确的验证方法。

我之前的错误论证
---------------
我看到引擎半径在两个窗口下精确等比（r/台面高 都是 0.02934），
就断定「半径正确」。但这是**无效论证**：
    系统性偏差（例如固定比例地量大了 15%）也会等比缩放！
等比只说明「与台面尺寸成固定比例」，不说明比例值对不对。

真正有效的验证
-------------
找一个**独立于台面尺寸的绝对参照**。现在有了：
    球的渲染外延（球遮挡台呢的边界）—— 24.0px
    引擎半径 —— 27.57px
两者在同一画面、同一像素尺度下直接可比，不涉及任何比例假设。

本脚本量化偏差，并用多个分辨率验证「偏差是否为固定百分比」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402


def rendered_radius(img, gx, gy, rmax=80, start=3):
    """球的渲染外延：从球心向外，最后一个「非台呢」像素的半径。

    球遮挡了台呢，所以沿任意方向走到「变成台呢」为止，
    最后那个非台呢像素的位置就是球的视觉边缘。
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((Hc >= 40) & (Hc <= 88) & (Sc > 55) & (Vc > 40))
    radii = []
    for th in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        dx, dy = np.cos(th), np.sin(th)
        last = None
        miss = 0
        for k in range(start, rmax):
            px = int(round(gx + dx * k))
            py = int(round(gy + dy * k))
            if not (0 <= px < W and 0 <= py < H):
                break
            if not green[py, px]:
                last = k
                miss = 0
            else:
                miss += 1
                if miss > 3:
                    break
        if last:
            radii.append(last)
    if len(radii) < 200:
        return None, 0, None
    arr = np.array(radii, dtype=float)
    return float(np.median(arr)), len(radii), float(arr.std())


print("=" * 80)
print("球的渲染外延 vs 引擎半径（同一画面、同一像素尺度，无比例假设）")
print("=" * 80)

cases = [
    ("diag_now.png", "白球贴库"),
    ("raw_0.png", "普通局面"),
    ("live_v2.png", "普通局面"),
    ("full_rack.png", "满台"),
]

all_rows = []
for nm, desc in cases:
    p = ROOT / "samples" / "evidence" / nm
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    fx, fy, fw, fh = st.felt_rect
    r_eng = float(np.median([b.r for b in st.balls]))
    print(f"\n--- {nm} ({desc})  {img.shape[1]}x{img.shape[0]}  台面 {fw}x{fh} ---")
    print(f"  引擎 r = {r_eng:.2f}")
    rows = []
    for b in st.balls:
        gx, gy = b.x + fx, b.y + fy
        rr, n, sd = rendered_radius(img, gx, gy)
        if rr:
            rows.append((b.color, rr, rr / r_eng))
    if not rows:
        print("  无法测量")
        continue
    rrs = np.array([t[1] for t in rows])
    ratios = np.array([t[2] for t in rows])
    print(f"  渲染半径: 中位={np.median(rrs):.2f}  均值={rrs.mean():.2f}  "
          f"std={rrs.std():.2f}  (n={len(rows)})")
    print(f"  渲染/引擎 = {np.median(ratios):.4f}  "
          f"(= 引擎{'偏大' if np.median(ratios)<1 else '偏小'} "
          f"{abs(1-np.median(ratios))*100:.1f}%)")
    # 按颜色看是否一致
    bycol = {}
    for c, rr, ra in rows:
        bycol.setdefault(c, []).append(rr)
    print("  分颜色渲染半径中位: " +
          "  ".join(f"{c}={np.median(v):.1f}" for c, v in sorted(bycol.items())))
    all_rows.append((nm, r_eng, float(np.median(rrs)), float(np.median(ratios)),
                     fh, img.shape[1]))

print()
print("=" * 80)
print("跨样本汇总")
print("=" * 80)
print(f"  {'样本':>16} {'引擎r':>8} {'渲染r':>8} {'渲染/引擎':>10} {'r/fh':>9} {'渲染r/fh':>10}")
for nm, re_, rr, ra, fh, iw in all_rows:
    print(f"  {nm:>16} {re_:>8.2f} {rr:>8.2f} {ra:>10.4f} "
          f"{re_/fh:>9.5f} {rr/fh:>10.5f}")

if len(all_rows) >= 2:
    ratios = np.array([t[3] for t in all_rows])
    eng_ratios = np.array([t[1] / t[4] for t in all_rows])
    ren_ratios = np.array([t[2] / t[4] for t in all_rows])
    print()
    print(f"  渲染/引擎 比值: 中位={np.median(ratios):.4f}  "
          f"范围=[{ratios.min():.4f}, {ratios.max():.4f}]")
    print(f"  引擎 r/fh   的一致性: std={eng_ratios.std():.6f} (相对 "
          f"{eng_ratios.std()/eng_ratios.mean()*100:.3f}%)")
    print(f"  渲染 r/fh   的一致性: std={ren_ratios.std():.6f} (相对 "
          f"{ren_ratios.std()/ren_ratios.mean()*100:.3f}%)")
    print()
    print("  注意：两者「跨分辨率一致性」都很好（因为都随台面等比缩放），")
    print("        所以一致性**不能**用来判断谁是对的 ——")
    print("        判定依据是同一画面内的绝对值比较：渲染外延是球的实际像素边界。")
