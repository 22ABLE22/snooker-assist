# -*- coding: utf-8 -*-
"""袋口精度复核（修正配对逻辑）。

上一版测量脚本用「最近暗洞」配对，中袋会误配到角袋。
本版按「角袋/中袋」分别限定搜索区域，避免错配。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402

FRAMES = [ROOT / "samples" / "evidence" / "live_v2.png"]
FRAMES += sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:2]


def dark_holes(img: np.ndarray) -> list[tuple[float, float, float]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    m = (V < 60).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 200:
            continue
        bw, bh = int(stats[i, 2]), int(stats[i, 3])
        if a / max(bw * bh, 1) < 0.5:
            continue
        r_eq = float(np.sqrt(a / np.pi))
        if r_eq > 70:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), r_eq))
    return out


print("=" * 78)
print("袋口精度：detect 输出 vs 整图暗洞真值")
print("=" * 78)

for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    st = detect.analyze_frame(img)
    fx, fy, fw, fh = st.felt_rect
    r = st.balls[0].r if st.balls else 19.77
    holes = dark_holes(img)

    print(f"\n--- {fp.stem}  felt=({fx},{fy},{fw},{fh})  r={r:.2f} ---")
    print(f"  {'#':>2} {'类型':>4} {'detect 袋口':>18} {'真值暗洞':>18} {'偏差px':>8} {'占半径':>7}")
    devs = []
    for i, (a, b) in enumerate(st.pockets):
        gx, gy = fx + a * fw, fy + b * fh
        is_mid = abs(a - 0.5) < 0.15
        # 中袋：只找 x 接近的；角袋：只找靠角的
        if is_mid:
            cand = [h for h in holes if abs(h[0] - gx) < fw * 0.12]
        else:
            cand = [h for h in holes
                    if abs(h[0] - gx) < fw * 0.12 and abs(h[1] - gy) < fh * 0.12]
        if not cand:
            print(f"  {i+1:>2} {'中袋' if is_mid else '角袋':>4} ({gx:>7.0f},{gy:>7.0f}) "
                  f"{'未找到':>18}")
            continue
        j = min(range(len(cand)),
                key=lambda k: (cand[k][0] - gx) ** 2 + (cand[k][1] - gy) ** 2)
        hx, hy, _hr = cand[j]
        dv = float(np.hypot(hx - gx, hy - gy))
        devs.append((is_mid, dv))
        print(f"  {i+1:>2} {'中袋' if is_mid else '角袋':>4} ({gx:>7.0f},{gy:>7.0f}) "
              f"({hx:>7.0f},{hy:>7.0f}) {dv:>8.1f} {dv/r:>7.2f}")

    corner = [d for m, d in devs if not m]
    mid = [d for m, d in devs if m]
    if corner:
        print(f"  角袋偏差: 中位={np.median(corner):.1f}px max={max(corner):.1f}px")
    if mid:
        print(f"  中袋偏差: 中位={np.median(mid):.1f}px max={max(mid):.1f}px")
