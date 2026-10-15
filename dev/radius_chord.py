# -*- coding: utf-8 -*-
"""用「过球心的水平弦长」直接量球直径，验证距离变换半径是否正确。

对半径 r 的圆盘，过圆心的水平弦长 = 2r，这是无偏的几何测量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import detect2 as D2  # noqa: E402

for tag, p in (("win_best", "samples/recognize_12frames/frame_01_raw.png"), ("now", "samples/recognize_12frames/frame_02_raw.png")):
    img = cv2.imread(str(ROOT / p))
    fx, fy, fw, fh = D2.find_felt(img)
    crop = img[fy + 3 : fy + fh - 3, fx + 3 : fx + fw - 3]
    d, lab = D2.response_map(crop)
    st = D2.analyze_frame2(img)
    m = (d > 20).astype(np.uint8)
    h, w = m.shape[:2]

    print(f"\n=== {tag}  报告半径 r={st.balls[0].r:.2f} ===")
    print(f"  {'颜色':>6} {'位置':>16} {'水平弦/2':>9} {'垂直弦/2':>9} {'均值':>7}")
    chords = []
    for b in st.balls:
        cx, cy = int(round(b.x)), int(round(b.y))
        # 水平弦
        row = m[cy, :]
        x0 = cx
        while x0 > 0 and row[x0 - 1] > 0:
            x0 -= 1
        x1 = cx
        while x1 < w - 1 and row[x1 + 1] > 0:
            x1 += 1
        ch = (x1 - x0 + 1) / 2.0
        # 垂直弦
        col = m[:, cx]
        y0 = cy
        while y0 > 0 and col[y0 - 1] > 0:
            y0 -= 1
        y1 = cy
        while y1 < h - 1 and col[y1 + 1] > 0:
            y1 += 1
        cv_ = (y1 - y0 + 1) / 2.0
        avg = (ch + cv_) / 2.0
        # 弦长超过球径说明与邻球/库边粘连，剔除
        if ch < b.r * 1.9 and cv_ < b.r * 1.9:
            chords.append(avg)
        if b.color in ("red", "cue", "black", "brown", "blue", "green"):
            print(f"  {b.color:>6} ({b.x:>6.1f},{b.y:>6.1f}) {ch:>9.2f} {cv_:>9.2f} {avg:>7.2f}")
    chords = np.array(chords)
    print(f"  弦长法: 中位={np.median(chords):.2f} 均值={chords.mean():.2f} n={len(chords)}")
    print(f"  距离变换法: {st.balls[0].r:.2f}   差异={st.balls[0].r - np.median(chords):+.2f}px")
