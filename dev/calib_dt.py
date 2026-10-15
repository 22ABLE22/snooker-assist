# -*- coding: utf-8 -*-
"""标定距离变换阈值：使 DT 半径与「弦长法」实测一致。"""
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
    h, w = d.shape[:2]

    # 弦长法真值（只取干净孤立的球）
    chords = []
    for b in st.balls:
        cx, cy = int(round(b.x)), int(round(b.y))
        row = (d[cy, :] > 20).astype(np.uint8)
        x0 = cx
        while x0 > 0 and row[x0 - 1]:
            x0 -= 1
        x1 = cx
        while x1 < w - 1 and row[x1 + 1]:
            x1 += 1
        col = (d[:, cx] > 20).astype(np.uint8)
        y0 = cy
        while y0 > 0 and col[y0 - 1]:
            y0 -= 1
        y1 = cy
        while y1 < h - 1 and col[y1 + 1]:
            y1 += 1
        ch, cvv = (x1 - x0 + 1) / 2.0, (y1 - y0 + 1) / 2.0
        if ch < b.r * 1.9 and cvv < b.r * 1.9:
            chords.append((ch + cvv) / 2.0)
    truth = float(np.median(chords))

    print(f"\n=== {tag}  弦长法真值 r={truth:.2f} (n={len(chords)}) ===")
    print(f"  {'阈值':>6} {'DT中位':>8} {'与真值差':>9}")
    for thr in (5, 8, 10, 12, 15, 18, 20, 25, 30):
        m = (d > thr).astype(np.uint8)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        vals = []
        for b in st.balls:
            cx, cy = int(round(b.x)), int(round(b.y))
            rad = int(b.r * 1.6)
            x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
            y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
            sub = dt[y0:y1, x0:x1]
            if sub.size:
                pk = float(sub.max())
                if b.r * 0.75 <= pk <= b.r * 1.35:
                    vals.append(pk)
        if not vals:
            continue
        med = float(np.median(vals))
        print(f"  {thr:>6} {med:>8.2f} {med - truth:>+9.2f}")
