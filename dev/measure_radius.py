# -*- coding: utf-8 -*-
"""实测球半径：用非台呢掩码的距离变换最大值（球心到边缘 = 半径）。

用于判断是否该沿用 config.BALL_RADIUS_SCALE。
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
    ox, oy = st.crop_rect[0], st.crop_rect[1]

    # 非台呢掩码（宽松阈值，含球的全部像素）
    m = (d > 20).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)

    print(f"\n=== {tag}  检测半径 r={st.balls[0].r if st.balls else 0:.1f} ===")
    print(f"  {'颜色':>6} {'crop坐标':>16} {'掩码内dt最大值':>14} {'半宽(bbox)':>10}")
    radii = []
    for b in st.balls:
        cx, cy = int(round(b.x)), int(round(b.y))
        rad = int(b.r) + 4
        x0, x1 = max(0, cx - rad), min(m.shape[1], cx + rad + 1)
        y0, y1 = max(0, cy - rad), min(m.shape[0], cy + rad + 1)
        sub = dt[y0:y1, x0:x1]
        # 距离变换在该球的峰值 ≈ 半径
        peak = float(sub.max())
        # bbox 半宽：该球连通域
        sub_m = (m[y0:y1, x0:x1] > 0).astype(np.uint8)
        n, _l, stats, _c = cv2.connectedComponentsWithStats(sub_m, connectivity=8)
        # 取包含中心的连通域
        lab_c = _l[cy - y0, cx - x0] if 0 <= cy - y0 < sub_m.shape[0] and 0 <= cx - x0 < sub_m.shape[1] else 0
        halfw = -1
        if lab_c > 0:
            w_ = int(stats[lab_c, cv2.CC_STAT_WIDTH])
            h_ = int(stats[lab_c, cv2.CC_STAT_HEIGHT])
            halfw = (w_ + h_) / 4.0
        radii.append(peak)
        if b.color in ("red", "cue", "black", "brown", "blue", "green"):
            print(f"  {b.color:>6} ({b.x:>6.1f},{b.y:>6.1f}) {peak:>14.2f} {halfw:>10.2f}")
    radii = np.array(radii)
    print(f"  dt 峰值: 中位={np.median(radii):.2f} 均值={radii.mean():.2f} "
          f"min={radii.min():.2f} max={radii.max():.2f}")
    print(f"  -> 检测值 {st.balls[0].r:.1f} / 实测中位 {np.median(radii):.2f} "
          f"= 比值 {st.balls[0].r / np.median(radii):.3f}")
