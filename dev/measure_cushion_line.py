# -*- coding: utf-8 -*-
"""直接测量台泥内沿（库边面），与 play_rect 比对并标定偏差。

库边面 = 台呢（绿色）区域的内边界。geometry 的镜面公式 y=r 要求
felt_rect 的 y=0 恰好是库边面，所以必须用台呢内沿而不是袋心外框。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import detect  # noqa: E402
import detect2 as D2  # noqa: E402

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:4]
FRAMES += [ROOT / "samples/recognize_12frames" / "frame_01_raw.png", ROOT / "samples/recognize_12frames" / "frame_02_raw.png"]

for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    st = detect.analyze_frame(img)
    fx, fy, fw, fh = st.felt_rect
    r = st.balls[0].r

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((H >= 40) & (H <= 88) & (S > 55) & (V > 40)).astype(np.uint8)

    # 取 play_rect 中部的一条竖带/横带，逐行统计绿色占比，
    # 找到「从库边进入台呢」的位置
    xa, xb = fx + int(fw * 0.25), fx + int(fw * 0.75)
    ya, yb = fy + int(fh * 0.25), fy + int(fh * 0.75)

    def first_row_with_green(x0, x1, y_start, y_end, step):
        ys = range(y_start, y_end, step)
        for y in ys:
            if green[y, x0:x1].mean() > 0.90:
                return y
        return None

    top = first_row_with_green(xa, xb, 0, img.shape[0], 1)
    # 从下往上
    bot = None
    for y in range(img.shape[0] - 1, -1, -1):
        if green[y, xa:xb].mean() > 0.90:
            bot = y
            break
    left = None
    for x in range(img.shape[1]):
        if green[ya:yb, x].mean() > 0.90:
            left = x
            break
    right = None
    for x in range(img.shape[1] - 1, -1, -1):
        if green[ya:yb, x].mean() > 0.90:
            right = x
            break

    print(f"\n=== {fp.stem} ===")
    print(f"  play_rect 图像 = ({fx},{fy}) .. ({fx+fw},{fy+fh})   r={r:.2f}")
    print(f"  台呢内沿 图像 = ({left},{top}) .. ({right},{bot})")
    if all(v is not None for v in (left, top, right, bot)):
        print(f"  偏差 (台呢内沿 - play_rect): "
              f"左={left-fx:+d} 上={top-fy:+d} 右={right-(fx+fw):+d} 下={bot-(fy+fh):+d}")
        print(f"  台呢内沿尺寸 = {right-left}x{bot-top} aspect={(right-left)/max(bot-top,1):.4f}")
        print(f"  play_rect尺寸 = {fw}x{fh} aspect={fw/fh:.4f}")
        # 贴库球检验：用台呢内沿作坐标系
        ys_ = [b.y for b in st.balls]
        xs_ = [b.x for b in st.balls]
        print(f"  若用台呢内沿: 最小上距={min(ys_)-(top-fy):+.1f}px "
              f"最小左距={min(xs_)-(left-fx):+.1f}px")
