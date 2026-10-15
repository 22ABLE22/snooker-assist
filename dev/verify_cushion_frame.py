# -*- coding: utf-8 -*-
"""稳健的库边面测量：要求「向内连续 N 像素都是台呢」才算真正的台呢起点。

上一版只要求单行/单列的绿色占比达标，会被库边上的绿色高光或
装饰菱形骗到（左右两侧各差 13px 的疑点就来自这里）。
本版要求：从候选位置向内连续 run 个像素都是绿色，才算台呢起点。
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

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:2]
FRAMES += [ROOT / "samples" / "evidence" / "live_v2.png"]

RUN = 40          # 连续台呢长度要求（≈2 个球直径）
NEED = 0.95       # 该段内绿色占比


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (H >= 40) & (H <= 88) & (S > 55) & (V > 40)


def edge_robust(g, guess, run=RUN, need=NEED):
    """返回四条台呢起点的中位估计（对整条边采样，取中位以抗局部干扰）。"""
    gx, gy, gw, gh = guess
    Hh, Ww = g.shape
    tops, bots, lefts, rights = [], [], [], []

    # 垂直方向：沿 x 取多列
    for x in range(gx + int(gw * 0.15), gx + int(gw * 0.85), max(1, gw // 60)):
        col = g[:, x]
        # 从上往下找第一个「连续 run 像素绿色」的位置
        for y in range(max(0, gy - 80), min(Hh - run, gy + 140)):
            if col[y:y + run].mean() >= need:
                tops.append(y)
                break
        for y in range(min(Hh - run - 1, gy + gh + 80), max(run, gy + gh - 140), -1):
            if col[y - run:y].mean() >= need:
                bots.append(y)
                break
    # 水平方向：沿 y 取多行
    for y in range(gy + int(gh * 0.15), gy + int(gh * 0.85), max(1, gh // 40)):
        row = g[y, :]
        for x in range(max(0, gx - 80), min(Ww - run, gx + 140)):
            if row[x:x + run].mean() >= need:
                lefts.append(x)
                break
        for x in range(min(Ww - run - 1, gx + gw + 80), max(run, gx + gw - 140), -1):
            if row[x - run:x].mean() >= need:
                rights.append(x)
                break

    med = lambda a: float(np.median(a)) if a else None
    return med(tops), med(bots), med(lefts), med(rights)


print("=" * 92)
print("稳健库边测量（要求向内连续 %d px 台呢）" % RUN)
print("=" * 92)

for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    bx, by, bw, bh = st.felt_rect
    r = float(np.median([b.r for b in st.balls]))
    guess = D2.find_felt(img)
    g = green_mask(img)
    top, bot, left, right = edge_robust(g, guess)

    print(f"\n--- {fp.stem}  r={r:.2f} ---")
    print(f"  稳健台呢起点 : 上={top} 下={bot} 左={left} 右={right}")
    print(f"  play_rect    : 上={by} 下={by+bh} 左={bx} 右={bx+bw}")
    if None not in (top, bot, left, right):
        w, h = right - left, bot - top
        print(f"  稳健台呢矩形 : ({left:.0f},{top:.0f}) {w:.0f}x{h:.0f}  aspect={w/h:.4f}")
        print(f"  play_rect 尺寸: {bw}x{bh}  aspect={bw/bh:.4f}")
        print(f"  差(play-felt): 左{bx-left:+.0f} 上{by-top:+.0f} "
              f"右{(bx+bw)-right:+.0f} 下{(by+bh)-bot:+.0f}")
        print(f"  -> 归一化为球半径: 左{(bx-left)/r:+.2f}r 上{(by-top)/r:+.2f}r "
              f"右{((bx+bw)-right)/r:+.2f}r 下{((by+bh)-bot)/r:+.2f}r")
