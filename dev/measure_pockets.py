# -*- coding: utf-8 -*-
"""量化误差来源：袋口位置与球心精度。

瞄准精度取决于三件事：
  1. 球心位置（px）
  2. 球半径（影响幽灵球落点）
  3. 袋口位置（影响目标方向）

其中 (3) 最容易被忽视：geometry 假设袋口在「比赛区四角 + 长边中点」，
但游戏的真实袋口未必在那里。本脚本实测真实袋心与假设位置的偏差。
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

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))

print("=" * 78)
print("袋口位置：实测 vs geometry 假设")
print("=" * 78)
print(f"{'帧':>14} {'实测袋心（比赛区归一化）':<58} {'假设位置'}")
print("-" * 78)

for fp in FRAMES[:3]:
    img = cv2.imread(str(fp))
    st = detect.analyze_frame(img)
    fx, fy, fw, fh = st.felt_rect
    # 实测袋心（引擎内部坐标 -> 图像坐标 -> 比赛区归一化）
    dbg: dict = {}
    D2.analyze_frame2(img, felt_override=st.felt_rect, debug=dbg)
    pockets_crop = dbg["pockets"]
    cx0, cy0 = st.felt_rect[0], st.felt_rect[1]
    # detect.py 的坐标系：crop 原点 = felt_rect 原点（锁定模式）
    meas = [((px - 0) / fw, (py - 0) / fh) for px, py in pockets_crop]
    meas_s = " ".join(f"({a:.3f},{b:.3f})" for a, b in meas)
    print(f"{fp.stem:>14} {meas_s:<58}")

print()
print("geometry 假设（default_pockets / detect.build_pockets）:")
print("  角袋 (0,0) (1,0) (0,1) (1,1)   中袋 (0.5,0) (0.5,1)")
print()

# 逐帧算偏差
print("=" * 78)
print("偏差量化（像素）")
print("=" * 78)
print(f"{'帧':>14} {'角袋偏差(px)':>34} {'中袋偏差(px)':>20}")
print("-" * 78)
for fp in FRAMES[:6]:
    img = cv2.imread(str(fp))
    st = detect.analyze_frame(img)
    fx, fy, fw, fh = st.felt_rect
    dbg = {}
    D2.analyze_frame2(img, felt_override=st.felt_rect, debug=dbg)
    pockets = [(px, py) for px, py in dbg["pockets"]]
    # 把实测袋心分到「角袋」与「中袋」
    corner_assumed = [(0, 0), (fw, 0), (0, fh), (fw, fh)]
    mid_assumed = [(fw / 2, 0), (fw / 2, fh)]
    corner_dev, mid_dev = [], []
    for px, py in pockets:
        dc = min(np.hypot(px - ax, py - ay) for ax, ay in corner_assumed)
        dm = min(np.hypot(px - ax, py - ay) for ax, ay in mid_assumed)
        if dc < dm:
            corner_dev.append(dc)
        else:
            mid_dev.append(dm)
    cs = " ".join(f"{d:5.1f}" for d in sorted(corner_dev))
    ms = " ".join(f"{d:5.1f}" for d in sorted(mid_dev))
    print(f"{fp.stem:>14} {cs:>34} {ms:>20}")

print()
print("说明：偏差 = 实测袋心 到 最近假设位置 的距离（像素）。")
print("      偏差越大，说明用几何假设代替实测袋心，瞄准方向误差越大。")
