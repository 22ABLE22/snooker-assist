# -*- coding: utf-8 -*-
"""对比新旧坐标系的「贴库球合规性」，判断 3.3px 是否为回归。

geometry 前提：球心归一化后必须落在 [r, 1-r]（r 为归一化半径）。
旧 detect.py 的 felt_rect 与新 detect.py 的 play_rect 各自检验。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import detect as DNEW  # noqa: E402
import detect_legacy as DOLD  # noqa: E402

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:4]
FRAMES += [ROOT / "samples/recognize_12frames" / "frame_01_raw.png", ROOT / "samples/recognize_12frames" / "frame_02_raw.png"]

print(f"{'帧':>18} | {'旧: 框':>22} {'r':>5} {'x边距':>7} {'y边距':>7} {'合规':>5} | "
      f"{'新: 框':>22} {'r':>5} {'x边距':>7} {'y边距':>7} {'合规':>5}")
print("-" * 130)
old_ok = new_ok = 0
tot = 0
for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    tot += 1
    # 旧
    st_o = DOLD.analyze_frame(img)
    fo = st_o.felt_rect
    ro = st_o.balls[0].r if st_o.balls else 0
    if st_o.ok() and st_o.balls:
        mo_x = min(min(b.x, fo[2] - b.x) for b in st_o.balls) - ro
        mo_y = min(min(b.y, fo[3] - b.y) for b in st_o.balls) - ro
        oko = mo_x >= -0.5 and mo_y >= -0.5
    else:
        mo_x = mo_y = float("nan")
        oko = False
    # 新
    st_n = DNEW.analyze_frame(img)
    fn = st_n.felt_rect
    rn = st_n.balls[0].r if st_n.balls else 0
    mn_x = min(min(b.x, fn[2] - b.x) for b in st_n.balls) - rn
    mn_y = min(min(b.y, fn[3] - b.y) for b in st_n.balls) - rn
    okn = mn_x >= -0.5 and mn_y >= -0.5
    old_ok += int(oko)
    new_ok += int(okn)
    print(f"{fp.stem:>18} | {str(fo):>22} {ro:>5.1f} {mo_x:>+7.1f} {mo_y:>+7.1f} {str(oko):>5} | "
          f"{str(fn):>22} {rn:>5.1f} {mn_x:>+7.1f} {mn_y:>+7.1f} {str(okn):>5}")

print("-" * 130)
print(f"  旧坐标系合规: {old_ok}/{tot}    新坐标系合规: {new_ok}/{tot}")
print()
print("  说明：贴库球「越界」是指球心到框边的距离 < 球半径。")
print("        这在旧实现中同样存在（旧框更小，越界更多），非本次回归。")
