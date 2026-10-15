# -*- coding: utf-8 -*-
"""标定 play_rect 的库边内缩量。

方法：用「贴库球」反推。若某球贴库，则其球心到库边面的距离应恰为 r。
取每帧四条边上最贴库的球，统计「球心到 play_rect 边」与 r 的差，
即为 play_rect 相对真实库边的**外扩量**。
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
FRAMES += [ROOT / "samples/recognize_12frames" / "frame_01_raw.png", ROOT / "samples/recognize_12frames" / "frame_02_raw.png"]

print(f"{'帧':>18} {'r':>6} {'左':>7} {'右':>7} {'上':>7} {'下':>7}")
print("-" * 62)
gaps = {"left": [], "right": [], "top": [], "bottom": []}
for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    fx, fy, fw, fh = st.felt_rect
    r = st.balls[0].r
    g = {
        "left": min(b.x for b in st.balls) - r,
        "right": min(fw - b.x for b in st.balls) - r,
        "top": min(b.y for b in st.balls) - r,
        "bottom": min(fh - b.y for b in st.balls) - r,
    }
    for k in gaps:
        gaps[k].append(g[k])
    print(f"{fp.stem:>18} {r:>6.2f} {g['left']:>+7.1f} {g['right']:>+7.1f} "
          f"{g['top']:>+7.1f} {g['bottom']:>+7.1f}")

print("-" * 62)
for k in gaps:
    a = np.array(gaps[k])
    print(f"  {k:>7}: min={a.min():+.1f} 中位={np.median(a):+.1f} max={a.max():+.1f}")

allg = np.concatenate([np.array(gaps[k]) for k in gaps])
print(f"\n  全部: min={allg.min():+.1f}px 中位={np.median(allg):+.1f}px")
print(f"  -> play_rect 需向内收缩约 {max(0, -allg.min()):.1f}px（按最贴库的球计）")
print(f"  -> 或按中位数收缩 {max(0, -np.median(allg)):.1f}px")

# 用「角袋心」再交叉验证一次
print(f"\n=== 交叉验证：实测角袋心 vs play_rect 四角 ===")
for fp in FRAMES[:3]:
    img = cv2.imread(str(fp))
    st = detect.analyze_frame(img)
    fx, fy, fw, fh = st.felt_rect
    print(f"  {fp.stem}: play_rect=({fx},{fy},{fw},{fh}) 四角=({fx},{fy}) ({fx+fw},{fy}) "
          f"({fx},{fy+fh}) ({fx+fw},{fy+fh})")
