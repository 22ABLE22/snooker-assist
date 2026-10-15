# -*- coding: utf-8 -*-
"""实时稳定性测试：连续抓多帧，检查球数与颜色分布是否稳定。

fp_frame 那次多出 1 颗「棕球」是运动模糊瞬时伪影，
需要确认它在多帧采样下的出现频率。
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import capture  # noqa: E402
import bench  # noqa: E402
import detect as D  # noqa: E402

w = bench.find_game_window()
if not w:
    print("未找到可见的「腾讯桌球」窗口 —— 请先打开小程序并保持窗口可见。")
    raise SystemExit(0)
print(f"窗口 hwnd={w['hwnd']}")
N = 15
counts = []
color_sets = []
for i in range(N):
    img = capture.print_window_bgr(w["hwnd"])
    if img is None:
        print(f"  {i+1}: 抓取失败")
        continue
    st = D.analyze_frame(img)
    cnt = {}
    for b in st.balls:
        cnt[b.color] = cnt.get(b.color, 0) + 1
    counts.append(len(st.balls))
    color_sets.append(tuple(sorted(cnt.items())))
    flag = "" if len(st.balls) == 19 else "  <<< 异常"
    print(f"  {i+1:>2}: n={len(st.balls):>3}  {dict(sorted(cnt.items()))}{flag}")
    time.sleep(0.35)

print(f"\n球数分布: {dict(Counter(counts))}")
print(f"颜色组合分布:")
for combo, c in Counter(color_sets).most_common():
    print(f"  {c:>3}x  {dict(combo)}")
