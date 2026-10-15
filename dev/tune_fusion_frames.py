# -*- coding: utf-8 -*-
"""确定多帧融合的最优帧数：在时间预算内最大化鲁棒性收益。

权衡：
  * 帧数↑ → 噪声按 √N 衰减、瞬时伪检更容易被投票剔除；
  * 帧数↑ → 耗时线性增长（首帧含半径扫描 ~1.1s，后续 ~0.26s/帧）。
预算 8s ⇒ 上限约 1 + (8-1.1)/0.26 ≈ 27 帧。但收益会递减。

本脚本实测 1/3/5/7/10 帧的耗时与离散度，给出推荐值。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import capture  # noqa: E402
import detect  # noqa: E402

BUDGET = 8.0


def get_frames(n: int) -> tuple[list, str]:
    info = capture.find_window_rect()
    if info:
        out = []
        for _ in range(n):
            im = capture.print_window_bgr(info["hwnd"])
            if im is not None:
                out.append(im)
            time.sleep(0.08)
        if out:
            return out, "实时"
    fr = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))
    return [cv2.imread(str(f)) for f in fr[:n]], "样本"


def main() -> int:
    max_n = 12
    frames, src = get_frames(max_n)
    print(f"来源: {src}  可用帧: {len(frames)}")
    print()
    print(f"{'帧数':>5} {'球数':>5} {'半径':>8} {'耗时s':>7} {'是否超预算':>10} {'平均帧间离散px':>14}")
    print("-" * 62)

    for n in (1, 3, 5, 7, 10, len(frames)):
        n = min(n, len(frames))
        sub = frames[:n]
        t0 = time.time()
        st = detect.analyze_frames(sub)
        dt = time.time() - t0
        # 离散度：同色最近邻的帧间标准差
        spreads = []
        if n >= 3 and st.balls:
            dets_per = []
            for im in sub:
                s = detect.analyze_frame(im)
                dets_per.append(s.balls)
            base = dets_per[0]
            for b0 in base:
                pts = []
                for dp in dets_per[1:]:
                    cand = [x for x in dp if x.color == b0.color]
                    if not cand:
                        continue
                    j = min(cand, key=lambda x: (x.x - b0.x) ** 2 + (x.y - b0.y) ** 2)
                    pts.append((j.x, j.y))
                if pts:
                    a = np.array(pts)
                    spreads.append(float(np.sqrt(a[:, 0].var() + a[:, 1].var())))
        sp = f"{np.mean(spreads):.3f}" if spreads else "—"
        over = "是!" if dt > BUDGET else "否"
        print(f"{n:>5} {len(st.balls):>5} {st.balls[0].r if st.balls else 0:>8.3f} "
              f"{dt:>7.2f} {over:>10} {sp:>14}")

    print()
    print(f"预算 {BUDGET}s。首帧含半径扫描(~1.1s)，后续每帧复用半径(~0.26s)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
