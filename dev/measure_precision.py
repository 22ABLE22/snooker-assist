# -*- coding: utf-8 -*-
"""精度测量：同一球局的 12 帧，检出结果应当完全一致。

这 12 帧是同一布局（mean 几乎相同），因此：
  * 球心在不同帧间的**离散度 = 检测噪声**（越低越好）；
  * 半径、颜色的帧间波动同理。

输出每一项的噪声水平，作为后续精度改进的基线。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))


def detect_all() -> list[list[detect.Ball]]:
    out = []
    for fp in FRAMES:
        img = cv2.imread(str(fp))
        st = detect.analyze_frame(img)
        out.append(st.balls)
    return out


def main() -> int:
    runs = detect_all()
    print(f"帧数: {len(runs)}  每帧球数: {[len(r) for r in runs]}")

    # 以第 1 帧为基准，按颜色 + 最近邻做跨帧配对
    base = runs[0]
    n = len(base)
    # tracks[i] = 每帧里对应基准第 i 颗球的 (x, y, r, color)
    tracks: list[list[tuple[float, float, float, str]]] = [[] for _ in range(n)]
    for balls in runs:
        used = set()
        for i, b0 in enumerate(base):
            # 优先同色且最近的
            cands = [
                (j, (balls[j].x - b0.x) ** 2 + (balls[j].y - b0.y) ** 2)
                for j in range(len(balls))
                if j not in used and balls[j].color == b0.color
            ]
            if not cands:
                cands = [(j, (balls[j].x - b0.x) ** 2 + (balls[j].y - b0.y) ** 2)
                         for j in range(len(balls)) if j not in used]
            if not cands:
                continue
            j, _d = min(cands, key=lambda t: t[1])
            used.add(j)
            bb = balls[j]
            tracks[i].append((bb.x, bb.y, bb.r, bb.color))

    print(f"\n{'#':>3} {'颜色':>7} {'基准坐标':>16} {'中心σ(px)':>10} {'半径σ':>8} {'颜色稳定':>8}")
    print("-" * 66)
    cx_sig, cy_sig, r_sig = [], [], []
    colour_bad = 0
    for i, tr in enumerate(tracks):
        if len(tr) < 2:
            print(f"{i+1:>3} 配对不足")
            continue
        xs = np.array([t[0] for t in tr])
        ys = np.array([t[1] for t in tr])
        rs = np.array([t[2] for t in tr])
        cols = {t[3] for t in tr}
        stable = len(cols) == 1
        if not stable:
            colour_bad += 1
        cx_sig.append(float(xs.std()))
        cy_sig.append(float(ys.std()))
        r_sig.append(float(rs.std()))
        print(f"{i+1:>3} {tr[0][3]:>7} ({xs.mean():>7.2f},{ys.mean():>7.2f}) "
              f"{np.hypot(xs.std(), ys.std()):>10.3f} {rs.std():>8.4f} "
              f"{'是' if stable else str(cols):>8}")

    print("-" * 66)
    print(f"  中心噪声 σ: x={np.mean(cx_sig):.3f}px  y={np.mean(cy_sig):.3f}px  "
          f"合成={np.hypot(np.mean(cx_sig), np.mean(cy_sig)):.3f}px")
    print(f"  中心噪声 max: {np.hypot(np.array(cx_sig), np.array(cy_sig)).max():.3f}px")
    print(f"  半径噪声 σ: 均值={np.mean(r_sig):.4f}px  max={np.max(r_sig):.4f}px")
    print(f"  颜色不稳定球数: {colour_bad}/{n}")

    # 半径的帧间一致性
    all_r = [[b.r for b in balls] for balls in runs]
    print(f"\n  各帧半径中位数: {[round(float(np.median(r)),3) for r in all_r]}")
    print(f"  半径帧间极差: {max(np.median(r) for r in all_r) - min(np.median(r) for r in all_r):.4f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
