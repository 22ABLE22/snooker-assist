# -*- coding: utf-8 -*-
"""多帧融合实测：抓 N 帧 → 各自检测 → 融合 → 对比。

验证三件事：
  1. 融合是否干掉单帧的瞬时伪检（瞄准圈、球杆扫过、特效）；
  2. 融合后球心离散度是否低于单帧；
  3. 复用半径后，多帧总耗时是否仍在预算内。
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
import detect2 as D2  # noqa: E402
from fusion import fuse_detections, fuse_summary  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25


def grab_frames(n: int, gap: float) -> list[np.ndarray]:
    info = capture.find_window_rect()
    if info:
        out = []
        for i in range(n):
            img = capture.print_window_bgr(info["hwnd"])
            if img is not None:
                out.append(img)
            if i < n - 1:
                time.sleep(gap)
        if out:
            print(f"来源: 实时窗口 {info['width']}x{info['height']}  抓到 {len(out)} 帧")
            return out
    fr = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))
    print(f"来源: 样本帧（未找到游戏窗口） 取 {min(n, len(fr))} 张")
    return [cv2.imread(str(f)) for f in fr[:n]]


def engine_balls(img: np.ndarray, exp_r: float | None):
    """跑引擎，返回 (球列表, felt_rect, 实际用的半径)。"""
    felt = detect.detect_felt(img)
    if felt is None:
        return [], None, exp_r
    dbg: dict = {}
    st = D2.analyze_frame2(img, felt_override=felt, exp_r=exp_r, debug=dbg)
    balls = [
        {"x": b.x, "y": b.y, "r": b.r, "color": b.color, "conf": b.conf}
        for b in st.balls
    ]
    r_used = float(dbg.get("r_final") or dbg.get("exp_r") or 0.0)
    return balls, felt, (exp_r if exp_r else r_used)


def main() -> int:
    frames = grab_frames(N, GAP)
    if not frames:
        print("没有可用帧")
        return 1

    t_all = time.time()
    per_frame: list[list[dict]] = []
    times: list[float] = []
    r_cache: float | None = None

    for i, img in enumerate(frames):
        t0 = time.time()
        balls, felt, r_cache_new = engine_balls(img, r_cache)
        dt = time.time() - t0
        times.append(dt)
        per_frame.append(balls)
        mode = "完整(含半径扫描)" if i == 0 else f"复用半径 r={r_cache:.2f}"
        if r_cache is None and r_cache_new:
            r_cache = r_cache_new
        print(f"  帧{i+1}: {len(balls):>2} 颗  ({mode}, {dt*1000:.0f} ms)")
    t_total = time.time() - t_all

    fused = fuse_detections(per_frame)
    summ = fuse_summary(per_frame, fused)

    print()
    print("=" * 78)
    print("融合结果")
    print("=" * 78)
    print(f"  单帧球数        : {summ['per_frame_counts']}")
    print(f"  单帧最大球数    : {summ['per_frame_max']}")
    print(f"  融合后          : {summ['fused']} 颗  "
          f"(剔除 {summ['rejected']} 个瞬时伪检)")
    print(f"  融合后平均离散  : {summ['mean_spread']:.3f} px")
    print(f"  总耗时          : {t_total:.2f} s   单帧均值 {np.mean(times)*1000:.0f} ms")
    print()
    print(f"  {'#':>3} {'颜色':>7} {'融合坐标':>18} {'r':>7} {'票数':>5} {'离散px':>8} {'conf':>6}")
    for i, b in enumerate(fused, 1):
        print(f"  {i:>3} {b['color']:>7} ({b['x']:>7.1f},{b['y']:>7.1f}) "
              f"{b['r']:>7.2f} {b['votes']:>5} {b['spread']:>8.3f} {b['conf']:>6.2f}")

    if len(frames) > 1:
        base = per_frame[0]
        spreads = []
        for b0 in base:
            pts = []
            for fr in per_frame[1:]:
                cand = [x for x in fr if x["color"] == b0["color"]]
                if not cand:
                    continue
                j = min(cand, key=lambda x: (x["x"] - b0["x"]) ** 2 + (x["y"] - b0["y"]) ** 2)
                pts.append((j["x"], j["y"]))
            if pts:
                arr = np.array(pts)
                spreads.append(float(np.sqrt(arr[:, 0].var() + arr[:, 1].var())))
        if spreads:
            print()
            print(f"  未融合的单帧离散: 中位={np.median(spreads):.3f}px  max={max(spreads):.3f}px")
            print(f"  融合后离散      : 均值={summ['mean_spread']:.3f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
