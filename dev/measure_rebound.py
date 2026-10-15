# -*- coding: utf-8 -*-
"""实测游戏的真实物理：库边反弹规律。

背景
----
`geometry.py` 假设库边是**理想镜面**（入射角 = 反射角），
且已发现游戏的球/台面比例是真实斯诺克的 2 倍 —— 说明它用自己的简化物理，
理想镜面**从未被验证过**。

方法
----
连拍球在库边的运动，逐帧取球心，拟合轨迹：
  * 碰库前若干帧做直线拟合 → 入射方向；
  * 碰库后若干帧做直线拟合 → 反射方向；
  * 比较入射角与反射角（相对库边法线）。

用法
----
  1. 运行本脚本开始连拍；
  2. 在游戏里击球，让球撞库；
  3. 脚本自动找出「方向突变」的帧作为碰库时刻并计算角度。
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = ROOT / "output" / "motion"
OUT.mkdir(parents=True, exist_ok=True)

info = capture.find_window_rect()
if not info:
    print("未找到游戏窗口")
    raise SystemExit(1)
print(f"窗口 {info['width']}x{info['height']}")
print(f"开始连拍 {N} 帧 —— 请在游戏里击球并让球撞库…")

frames: list[np.ndarray] = []
t0 = time.time()
for _ in range(N):
    im = capture.print_window_bgr(info["hwnd"])
    if im is not None:
        frames.append(im)
dt = time.time() - t0
print(f"抓完 {len(frames)} 帧，用时 {dt:.2f}s  ≈ {len(frames)/max(dt,1e-6):.1f} fps")

# 逐帧识别（复用半径）
r_cache: float | None = None
seq: list[list[dict]] = []
felt_cache = None
for i, img in enumerate(frames):
    felt = felt_cache or detect.detect_felt(img)
    if felt is None:
        seq.append([])
        continue
    felt_cache = felt
    st = D2.analyze_frame2(img, felt_override=felt, exp_r=r_cache)
    if r_cache is None and st.balls:
        r_cache = float(np.median([b.r for b in st.balls]))
    seq.append([{"x": b.x, "y": b.y, "color": b.color} for b in st.balls])

counts = [len(s) for s in seq]
print(f"逐帧球数: {counts[:30]}{'...' if len(counts)>30 else ''}")

# 对每颗球做轨迹跟踪（按颜色 + 最近邻）
ref_i = int(np.argmax(counts))
ref = seq[ref_i]
tracks: dict[int, list[tuple[int, float, float]]] = {k: [] for k in range(len(ref))}
for fi, fr in enumerate(seq):
    if not fr:
        continue
    used = set()
    for k, b0 in enumerate(ref):
        cand = [(j, (fr[j]["x"] - b0["x"]) ** 2 + (fr[j]["y"] - b0["y"]) ** 2)
                for j in range(len(fr))
                if j not in used and fr[j]["color"] == b0["color"]]
        if not cand:
            continue
        j, d2 = min(cand, key=lambda t: t[1])
        used.add(j)
        tracks[k].append((fi, fr[j]["x"], fr[j]["y"]))

print()
print("轨迹位移（找在动的球）：")
moving = []
for k, tr in tracks.items():
    if len(tr) < 8:
        continue
    xs = np.array([t[1] for t in tr])
    ys = np.array([t[2] for t in tr])
    span = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    print(f"  #{k} {ref[k]['color']:<7} 样本={len(tr):>3}  最大跨度={span:>7.1f}px")
    if span > 3 * (r_cache or 20):
        moving.append((k, tr))

if not moving:
    print()
    print("结论：画面中没有明显运动的球（球未滚动或已停止）。")
    print("      请运行后在游戏里击球撞库，让球在连拍期间处于滚动状态。")
    raise SystemExit(0)

# 对运动的球找方向突变点（碰库时刻）
print()
print("=" * 78)
print("运动轨迹分析")
print("=" * 78)
for k, tr in moving:
    tr = sorted(tr)
    P = np.array([[t[1], t[2]] for t in tr], dtype=float)
    F = np.array([t[0] for t in tr])
    if len(P) < 8:
        continue
    # 相邻帧速度
    V = np.diff(P, axis=0)
    sp = np.linalg.norm(V, axis=1)
    # 找方向突变：相邻速度向量夹角
    angs = []
    for i in range(len(V) - 1):
        a, b = V[i], V[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1.0 or nb < 1.0:
            angs.append(0.0)
            continue
        c = float(np.dot(a, b) / (na * nb))
        angs.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    if not angs:
        continue
    j = int(np.argmax(angs))
    print(f"\n#{k} {ref[k]['color']}  帧数={len(P)}")
    print(f"  速度: 中位={np.median(sp):.1f}px/帧  max={sp.max():.1f}")
    print(f"  最大方向突变 = {angs[j]:.1f}度 @ 帧{F[j]}→{F[j+1]}→{F[j+2]}")
    # 突变前后各取若干帧做直线拟合
    W = min(5, j, len(P) - j - 2)
    if W >= 3:
        before = P[max(0, j - W):j + 1]
        after = P[j + 1:j + 1 + W]
        def fit_dir(Q):
            vx, vy, x0, y0 = cv2.fitLine(Q.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            return np.array([vx, vy])
        d1 = fit_dir(before)
        d2 = fit_dir(after)
        print(f"  入射方向 = ({d1[0]:+.4f},{d1[1]:+.4f})   反射方向 = ({d2[0]:+.4f},{d2[1]:+.4f})")
        # 判断撞的是哪条库（看突变点位置离哪边近）
        fx, fy, fw, fh = felt_cache
        px, py = P[j + 1]
        d = {"左": px, "右": fw - px, "上": py, "下": fh - py}
        which = min(d, key=d.get)
        print(f"  碰库点 = ({px:.1f},{py:.1f})  最近边 = {which}  距边={d[which]:.1f}px "
              f"(= {d[which]/(r_cache or 20):.2f}r)")
        # 入射/反射角（相对该库法线）
        if which in ("左", "右"):
            n = np.array([1.0, 0.0])
        else:
            n = np.array([0.0, 1.0])
        a1 = np.degrees(np.arccos(np.clip(abs(float(np.dot(d1, n))), -1, 1)))
        a2 = np.degrees(np.arccos(np.clip(abs(float(np.dot(d2, n))), -1, 1)))
        print(f"  入射角(相对法线) = {a1:.2f}度   反射角 = {a2:.2f}度")
        print(f"  >>> 理想镜面应为相等；实测差 = {a2 - a1:+.2f}度")
