# -*- coding: utf-8 -*-
"""用游戏自绘的瞄准指示环，独立验证 geometry 的库边坐标系。

背景
----
`geometry.py` 假设库边在 felt_rect 边界上（球心到库边 == r）。
这个假设此前**从未被独立验证**——我只能靠像素统计去猜库边在哪，
多次尝试都不收敛。本脚本用游戏自己画的图元做验证。

发现
----
游戏在瞄准时会画：白色路径线 + 一个空心环。
环的几何自洽性指向：**环 = 白球贴到库边时的球心位置**。
判据（H1 成立，H2 不成立）：
  * 到最近 play_rect 边距离 = 0.976r（若为幽灵球则应为 2r，实测 4.6r，排除）
  * 该比值在 3 帧上完全一致（std=0.0000）

推论
----
若环心就是「白球贴库时的球心」，则：
    库边面位置 = 环心 - r
本脚本计算该值与 play_rect 边界的差，即 geometry 坐标系的系统偏差。

局限（必须说明）
----------------
  * 只有 **1 个有效样本**（游戏画面静止，且实测**合成鼠标移动不被响应**：
    游戏是 Chrome 渲染，SetCursorPos 后画面变化仅 0.01%，
    因此无法通过脚本改变瞄准方向来采集多方向样本）。
  * 要采集多样本，需**真人移动鼠标**改变瞄准后各截一帧。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402


def fit_circle(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return float(cx), float(cy), float(np.sqrt(sol[2] + cx * cx + cy * cy))


def extract_ring(img, st):
    """找『尺寸接近球、近正方、低填充』的空心环。"""
    fx, fy, fw, fh = st.felt_rect
    r = float(np.median([b.r for b in st.balls]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    white = (((S < 80) & (V > 150)).astype(np.uint8) * 255)
    roi = np.zeros_like(white)
    roi[fy + 8:fy + fh - 8, fx + 8:fx + fw - 8] = 255
    white = cv2.bitwise_and(white, roi)
    for b in st.balls:
        cv2.circle(white, (int(b.x) + fx, int(b.y) + fy), int(b.r) + 6, 0, -1)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(white, connectivity=8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 40:
            continue
        bx, by, bw, bh = (int(stats[i, 0]), int(stats[i, 1]),
                          int(stats[i, 2]), int(stats[i, 3]))
        fill = a / max(bw * bh, 1)
        # 环：近正方 + 低填充 + 外接框接近球直径
        if not (0.6 <= bw / max(bh, 1) <= 1.7 and fill < 0.62):
            continue
        if not (0.5 * 2 * r <= bw <= 2.0 * 2 * r):
            continue
        ys, xs = np.nonzero(white[by:by + bh, bx:bx + bw])
        pts = np.column_stack([xs + bx, ys + by]).astype(np.float64)
        if len(pts) < 25:
            continue
        cx, cy, rr = fit_circle(pts)
        best = (cx, cy, rr, a)
        break
    return best, r, (fx, fy, fw, fh)


print("=" * 88)
print("用游戏自绘瞄准环验证 geometry 库边坐标系")
print("=" * 88)

rows = []
for nm in ("raw_0.png", "raw_1.png", "raw_2.png"):
    p = ROOT / "samples" / "evidence" / nm
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    ring, r, (fx, fy, fw, fh) = extract_ring(img, st)
    if ring is None:
        print(f"{nm}: 未找到环")
        continue
    rcx, rcy, rr_in, area = ring
    # 环心到四条 play_rect 边的距离
    d = {"左": rcx - fx, "右": (fx + fw) - rcx,
         "上": rcy - fy, "下": (fy + fh) - rcy}
    side = min(d, key=d.get)
    dist = d[side]

    # 关键推论：若环心 = 白球贴库时的球心，则库边面 = 环心 - r
    if side == "左":
        face = rcx - r
    elif side == "右":
        face = rcx + r
    elif side == "上":
        face = rcy - r
    else:
        face = rcy + r

    edge = {"左": fx, "右": fx + fw, "上": fy, "下": fy + fh}[side]
    diff = face - edge

    rows.append((nm, r, side, dist, rcx, rcy, face, edge, diff))
    print(f"\n--- {nm} ---")
    print(f"  球半径 r            = {r:.3f}px")
    print(f"  环心                = ({rcx:.2f},{rcy:.2f})   内沿r={rr_in:.2f}")
    print(f"  环心到最近边({side})  = {dist:.3f}px = {dist/r:.4f} r")
    print(f"  反推库边面 (= 环心 -/+ r) = {face:.2f}px")
    print(f"  play_rect 该边       = {edge}px")
    print(f"  >>> 偏差            = {diff:+.2f}px = {diff/r:+.3f} r")

if rows:
    diffs = np.array([t[8] for t in rows])
    ratios = np.array([t[3] / t[1] for t in rows])
    print()
    print("=" * 88)
    print(f"样本数 = {len(rows)}  （注：画面静止，实为同一样本的多次重复）")
    print(f"『环心到最近边』/ r     : 中位={np.median(ratios):.4f}  std={ratios.std():.4f}")
    print(f"『反推库边面 - play_rect』: 中位={np.median(diffs):+.2f}px  "
          f"= {np.median(diffs)/rows[0][1]:+.3f} r")
    print()
    print("解读：")
    m = float(np.median(diffs))
    r0 = rows[0][1]
    if abs(m) < 1.0:
        print(f"  偏差仅 {abs(m):.2f}px（{abs(m)/r0*100:.1f}% 球半径）——")
        print("  **geometry 的库边坐标系得到游戏引擎自身图元的独立验证**：")
        print("  felt_rect 边界确实就在库边面上，镜面假设成立。")
        print("  这解答了此前『play_rect 是否偏外 0.24~0.77r』的悬疑：答案是**不偏**。")
    else:
        print(f"  偏差 {m:+.2f}px 值得关注，建议按此量修正 felt_rect。")
    print()
    print("局限：")
    print("  * 有效样本仅 1 个方向（游戏画面静止）。")
    print("  * 实测**合成鼠标移动不被响应**（游戏为 Chrome 渲染，")
    print("    SetCursorPos 后画面变化 0.01%），故无法脚本化采集多方向样本。")
    print("  * 要覆盖四个方向，需真人改变瞄向后各截一帧再跑本脚本。")
