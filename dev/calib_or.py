# -*- coding: utf-8 -*-
"""标定「色差 OR Vstd」判据的安全性。

球有两种成因，任一成立即为球：
  A. 颜色明显不同于台呢（真实球 + 合成纯色球）；
  B. 明暗渐变明显（与台呢同色的球，如绿球）。
台呢噪声应两者都不成立。
先应用 fill>=0.85（实心圆盘）再统计，看噪声残留。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import detect2 as D2  # noqa: E402

CASES = [
    ("真实 now", "samples/recognize_12frames/frame_02_raw.png", 19),
    ("真实 win_best", "samples/recognize_12frames/frame_01_raw.png", 19),
    ("真实 f01", "samples/recognize_12frames/frame_01_raw.png", 19),
    ("合成 demo", "output/demo_input.png", 5),
]

for tag, p, expect in CASES:
    fp = ROOT / p
    if not fp.exists():
        continue
    img = cv2.imread(str(fp))
    fx, fy, fw, fh = D2.find_felt(img)
    crop = img[fy + 3 : fy + fh - 3, fx + 3 : fx + fw - 3]
    h, w = crop.shape[:2]
    d, lab = D2.response_map(crop)
    pockets = D2.detect_pocket_centers(crop)
    play = D2.play_rect_from_pockets(pockets, w, h)
    V = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    r_est, _ = D2.estimate_radius(d, h)
    flat = d.reshape(-1)
    felt_lab = np.median(lab.reshape(-1, 3)[flat <= np.percentile(flat, 40)], axis=0).astype(np.float32)

    col = D2.blob_score(d, r_est, felt_tau=20.0).copy()
    vstd_map = D2.brightness_std_map(crop, r_est).copy()
    for px, py in pockets:
        cv2.circle(col, (int(round(px)), int(round(py))), int(round(r_est * 1.8)), 0.0, -1)
        cv2.circle(vstd_map, (int(round(px)), int(round(py))), int(round(r_est * 1.8)), 0.0, -1)
    px0, py0, pw, ph = play
    mk = np.zeros((h, w), np.uint8)
    mk[int(py0) : int(py0 + ph) + 1, int(px0) : int(px0 + pw) + 1] = 1
    mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    col[mk == 0] = 0.0
    vstd_map[mk == 0] = 0.0

    sep = max(4.0, r_est)
    peaks = D2._local_maxima(col, min_sep=sep, min_val=1.0) + \
            D2._local_maxima(vstd_map, min_sep=sep, min_val=6.0)
    # 去重
    seen, uniq = [], []
    for px, py, sv in sorted(peaks, key=lambda t: -t[2]):
        if any((px - a) ** 2 + (py - b) ** 2 < (r_est) ** 2 for a, b in seen):
            continue
        seen.append((px, py))
        uniq.append((px, py, sv))

    # 采样每个峰：fill, 色差, vstd
    samples = []
    for px, py, _sv in uniq:
        fill = D2.disk_fill_ratio(d, px, py, r_est)
        if fill < 0.85:
            continue
        lm, vs = D2._inner_stats(lab, V, px, py, r_est, frac=0.62)
        if lm is None:
            continue
        df = float(np.linalg.norm(lm - felt_lab))
        vv = max(vs, float(vstd_map[py, px]))
        samples.append((px, py, df, vv, fill))

    print(f"\n=== {tag} (期望 {expect} 颗) ===")
    print(f"  通过 fill>=0.85 的候选: {len(samples)}")
    print(f"  {'crop坐标':>16} {'色差':>7} {'vstd':>7} {'覆盖率':>7}")
    for px, py, df, vv, fill in samples:
        print(f"  ({px:>6},{py:>6}) {df:>7.1f} {vv:>7.2f} {fill*100:>6.1f}%")

    # 试不同 (T1, T2) 组合，OR 逻辑
    print(f"  OR 判据命中数：")
    print(f"    {'T1\\T2':>7} " + " ".join(f"{t2:>5}" for t2 in (3, 5, 8, 12, 16, 20)))
    for t1 in (16, 20, 25, 30):
        row = []
        for t2 in (3, 5, 8, 12, 16, 20):
            n = sum(1 for _px, _py, df, vv, _f in samples if df > t1 or vv > t2)
            row.append(n)
        print(f"    {t1:>7} " + " ".join(f"{c:>5}" for c in row))
