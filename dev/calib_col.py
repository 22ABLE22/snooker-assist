# -*- coding: utf-8 -*-
"""标定 col（色彩通道斑点分）门限，排除水印/logo 弧线造成的误检。

误检特征：台呢水印「腾讯桌球」的弧形亮线穿过球盘，使
  * disk_fill_ratio 略高（85.8%），与贴库真球（87.7%）重叠 → 覆盖率高不住；
  * 但环形区含大量水印像素 → 环形均值被抬高 → col 很低。
真球（含贴库绿球）与合成纯色球的 col 都明显更高。
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
    ("now", "samples/recognize_12frames/frame_02_raw.png", 19),
    ("win_best", "samples/recognize_12frames/frame_01_raw.png", 19),
    ("f01", "samples/recognize_12frames/frame_01_raw.png", 19),
    ("f05", "samples/recognize_12frames/frame_05_raw.png", 19),
    ("fp_frame", "samples/recognize_12frames/frame_03_raw.png", 19),
    ("demo", "output/demo_input.png", 5),
]

allrows = []
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
    seen, uniq = [], []
    for px, py, sv in sorted(peaks, key=lambda t: -t[2]):
        if any((px - a) ** 2 + (py - b) ** 2 < r_est ** 2 for a, b in seen):
            continue
        seen.append((px, py))
        uniq.append((px, py, sv))

    # 真值位置（用出厂 detect 的结果作参照）
    import detect as D
    st = D.analyze_frame(img)
    ox, oy = fx + 3, fy + 3
    truth = {(round(b.x + st.felt_rect[0] - ox), round(b.y + st.felt_rect[1] - oy)) for b in st.balls}

    for px, py, _sv in uniq:
        fill = D2.disk_fill_ratio(d, px, py, r_est)
        if fill < 0.85:
            continue
        lm, vs = D2._inner_stats(lab, V, px, py, r_est, frac=0.62)
        if lm is None:
            continue
        df = float(np.linalg.norm(lm - felt_lab))
        vv = max(vs, float(vstd_map[py, px]))
        cv = float(col[py, px])
        is_true = any((px - tx) ** 2 + (py - ty) ** 2 < (r_est * 1.3) ** 2 for tx, ty in truth)
        allrows.append((tag, px, py, df, vv, fill, cv, is_true))

print(f"{'图像':>10} {'位置':>14} {'色差':>7} {'Vstd':>7} {'覆盖':>7} {'col':>7}  真值?")
print("-" * 72)
for tag, px, py, df, vv, fill, cv, is_true in allrows:
    print(f"{tag:>10} ({px:>5},{py:>5}) {df:>7.1f} {vv:>7.2f} {fill*100:>6.1f}% {cv:>7.2f}  "
          f"{'真球' if is_true else '<<< 误检'}")

t = np.array([r[6] for r in allrows if r[7]])
f = np.array([r[6] for r in allrows if not r[7]])
print(f"\ncol: 真球 min={t.min():.2f} (n={len(t)})   误检 max={f.max() if len(f) else float('nan'):.2f} (n={len(f)})")
if len(f):
    print(f"     -> 门限可设在 {f.max():.1f} ~ {t.min():.1f} 之间，例如 {max(12.0, f.max()*1.3):.1f}")
