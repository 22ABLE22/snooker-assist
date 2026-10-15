# -*- coding: utf-8 -*-
"""定位实时帧里的多余「棕球」——斯诺克只有 1 颗棕球，出现 2 颗必有一误检。

同时用斯诺克规则约束做最终校验：彩球各最多 1 颗、红球最多 15 颗。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import capture  # noqa: E402
import bench  # noqa: E402
import detect2 as D2  # noqa: E402
import detect as D  # noqa: E402

w = bench.find_game_window()
img = capture.print_window_bgr(w["hwnd"])
cv2.imwrite(str(ROOT / "output/dev" / "fp_frame.png"), img)

st = D.analyze_frame(img)
fx, fy, fw, fh = st.felt_rect
print(f"felt={st.felt_rect}  n={len(st.balls)}")
print(f"\n{'#':>3} {'颜色':>7} {'全图坐标':>16} {'r':>6} {'conf':>5}")
for i, b in enumerate(st.balls, 1):
    print(f"{i:>3} {b.color:>7} ({fx+b.x:>6.0f},{fy+b.y:>6.0f}) {b.r:>6.2f} {b.score:>5.2f}")

browns = [b for b in st.balls if b.color == "brown"]
print(f"\n棕球 {len(browns)} 颗:")
for b in browns:
    print(f"  ({fx+b.x:.0f},{fy+b.y:.0f}) conf={b.score:.2f}")

# 逐候选看判据
dbg = {}
st2 = D2.analyze_frame2(img, felt_override=st.felt_rect, debug=dbg)
lab = dbg["lab"]; d = dbg["resp"]; vstd_map = dbg["vstd"]
V = cv2.cvtColor(img[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
felt_lab = np.array(dbg["felt_lab"], dtype=np.float32)
r_est = dbg["exp_r"]
h, ww = d.shape[:2]
print(f"\ncrop={d.shape[1]}x{d.shape[0]} r_est={r_est:.2f} 台呢Lab=({felt_lab[0]:.0f},{felt_lab[1]:.0f},{felt_lab[2]:.0f})")

print(f"\n=== 各检出球的判据 ===")
print(f"  {'全图坐标':>16} {'色差':>7} {'Vstd':>7} {'覆盖率':>8} {'col':>7}")
for b in st2.balls:
    icx, icy = int(round(b.x)), int(round(b.y))
    if not (0 <= icx < ww and 0 <= icy < h):
        continue
    lm, vs = D2._inner_stats(lab, V, icx, icy, r_est, frac=0.62)
    df = float(np.linalg.norm(lm - felt_lab))
    fill = D2.disk_fill_ratio(d, icx, icy, r_est)
    print(f"  ({icx+fx:>6},{icy+fy:>6}) {df:>7.1f} {max(vs,float(vstd_map[icy,icx])):>7.2f} "
          f"{fill*100:>7.1f}% {b.cov:>7.1f}  {b.color}")
