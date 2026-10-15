# -*- coding: utf-8 -*-
"""精测球体半径：把「阴影」正确归入台呢，只量球体本体。

上一版的漏洞
-----------
判据是「非台呢 = 非绿(40~88) 且 S>55 且 **V>40**」。
球的深绿阴影 V 可能 < 40，于是被判成「非台呢」而计入球体，
使测得的半径偏大 —— 这正好会**掩盖**真正的偏差，或造成偏差假象。

修正
----
台呢掩码去掉 V 下限（只靠色相+饱和度），让深绿阴影仍算台呢。
再对多组判据做敏感性分析，看结论是否稳健。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import capture  # noqa: E402
import detect  # noqa: E402


def radial_frac(mask, cx, cy, rmax, n_ang=720):
    """mask=1 表示「台呢」。返回各半径处「非台呢」占比。"""
    H, W = mask.shape
    ths = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    ct, st_ = np.cos(ths), np.sin(ths)
    ks = np.arange(1, rmax)
    frac = np.zeros(len(ks))
    for j, k in enumerate(ks):
        px = np.round(cx + ct * k).astype(int)
        py = np.round(cy + st_ * k).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        frac[j] = np.nan if not ok.any() else 1.0 - mask[py[ok], px[ok]].mean()
    return ks, frac


def r_half(ks, frac):
    v = ~np.isnan(frac)
    ks_, fr_ = ks[v], frac[v]
    for j in range(1, len(fr_)):
        if fr_[j - 1] >= 0.5 > fr_[j]:
            t = (fr_[j - 1] - 0.5) / max(fr_[j - 1] - fr_[j], 1e-9)
            return float(ks_[j - 1] + t * (ks_[j] - ks_[j - 1]))
    return float("nan")


def felt_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        # 旧判据（V>40，会把深绿阴影漏给球）
        "S>55,V>40": ((Hc >= 40) & (Hc <= 88) & (Sc > 55) & (Vc > 40)),
        # 去掉 V 下限
        "S>55,V>0 ": ((Hc >= 40) & (Hc <= 88) & (Sc > 55) & (Vc > 0)),
        # 放宽色相 + 低饱和要求
        "H35-95,S>35": ((Hc >= 35) & (Hc <= 95) & (Sc > 35)),
        # 很宽松：任何偏绿
        "H30-100,S>25": ((Hc >= 30) & (Hc <= 100) & (Sc > 25)),
    }


def measure(img, st, tag):
    fx, fy, fw, fh = st.felt_rect
    r_eng = float(np.median([b.r for b in st.balls]))
    masks = felt_masks(img)
    print(f"\n{'='*76}")
    print(f"{tag}  {img.shape[1]}x{img.shape[0]}  台面 {fw}x{fh}")
    print(f"  引擎 r = {r_eng:.3f}   r/fh = {r_eng/fh:.5f}")
    print("=" * 76)
    print(f"  {'判据':>13} " + " ".join(f"{k:>12}" for k in
          ("cue", "red", "blue", "black", "yellow")))
    print("  " + "-" * 74)
    res = {}
    for mname, m in masks.items():
        row = []
        for color in ("cue", "red", "blue", "black", "yellow"):
            cands = [b for b in st.balls if b.color == color]
            if not cands:
                row.append(float("nan"))
                continue
            b = cands[0]
            ks, frac = radial_frac(m, b.x + fx, b.y + fy, int(r_eng * 2.0))
            row.append(r_half(ks, frac))
        res[mname] = row
        print(f"  {mname:>13} " + " ".join(f"{v:>12.2f}" for v in row))
    print()
    for mname, row in res.items():
        arr = np.array([v for v in row if not np.isnan(v)])
        if len(arr) == 0:
            continue
        print(f"  {mname:>13}: 中位={np.median(arr):>6.2f}  "
              f"/引擎={np.median(arr)/r_eng:>6.4f}  "
              f"相对std={arr.std()/arr.mean()*100:>5.2f}%")
    return r_eng, res


info = capture.find_window_rect()
img = capture.print_window_bgr(info["hwnd"])
st = detect.analyze_frame(img)
if st.ok() and st.balls:
    e1, res1 = measure(img, st, "当前实时帧")

for nm in ("diag_now.png", "raw_0.png", "live_v2.png", "full_rack.png"):
    p = ROOT / "samples" / "evidence" / nm
    if not p.exists():
        continue
    im = cv2.imread(str(p))
    s = detect.analyze_frame(im)
    if s.ok() and s.balls:
        measure(im, s, nm)
