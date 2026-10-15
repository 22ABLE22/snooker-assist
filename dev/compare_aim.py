# -*- coding: utf-8 -*-
"""对比新旧检测器喂给几何后的**瞄准结果**差异。

检测准确度最终要体现在杆向上。用同一帧、同一目标、同一模式，
分别用旧 detect_legacy 与新 detect（出厂路径）跑 analyze，比较首选杆向。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect_legacy as DOLD  # noqa: E402
import detect as DNEW  # noqa: E402
import geometry  # noqa: E402


def plans_with(detector, img, target="red", mode="pot"):
    """用指定检测器 + 同一套几何跑一遍。"""
    st = detector.analyze_frame(img)
    if not st.ok() or not st.balls:
        return None, st
    fx, fy, fw, fh = st.felt_rect
    geometry.set_table_aspect(fw / fh if fh else None)
    asp = geometry.get_aspect(fw / fh if fh else None)
    r = float(np.median([b.r for b in st.balls])) / fh
    cue_b = st.cue_ball()
    if cue_b is None:
        return None, st
    cue = (cue_b.x / fw, cue_b.y / fh)
    targets = [b for b in st.balls if b.color == target]
    if not targets:
        return None, st
    tpts = [(b.x / fw, b.y / fh) for b in targets]
    others = [(b.x / fw, b.y / fh) for b in st.balls if b.color != "cue"]
    ocols = [b.color for b in st.balls if b.color != "cue"]
    if mode == "pot":
        plans = geometry.find_pot_paths(
            cue, tpts, st.pockets, target, others, r,
            include_object_bank=True, aspect=asp, other_colors=ocols,
        )
    else:
        plans = geometry.find_escape_paths_ranked(
            cue, tpts, target, others, r, max_cushions=2, pockets=st.pockets, aspect=asp,
        )
    return geometry.sort_plans(plans, asp), st


FRAMES = [ROOT / "samples/recognize_12frames" / "frame_02_raw.png", ROOT / "samples/recognize_12frames" / "frame_01_raw.png"]
FRAMES += sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:3]

for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    print(f"\n=== {fp.stem} ===")
    for mode in ("pot", "escape"):
        po, so = plans_with(DOLD, img, mode=mode)
        pn, sn = plans_with(DNEW, img, mode=mode)
        def desc(pl, st):
            if not pl:
                return "无方案"
            b = pl[0]
            return (f"{b.kind:<14} {'>'.join(b.cushions) or '直接':<10} "
                    f"长={b.length:.3f} 分={b.score:6.1f} 杆向=({b.aim_dir[0]:+.4f},{b.aim_dir[1]:+.4f})")
        print(f"  [{mode}]")
        print(f"    旧: {desc(po, so)}   (球数 {len(so.balls) if so.ok() else 0})")
        print(f"    新: {desc(pn, sn)}   (球数 {len(sn.balls) if sn.ok() else 0})")
        if po and pn:
            ao, an = np.array(po[0].aim_dir), np.array(pn[0].aim_dir)
            dot = float(np.clip(np.dot(ao, an), -1, 1))
            ang = np.degrees(np.arccos(dot))
            print(f"    杆向夹角差: {ang:.3f}°")
