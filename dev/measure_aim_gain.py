# -*- coding: utf-8 -*-
"""公平量化袋口精度对瞄准的影响：**同一颗目标球、同一个袋**，只变袋口位置。

上一版直接比「首选方案」，但袋口一变可能选中不同目标球，
149° 的差异其实来自换了目标，不是袋口精度。本版逐球逐袋对比，
隔离出袋口位置本身带来的角度误差。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402
import geometry  # noqa: E402

FRAMES = [ROOT / "samples" / "evidence" / "live_v2.png"]
FRAMES += sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:3]

# 修复前的真实行为：袋口 = 旧 felt_rect 的四角与长边中点
# 旧 felt_rect 比新的小一圈（(156,225,1271,660) vs (149,218,1286,674)）
OLD_FELT = (156, 225, 1271, 660)


def ghost_aim(cue, target, pocket, r_norm, aspect):
    """目标球被击向 pocket 时，白球应瞄准的方向（单位向量）。

    标准做法：幽灵球中心 = 目标球心 + r * (从袋指向目标球的单位向量)。
    """
    tv = np.array(pocket) - np.array(target)
    n = np.linalg.norm(tv)
    if n < 1e-9:
        return None
    # 物理坐标下算方向（消除各向异性影响）
    tv_phys = np.array([tv[0] * aspect, tv[1]])
    n_phys = np.linalg.norm(tv_phys)
    if n_phys < 1e-9:
        return None
    u = tv / n
    ghost = np.array(target) - u * r_norm
    av = ghost - np.array(cue)
    na = np.linalg.norm(av)
    if na < 1e-9:
        return None
    return av / na


print("=" * 86)
print("同目标球、同袋口序号：袋口位置对幽灵球瞄准方向的影响")
print("=" * 86)

all_diffs = []
for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        continue
    st = detect.analyze_frame(img)
    if not st.ok() or not st.balls:
        continue
    fx, fy, fw, fh = st.felt_rect
    asp = fw / fh
    r = float(np.median([b.r for b in st.balls])) / fh
    cue_b = st.cue_ball()
    if cue_b is None:
        print(f"\n--- {fp.stem}: 无白球，跳过 ---")
        continue
    cue = (cue_b.x / fw, cue_b.y / fh)
    tgts = [b for b in st.balls if b.color == "red"]
    new_pk = list(st.pockets)

    # 旧袋口：以 OLD_FELT 为坐标系，四角 + 长边中点
    ox, oy, ow, oh = OLD_FELT
    old_pk_img = [
        (ox, oy), (ox + ow / 2, oy), (ox + ow, oy),
        (ox, oy + oh), (ox + ow / 2, oy + oh), (ox + ow, oy + oh),
    ]
    # 换算到「当前 felt_rect」的归一化坐标，保证两者在同一坐标系里比
    old_pk = [((px - fx) / fw, (py - fy) / fh) for px, py in old_pk_img]

    print(f"\n--- {fp.stem}  红球={len(tgts)} ---")
    diffs = []
    for b in tgts[:6]:
        t = (b.x / fw, b.y / fh)
        for pi in range(6):
            a_old = ghost_aim(cue, t, old_pk[pi], r, asp)
            a_new = ghost_aim(cue, t, new_pk[pi], r, asp)
            if a_old is None or a_new is None:
                continue
            ang = np.degrees(np.arccos(float(np.clip(np.dot(a_old, a_new), -1, 1))))
            diffs.append((ang, pi, b))
    if diffs:
        diffs.sort(reverse=True)
        angs = np.array([d[0] for d in diffs])
        print(f"  逐球逐袋角度差: n={len(angs)} 中位={np.median(angs):.3f}度 "
              f"均值={angs.mean():.3f}度 max={angs.max():.3f}度")
        worst = diffs[0]
        print(f"  最大差出现在 袋#{worst[1]+1} 目标({fx+worst[2].x:.0f},{fy+worst[2].y:.0f}): "
              f"{worst[0]:.3f}度")
        all_diffs.extend(angs.tolist())

if all_diffs:
    a = np.array(all_diffs)
    print("\n" + "=" * 86)
    print(f"汇总: 样本 {len(a)} 组 中位={np.median(a):.3f}度  均值={a.mean():.3f}度  "
          f"p95={np.percentile(a,95):.3f}度  max={a.max():.3f}度")
    print()
    print("参照：斯诺克袋口宽容度约 ±1 度；超过这个量级就会实际打丢。")
    print(f"  超过 1 度的比例: {(a>1).mean()*100:.1f}%")
