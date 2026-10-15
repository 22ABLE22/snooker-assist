# -*- coding: utf-8 -*-
"""验证融合的核心价值：能否剔除「只在个别帧出现」的瞬时伪检。

设计一个受控实验（真实数据无法稳定复现瞬时伪检）：
  用真实帧为底，在其中几帧上人为叠加一个「假球」（模拟球杆扫过/瞄准圈），
  然后分别看：
    * 单帧识别    —— 会不会把假球当成真球；
    * 多帧融合    —— 能否用投票把它剔除。
这是判断融合是否真的有用的**因果证据**，而不是只看球数一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402
from fusion import fuse_detections, fuse_summary  # noqa: E402

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:7]

# 用生产配置的帧数，保证结论与实际行为一致
import config  # noqa: E402

N_PROD = max(2, int(getattr(config, "FUSION_FRAMES", 5)))


def add_fake_ball(img: np.ndarray, cx: int, cy: int, r: int = 20) -> np.ndarray:
    """在台呢上的空白处画一个像球的圆（带高光，以通过 Vstd 证据门）。"""
    out = img.copy()
    cv2.circle(out, (cx, cy), r, (44, 44, 200), -1, lineType=cv2.LINE_AA)
    cv2.circle(out, (cx - r // 3, cy - r // 3), max(2, r // 3),
               (150, 150, 240), -1, lineType=cv2.LINE_AA)
    return out


def main() -> int:
    base = [cv2.imread(str(f)) for f in FRAMES][:N_PROD]
    if not base:
        print("缺少样本帧")
        return 1
    min_votes = max(2, int(np.ceil(0.6 * len(base))))
    print(f"生产配置: FUSION_FRAMES={N_PROD}  实际用 {len(base)} 帧  "
          f"min_votes={min_votes}（≥{min_votes} 帧命中才算球）")
    h, w = base[0].shape[:2]
    st0 = detect.analyze_frame(base[0])
    fx, fy, fw, fh = st0.felt_rect
    # 选一个远离所有真球的空白位置放「假球」
    r0 = float(np.median([b.r for b in st0.balls]))
    cand = None
    for ty in range(int(fy + 3 * r0), int(fy + fh - 3 * r0), int(2 * r0)):
        for tx in range(int(fx + 3 * r0), int(fx + fw - 3 * r0), int(2 * r0)):
            if all((tx - (fx + b.x)) ** 2 + (ty - (fy + b.y)) ** 2 > (4 * r0) ** 2
                   for b in st0.balls):
                cand = (tx, ty)
                break
        if cand:
            break
    if cand is None:
        print("找不到空白位置")
        return 1
    fx_, fy_ = cand
    print(f"真球数={len(st0.balls)}  假球放在图像 ({fx_},{fy_})  半径≈{r0:.0f}")
    print()

    for n_fake in range(1, len(base) + 1):
        print("=" * 72)
        print(f"实验：{len(base)} 帧中有 {n_fake} 帧被注入假球")
        print("=" * 72)
        frames = []
        for i, im in enumerate(base):
            frames.append(add_fake_ball(im, fx_, fy_) if i < n_fake else im)

        # 单帧识别（逐帧看，是否能发现假球）
        single_counts = []
        for i, im in enumerate(frames):
            st = detect.analyze_frame(im)
            single_counts.append(len(st.balls))
        print(f"  单帧球数: {single_counts}")
        print(f"    -> 注入假球的那 {n_fake} 帧比其余多 "
              f"{max(single_counts) - min(single_counts)} 颗（单帧无法分辨真假）")

        # 多帧融合：用 engine 级结果做投票
        import detect2 as D2
        per_frame = []
        r_cache = None
        for im in frames:
            felt = detect.detect_felt(im)
            st2 = D2.analyze_frame2(im, felt_override=felt, exp_r=r_cache)
            if r_cache is None and st2.balls:
                r_cache = float(np.median([b.r for b in st2.balls]))
            per_frame.append([
                {"x": b.x, "y": b.y, "r": b.r, "color": b.color, "conf": b.conf}
                for b in st2.balls
            ])
        fused = fuse_detections(per_frame)
        summ = fuse_summary(per_frame, fused)
        print(f"  融合后球数: {summ['fused']}  (单帧最大 {summ['per_frame_max']})"
              f"  剔除 {summ['rejected']} 个")

        # 检查假球是否被剔除
        fake_kept = [
            b for b in fused
            if (b["x"] - (fx_ - fx)) ** 2 + (b["y"] - (fy_ - fy)) ** 2 < (2 * r0) ** 2
        ]
        truth = 0
        for b in fused:
            near = min(((b["x"] - (b0.x)) ** 2 + (b["y"] - (b0.y)) ** 2)
                       for b0 in st0.balls)
            if near < (2 * r0) ** 2:
                truth += 1
        print(f"  融合结果中的真球数: {truth}/{len(st0.balls)}")
        print(f"  假球是否残留: {'是 <<< 融合未剔除' if fake_kept else '否，已剔除 [OK]'}")
        print()

    print()
    print("=" * 72)
    print("结论")
    print("=" * 72)
    print(f"  生产配置 {len(base)} 帧 / min_votes={min_votes}：")
    print(f"    假球出现 < {min_votes} 帧  -> 被投票剔除（单帧无从分辨）")
    print(f"    假球出现 ≥ {min_votes} 帧  -> 融合也留不住，会漏过")
    print()
    print("  这正是时域融合的价值与边界：它对「偶发/瞬时」干扰有效，")
    print("  对「持续存在」的错误（如稳定的 UI 图元）无效 —— 后者要靠单帧判据解决。")
    print("  若要更强的抗持续性干扰，可提高 FUSION_FRAMES（min_votes 随之提高），")
    print("  代价是耗时线性增长（预算 8s）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
