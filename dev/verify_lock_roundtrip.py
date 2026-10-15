# -*- coding: utf-8 -*-
"""验证 overlay F7 校准的往返一致性。

流程：detect_felt 初始化 calib_rect -> 存 felt_lock(相对坐标) -> 读回 -> 分析。
检查读回的框与分析结果是否与自动模式一致（即校准不会引入错位）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import detect  # noqa: E402
import analyze as A  # noqa: E402

for p in ("samples/recognize_12frames/frame_02_raw.png", "samples/recognize_12frames/frame_01_raw.png"):
    fp = ROOT / p
    img = cv2.imread(str(fp))
    H, W = img.shape[:2]

    # 1) detect_felt 初始化（overlay on_calibrate 的第一步）
    felt = detect.detect_felt(img)
    print(f"\n=== {fp.stem} {W}x{H} ===")
    print(f"  detect_felt -> {felt}")

    # 2) 存 felt_lock（相对坐标，overlay._save_felt_lock 的逻辑）
    x, y, w, h = felt
    rel = (x / W, y / H, w / W, h / H)
    print(f"  felt_lock rel = {tuple(round(v,6) for v in rel)}")

    # 3) 读回（overlay._felt_from_lock）
    rx, ry, rw, rh = rel
    back = (int(rx * W), int(ry * H), int(rw * W), int(rh * H))
    print(f"  读回 -> {back}   与初始一致={back == felt}")

    # 4) 两种模式分析，比较结果
    r_auto = A.analyze(img, target_color="red", mode="both", max_cushions=2)
    r_lock = A.analyze(img, target_color="red", mode="both", max_cushions=2, felt_override=back)
    same = (len(r_auto.state.balls) == len(r_lock.state.balls)
            and r_auto.state.felt_rect == r_lock.state.felt_rect)
    print(f"  自动: 球={len(r_auto.state.balls)} felt={r_auto.state.felt_rect} 方案={len(r_auto.plans)}")
    print(f"  锁定: 球={len(r_lock.state.balls)} felt={r_lock.state.felt_rect} 方案={len(r_lock.plans)}")
    print(f"  -> 一致={same}")
    if r_auto.plans and r_lock.plans:
        a1, a2 = r_auto.plans[0].aim_dir, r_lock.plans[0].aim_dir
        import numpy as np
        ang = np.degrees(np.arccos(float(np.clip(np.dot(a1, a2), -1, 1))))
        print(f"  首选杆向夹角差={ang:.4f}°")
