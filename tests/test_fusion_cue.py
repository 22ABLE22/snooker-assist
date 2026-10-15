# -*- coding: utf-8 -*-
"""多帧融合与半径复用的回归测试。

固化的 bug（会被本测试捕获）
--------------------------
`analyze_frames` 做半径复用时，曾把**半径校正后**的值当作
`analyze_frame2(exp_r=...)` 的初值回传。二者语义不同：
    exp_r    = 半径扫描的**初值**      （本帧 21.49）
    球对象的 r = 校正后的**最终值**    （本帧 24.77）
尺度不匹配会让**贴库球漏检**，后果：
    帧0 有白球、帧1~4 无白球 -> 融合投票 1 < min_votes=3
    -> 白球被整体剔除 -> 界面显示「未检测到白球」

本测试用同一帧重复 N 次构成「静止球局」，
断言融合前后球数一致、白球不丢。若半径复用语义再次写错，会立即失败。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402
import detect2 as D2  # noqa: E402

_fail: list[str] = []
_pass = 0


def check(cond: bool, msg: str) -> bool:
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)
    return cond


SAMPLES = [
    ("raw_0.png", "普通局面（1588 窗口）"),
    ("diag_now.png", "白球贴左库（1984 窗口）"),
    ("live_v2.png", "普通局面"),
    ("full_rack.png", "满台"),
]

print("=" * 78)
print("半径复用语义 + 多帧融合回归")
print("=" * 78)

for nm, desc in SAMPLES:
    p = ROOT / "samples" / "evidence" / nm
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    st1 = detect.analyze_frame(img)
    if not st1.ok() or not st1.balls:
        continue
    felt = st1.felt_rect
    dbg: dict = {}
    D2.analyze_frame2(img, felt_override=felt, debug=dbg)
    r_scan = dbg.get("exp_r")
    r_final = dbg.get("r_final")
    n_single = len(st1.balls)
    has_cue = st1.cue_ball() is not None

    print(f"\n--- {nm}  {desc} ---")
    print(f"  单帧: {n_single} 颗  白球={'有' if has_cue else '无'}  "
          f"r_scan={r_scan:.2f}  r_final={r_final:.2f}")

    # (1) 半径复用：用 r_scan 作为 exp_r，结果应与自动扫描一致
    if r_scan:
        st_a = D2.analyze_frame2(img, felt_override=felt, exp_r=r_scan)
        # (2) 反例：用 r_final 作为 exp_r（曾经的错误做法）
        st_b = D2.analyze_frame2(img, felt_override=felt, exp_r=r_final)
        n_a = len(st_a.balls)
        n_b = len(st_b.balls)
        cue_a = any(b.color == "cue" for b in st_a.balls)
        cue_b = any(b.color == "cue" for b in st_b.balls)
        print(f"  exp_r=r_scan({r_scan:.2f})   -> {n_a:>2} 颗  白球={'有' if cue_a else '无'}")
        print(f"  exp_r=r_final({r_final:.2f}) -> {n_b:>2} 颗  白球={'有' if cue_b else '无'}")
        # 复用扫描值必须与自动扫描等价（球数与白球都不变）
        check(n_a == n_single,
              f"{nm}: exp_r=r_scan 球数 {n_a} != 单帧 {n_single}")
        check(cue_a == has_cue,
              f"{nm}: exp_r=r_scan 白球状态变化")

    # (3) 多帧融合（模拟静止球局）：球数与白球必须与单帧一致
    n_frames = 5
    frames = [img] * n_frames
    st_f = detect.analyze_frames(frames)
    n_f = len(st_f.balls)
    cue_f = st_f.cue_ball() is not None
    print(f"  融合({n_frames}帧) -> {n_f:>2} 颗  白球={'有' if cue_f else '无'}")
    check(n_f == n_single,
          f"{nm}: 融合后球数 {n_f} != 单帧 {n_single}（应完全一致，因是同一帧）")
    check(cue_f == has_cue,
          f"{nm}: 融合后白球丢失（这是『检测不到白球』bug 的直接表现）")

    # (4) 融合结果的球心必须落在台面内
    fx, fy, fw, fh = st_f.felt_rect
    out = [b for b in st_f.balls if not (0 <= b.x <= fw and 0 <= b.y <= fh)]
    check(not out, f"{nm}: 融合后有 {len(out)} 颗球心越界")

print()
print("=" * 78)
# 额外：带轻微噪声的多帧，验证融合不会因帧间差异丢球
print("带噪声多帧（模拟真实连拍）")
print("=" * 78)
p = ROOT / "samples" / "evidence" / "diag_now.png"
if p.exists():
    img = cv2.imread(str(p))
    base = detect.analyze_frame(img)
    n_base = len(base.balls)
    cue_base = base.cue_ball() is not None
    rng = np.random.default_rng(12345)
    frames = [img]
    for _ in range(4):
        noise = rng.normal(0, 1.5, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(noisy)
    st_n = detect.analyze_frames(frames)
    print(f"  单帧 {n_base} 颗（白球={'有' if cue_base else '无'}）"
          f"  ->  噪声融合 {len(st_n.balls)} 颗"
          f"（白球={'有' if st_n.cue_ball() else '无'}）")
    check(abs(len(st_n.balls) - n_base) <= 1,
          f"噪声融合球数偏差过大: {len(st_n.balls)} vs {n_base}")
    check(st_n.cue_ball() is not None,
          "噪声融合后白球丢失")

print()
if _fail:
    print(f"失败 {len(_fail)} 项：")
    for m in _fail:
        print("  -", m)
    raise SystemExit(1)
print(f"全部通过（{_pass} 项断言）")
