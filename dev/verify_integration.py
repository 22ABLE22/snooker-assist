# -*- coding: utf-8 -*-
"""接入验证：用全部 12 帧实测走通「detect -> analyze -> 绘制」全链路。

检查项：
  1. 接口兼容：analyze_frame / detect_felt / TableState / Ball 成员齐全；
  2. 检测稳定：球数、半径、颜色分布在 12 帧上是否一致；
  3. 坐标系合规：球心归一化后必须落在 [rn, 1-rn]（geometry 的硬前提）；
  4. 几何可用：pot / escape 两种模式都能给出方案；
  5. 锁定模式：felt_override 与自动模式结果一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402
from analyze import analyze, draw_result  # noqa: E402

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))
print(f"测试帧数: {len(FRAMES)}")

# ---- 1. 接口兼容 ----
need_state = ["balls", "pockets", "felt_rect", "ok", "aspect", "cue_ball", "targets",
              "to_felt_px", "to_norm", "felt_px", "width", "height", "error"]
need_ball = ["x", "y", "r", "color", "score", "pos"]
st0 = detect.TableState(1, 1, (0, 0, 1, 1))
missing = [n for n in need_state if not hasattr(st0, n)]
b0 = detect.Ball(0, 0, 1, "red")
missing_b = [n for n in need_ball if not hasattr(b0, n)]
print(f"\n[1] 接口兼容")
print(f"    TableState 缺: {missing or '无'}")
print(f"    Ball 缺      : {missing_b or '无'}")
print(f"    公开函数: analyze_frame={callable(detect.analyze_frame)} "
      f"detect_felt={callable(detect.detect_felt)} "
      f"draw_detection_preview={callable(detect.draw_detection_preview)} "
      f"get_cloth_name={callable(detect.get_cloth_name)}")

# ---- 逐帧检测 ----
print(f"\n[2] 逐帧检测")
print(f"    {'帧':>16} {'球数':>5} {'半径':>7} {'白球':>6} {'颜色分布':<52} {'坐标合规':>8} {'pot':>5} {'esc':>5}")
rows = []
for fp in FRAMES:
    img = cv2.imread(str(fp))
    if img is None:
        print(f"    {fp.name:>16} 读取失败")
        continue
    st = detect.analyze_frame(img)
    if not st.ok():
        print(f"    {fp.name:>16} 失败: {st.error}")
        continue
    fx, fy, fw, fh = st.felt_rect
    r = st.balls[0].r if st.balls else 0
    # 归一化下半径是各向异性的：rx=r/fw, ry=r/fh（geometry 的约定）。
    # 但「球心可及范围」在像素域是各轴都留 r，故这里按像素比较。
    mx = min(min(b.x, fw - b.x) for b in st.balls) - r
    my = min(min(b.y, fh - b.y) for b in st.balls) - r
    okc = mx >= -0.5 and my >= -0.5
    cnt: dict[str, int] = {}
    for b in st.balls:
        cnt[b.color] = cnt.get(b.color, 0) + 1
    cue = st.cue_ball()
    # 几何可用性
    rp = analyze(img, target_color="red", mode="pot", max_cushions=2)
    re_ = analyze(img, target_color="red", mode="escape", max_cushions=2)
    rows.append((len(st.balls), r, okc, len(rp.plans), len(re_.plans)))
    print(f"    {fp.stem:>16} {len(st.balls):>5} {r:>7.2f} "
          f"{'有' if cue else '无':>6} {str(cnt):<52} {str(okc):>8} "
          f"{len(rp.plans):>5} {len(re_.plans):>5}")

arr = np.array([(n, r, int(o), p, e) for n, r, o, p, e in rows])
print(f"\n    球数: 唯一值={sorted(set(arr[:,0].astype(int)))}")
print(f"    半径: 唯一值={sorted(set(np.round(arr[:,1],2)))}")
print(f"    坐标合规: {arr[:,2].sum()}/{len(arr)} 帧通过")
print(f"    pot 方案数: min={arr[:,3].min()} max={arr[:,3].max()}")
print(f"    escape 方案数: min={arr[:,4].min()} max={arr[:,4].max()}")

# ---- 5. 锁定模式一致性 ----
print(f"\n[3] 锁定模式（overlay F7 路径）与自动模式一致性")
same = 0
for fp in FRAMES[:6]:
    img = cv2.imread(str(fp))
    f = detect.detect_felt(img)
    st_auto = detect.analyze_frame(img)
    st_lock = detect.analyze_frame(img, felt_override=f)
    n_auto = {(round(b.x), round(b.y), b.color) for b in st_auto.balls}
    n_lock = {(round(b.x), round(b.y), b.color) for b in st_lock.balls}
    eq = n_auto == n_lock
    same += int(eq)
    print(f"    {fp.stem:>16} 自动={len(st_auto.balls):>2} 锁定={len(st_lock.balls):>2} "
          f"完全一致={eq}")
print(f"    -> {same}/6 帧一致")

# ---- 输出标注图 ----
out = ROOT / "output/dev" / "adapter_check"
out.mkdir(parents=True, exist_ok=True)
for fp in FRAMES[:3]:
    img = cv2.imread(str(fp))
    res = analyze(img, target_color="red", mode="both", max_cushions=2)
    cv2.imwrite(str(out / f"{fp.stem}_det.png"), detect.draw_detection_preview(img, res.state))
    cv2.imwrite(str(out / f"{fp.stem}_plan.png"), draw_result(img, res, top_k=3))
print(f"\n标注图已存: {out}")
