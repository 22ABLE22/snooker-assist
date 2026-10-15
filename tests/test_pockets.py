# -*- coding: utf-8 -*-
"""中袋位置回归测试。

背景（本次修的 bug）
-------------------
上一轮为了让中袋更准，把中袋坐标改成「实测袋洞中心」。
但袋洞中心在**库区**、位于比赛区矩形**之外**约 19~22px（≈1 个球半径），
导致：
  * 袋口标注圆有一半跑到台面外（圆半径 33px，偏移 18.6px）；
  * **解球线路指向中袋时指向了台面外的假想点**。

正确做法：中袋 = 台面长边的**中点**（球滚落台面的位置），即 (0.5, 0) / (0.5, 1)。

本测试固化该约束，防止再次退化。
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

FRAMES = sorted((ROOT / "samples" / "recognize_12frames").glob("frame_*_raw.png"))[:3]
FRAMES += [ROOT / "samples" / "evidence" / "raw_0.png",
           ROOT / "samples" / "evidence" / "live_v2.png"]

_fail: list[str] = []
_pass = 0


def check(cond: bool, msg: str) -> bool:
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)
    return cond


print("=" * 78)
print("中袋位置回归")
print("=" * 78)
print(f"{'样本':>18} {'中上袋(归一)':>18} {'中下袋(归一)':>18} {'是否在台面内':>12}")
print("-" * 78)

for fp in FRAMES:
    if not fp.exists():
        continue
    img = cv2.imread(str(fp))
    st = detect.analyze_frame(img)
    if not st.ok():
        print(f"{fp.stem:>18} 识别失败")
        continue
    p = st.pockets
    mid_top, mid_bot = p[1], p[4]
    # 1) 中袋必须是长边中点
    ok_top = abs(mid_top[0] - 0.5) < 1e-6 and abs(mid_top[1] - 0.0) < 1e-6
    ok_bot = abs(mid_bot[0] - 0.5) < 1e-6 and abs(mid_bot[1] - 1.0) < 1e-6
    # 2) 必须在台面矩形内（不越界）
    inside = (0.0 <= mid_top[1] <= 1.0) and (0.0 <= mid_bot[1] <= 1.0)
    check(ok_top, f"{fp.stem}: 中上袋 {mid_top} != (0.5, 0)")
    check(ok_bot, f"{fp.stem}: 中下袋 {mid_bot} != (0.5, 1)")
    check(inside, f"{fp.stem}: 中袋超出台面矩形")
    # 3) 分类正确
    check(geometry._is_middle_pocket(mid_top), f"{fp.stem}: 中上袋未被判为中袋")
    check(geometry._is_middle_pocket(mid_bot), f"{fp.stem}: 中下袋未被判为中袋")
    for i in (0, 2, 3, 5):
        check(geometry._is_corner_pocket(p[i]), f"{fp.stem}: 角袋#{i} 未被判为角袋")

    print(f"{fp.stem:>18} ({mid_top[0]:.4f},{mid_top[1]:+.4f})   "
          f"({mid_bot[0]:.4f},{mid_bot[1]:+.4f})   {str(inside):>12}")

print("-" * 78)
print(f"通过 {_pass} 项断言" + (f"，失败 {len(_fail)} 项" if _fail else "，全部通过"))
for m in _fail:
    print("  -", m)

# 附加：中袋夹角阈值必须存在且为合理正数
print()
print("=" * 78)
print("中袋夹角规则（防止被误删）")
print("=" * 78)
try:
    import config
    th = float(config.MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG)
    print(f"  MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG = {th}")
    check(th > 0, "中袋最小夹角阈值应为正数")
    # 夹角 = 进球线与**长库(x 轴)**的夹角。
    #   浅角：目标球靠近中袋所在的库边、但横向偏离 → 连线几乎平行长库 → 小角度
    #   正对：目标球远离该库边 → 连线接近垂直长库 → 大角度
    # 注意 aspect 会缩放 x 分量，构造用例时按物理坐标想。
    a = 1.9
    tgt_shallow = (0.5 + 0.12, 0.02)   # 紧邻中上袋、横向偏 0.12 → 浅角
    tgt_head = (0.5, 0.5)              # 台面中心 → 接近垂直
    ang_shallow = geometry.middle_pocket_rail_angle_deg(tgt_shallow, (0.5, 0.0), a)
    ang_head = geometry.middle_pocket_rail_angle_deg(tgt_head, (0.5, 0.0), a)
    ok_shallow = geometry.middle_pocket_rail_ok(tgt_shallow, (0.5, 0.0), aspect=a)
    ok_head = geometry.middle_pocket_rail_ok(tgt_head, (0.5, 0.0), aspect=a)
    print(f"  浅角样本{tgt_shallow} 夹角={ang_shallow:.2f}度 -> 允许={ok_shallow}")
    print(f"  正对样本{tgt_head} 夹角={ang_head:.2f}度 -> 允许={ok_head}")
    # 阈值 15°：浅角样本必须低于阈值
    check(ang_shallow < th, f"浅角样本夹角 {ang_shallow:.2f} 应低于阈值 {th}")
    check(not ok_shallow, f"浅角({ang_shallow:.2f}度)应被拒绝，实际允许")
    check(ang_head >= th, f"正对样本夹角 {ang_head:.2f} 应不低于阈值")
    check(ok_head, "正对中袋应被允许")
    # 边界：恰好等于阈值应通过（代码用 >= min_deg - 1e-6）
    check(geometry.middle_pocket_rail_ok((0.5, 0.0), (0.5, 0.0)) is not None, "退化输入不应抛异常")
except Exception as e:
    check(False, f"中袋夹角规则异常: {e}")

print()
if _fail:
    print(f"失败 {len(_fail)} 项")
    raise SystemExit(1)
print(f"全部通过（{_pass} 项断言）")
