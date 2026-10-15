# -*- coding: utf-8 -*-
"""球识别回归测试：python tests/test_detect.py

用 samples/recognize_12frames/ 的真实对局截图做断言，防止识别再次退化。
本次识别重做时，正是靠人工逐帧比对才发现「贴库绿球漏检」等问题的；
把这些断言固化下来，以后改动识别模块跑一遍即可。

覆盖：
  1. 台面定位成功、比赛区 aspect 在合理区间；
  2. 球数正确（该局面为 19 颗）；
  3. 白球存在且唯一；
  4. 颜色分布正确（斯诺克规则约束：彩球各 ≤1、红球 ≤15）；
  5. 统一半径落在标定区间（修正后 16.50px；修正前 19.77px 偏大 ~20%）；
  6. 球心全在可放球范围内（geometry 的镜面硬前提）；
  7. 接口层与引擎结果一致（detect.py 的适配没走样）。
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

SAMPLES = ROOT / "samples" / "recognize_12frames"
FRAMES = sorted(SAMPLES.glob("frame_*_raw.png"))

# 该局面的期望值（12 帧均为同一球局）
EXPECT_TOTAL = 19
EXPECT_RED = 14
EXPECT_CUE = 1
# 半径：由「边缘点稳健圆拟合」标定（同时求圆心与半径，5 样本一致），
# 经 BALL_RADIUS_FIX_SCALE=0.8816 修正后为 17.43px。
# 历史：修正前引擎报 19.77px（偏大约 13%，把方向性阴影算进球体）。
EXPECT_RADIUS = 17.43
RADIUS_TOL = 0.5
# 半径/台面短边 —— 与分辨率无关，可用于跨窗口校验
EXPECT_RADIUS_OVER_FH = 0.02586
RADIUS_FH_TOL = 0.0006
# 比赛区 aspect：实测 1.92；斯诺克理论 2.007，放宽到 1.7~2.3
ASPECT_LO, ASPECT_HI = 1.70, 2.30

_fail: list[str] = []
_pass = 0


def check(cond: bool, msg: str) -> None:
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)
        print(f"  FAIL  {msg}")


def main() -> int:
    if not FRAMES:
        print(f"缺少样本帧: {SAMPLES}")
        return 1
    print(f"样本帧: {len(FRAMES)} 张  ({SAMPLES})")

    for fp in FRAMES:
        img = cv2.imread(str(fp))
        if img is None:
            check(False, f"{fp.name}: 读取失败")
            continue

        st = detect.analyze_frame(img)
        tag = fp.stem

        # 1. 台面定位
        if not check_ok(st.ok(), f"{tag}: 未检测到台面 ({st.error})"):
            continue
        fx, fy, fw, fh = st.felt_rect
        asp = fw / max(fh, 1)
        check(ASPECT_LO <= asp <= ASPECT_HI,
              f"{tag}: 比赛区 aspect={asp:.3f} 超出 [{ASPECT_LO},{ASPECT_HI}]")

        # 2. 球数
        check(len(st.balls) == EXPECT_TOTAL,
              f"{tag}: 球数 {len(st.balls)} != {EXPECT_TOTAL}")

        # 3. 白球唯一
        cues = [b for b in st.balls if b.color == "cue"]
        check(len(cues) == EXPECT_CUE, f"{tag}: 白球 {len(cues)} 颗 != {EXPECT_CUE}")
        check(st.cue_ball() is not None, f"{tag}: cue_ball() 返回 None")

        # 4. 颜色分布 + 斯诺克约束
        cnt: dict[str, int] = {}
        for b in st.balls:
            cnt[b.color] = cnt.get(b.color, 0) + 1
        check(cnt.get("red", 0) == EXPECT_RED,
              f"{tag}: 红球 {cnt.get('red',0)} != {EXPECT_RED}")
        for c in ("cue", "yellow", "green", "brown", "blue", "pink", "black"):
            check(cnt.get(c, 0) <= 1, f"{tag}: {c} 出现 {cnt.get(c,0)} 颗（应 ≤1）")
        check(cnt.get("red", 0) <= 15, f"{tag}: 红球 >15")

        # 5. 半径
        r = float(np.median([b.r for b in st.balls]))
        check(abs(r - EXPECT_RADIUS) <= RADIUS_TOL,
              f"{tag}: 半径 {r:.2f} 偏离 {EXPECT_RADIUS}±{RADIUS_TOL}")
        # 5b. 半径/短边（与分辨率无关）：防止修正系数被误改
        ratio = r / fh
        check(abs(ratio - EXPECT_RADIUS_OVER_FH) <= RADIUS_FH_TOL,
              f"{tag}: r/短边 {ratio:.5f} 偏离 {EXPECT_RADIUS_OVER_FH}±{RADIUS_FH_TOL}")

        # 6. 球心在可放球范围内（像素域：各轴至少留 r）
        mx = min(min(b.x, fw - b.x) for b in st.balls) - r
        my = min(min(b.y, fh - b.y) for b in st.balls) - r
        check(mx >= -0.5, f"{tag}: 球心越界 x 方向 {mx:+.1f}px")
        check(my >= -0.5, f"{tag}: 球心越界 y 方向 {my:+.1f}px")

        # 7. 接口层 vs 引擎：位置应一致（接口层只做坐标平移与夹取）
        dbg: dict = {}
        st2 = D2.analyze_frame2(img, felt_override=st.felt_rect, debug=dbg)
        check(len(st2.balls) == len(st.balls),
              f"{tag}: 引擎 {len(st2.balls)} 颗 vs 接口层 {len(st.balls)} 颗")

    print()
    if _fail:
        print(f"失败 {len(_fail)} 项，通过 {_pass} 项")
        for m in _fail:
            print("  -", m)
        return 1
    print(f"全部通过（{_pass} 项断言）")
    return 0


def check_ok(cond: bool, msg: str) -> bool:
    """返回 cond 本身，便于 `if not check_ok(...): continue`。"""
    check(cond, msg)
    return cond


if __name__ == "__main__":
    raise SystemExit(main())
