# -*- coding: utf-8 -*-
"""球心精度评估：用「合成真值」检验亚像素定位能力。

真实截图没有球心真值，所以这里合成一张可控图：
  已知球心/半径/颜色 → 用当前管线检测 → 比较误差。
这是唯一能给出**绝对**精度的手段（真实帧只能给相对一致性）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402

# 画一张模拟游戏画面：绿色台呢 + 木质库边 + 若干球
W, H = 1588, 959
FELT = (136, 205, 1313, 702)   # 与真实帧一致的台呢框
R = 19.77

COLORS = {
    "cue": (245, 245, 245),
    "red": (44, 44, 200),
    "yellow": (48, 214, 238),
    "green": (52, 150, 36),
    "brown": (32, 74, 126),
    "blue": (180, 78, 44),
    "pink": (196, 158, 236),
    "black": (26, 25, 24),
}


def synth(centers: list[tuple[float, float, str]]) -> np.ndarray:
    """按**真实游戏的几何**合成画面。

    关键：袋心比绿色台呢外框**内缩约 13px**（实测游戏：台呢 bbox (136,205,...)，
    角袋心在 (149,218)）。若把袋洞画在台呢框角点上，袋心会落到框外，
    与真实画面不符，会让基准失真。
    """
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (70, 60, 50)                      # 木质库边
    fx, fy, fw, fh = FELT
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (52, 125, 52), -1)   # 绿台呢
    # 六个袋洞：中心比台呢框内缩 13px，与真实游戏一致。
    # 颜色取 (40,38,36)：真实游戏袋洞周围 V 中位约 48~79，
    # 不是纯黑；若画成纯黑会让暗域阈值判据失真，基准也就不可信。
    inset = 13
    for px, py in ((fx + inset, fy + inset),
                   (fx + fw // 2, fy + inset),
                   (fx + fw - inset, fy + inset),
                   (fx + inset, fy + fh - inset),
                   (fx + fw // 2, fy + fh - inset),
                   (fx + fw - inset, fy + fh - inset)):
        cv2.circle(img, (int(px), int(py)), 26, (40, 38, 36), -1, lineType=cv2.LINE_AA)
    # 球：加一点高光让 Vstd 有值
    for cx, cy, name in centers:
        bgr = COLORS[name]
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(R), bgr, -1, lineType=cv2.LINE_AA)
        hl = (min(255, bgr[0] + 90), min(255, bgr[1] + 90), min(255, bgr[2] + 90))
        cv2.circle(img, (int(round(cx - R * 0.3)), int(round(cy - R * 0.3))),
                   max(2, int(R * 0.28)), hl, -1, lineType=cv2.LINE_AA)
    return img


def main() -> int:
    rng = np.random.default_rng(7)
    fx, fy, fw, fh = FELT
    # 真值：在台面内随机撒 12 颗，避开袋口
    truth = []
    names = ["red"] * 8 + ["cue", "blue", "brown", "black"]
    tries = 0
    while len(truth) < len(names) and tries < 2000:
        tries += 1
        x = rng.uniform(fx + 60, fx + fw - 60)
        y = rng.uniform(fy + 60, fy + fh - 60)
        if any((x - a) ** 2 + (y - b) ** 2 < (2.6 * R) ** 2 for a, b, _c in truth):
            continue
        for px, py in ((fx, fy), (fx + fw / 2, fy), (fx + fw, fy),
                       (fx, fy + fh), (fx + fw / 2, fy + fh), (fx + fw, fy + fh)):
            if (x - px) ** 2 + (y - py) ** 2 < (3.2 * R) ** 2:
                break
        else:
            truth.append((x, y, names[len(truth)]))

    img = synth(truth)
    cv2.imwrite(str(ROOT / "output" / "synthetic.png"), img)
    st = detect.analyze_frame(img)

    print(f"合成真值球数: {len(truth)}   检出: {len(st.balls)}")
    if not st.ok():
        print("检测失败:", st.error)
        return 1
    tfx, tfy, tfw, tfh = st.felt_rect
    print(f"真值台呢框={FELT}   检出框=({tfx},{tfy},{tfw},{tfh})")

    # 匹配
    print(f"\n{'真值':>16} {'颜色':>7} {'检出':>16} {'检出色':>7} {'误差px':>8} {'半径':>7}")
    errs = []
    for gx, gy, name in truth:
        if not st.balls:
            break
        j = min(range(len(st.balls)),
                key=lambda k: (st.balls[k].x + tfx - gx) ** 2 + (st.balls[k].y + tfy - gy) ** 2)
        b = st.balls[j]
        dx, dy = b.x + tfx - gx, b.y + tfy - gy
        e = float(np.hypot(dx, dy))
        errs.append(e)
        print(f"({gx:>6.1f},{gy:>6.1f}) {name:>7} ({b.x+tfx:>6.1f},{b.y+tfy:>6.1f}) "
              f"{b.color:>7} {e:>8.2f} {b.r:>7.2f}")
    if errs:
        a = np.array(errs)
        print(f"\n球心误差: 中位={np.median(a):.2f}px  均值={a.mean():.2f}px  "
              f"max={a.max():.2f}px")
        print(f"  相对球半径: 中位={np.median(a)/R*100:.1f}%  max={a.max()/R*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
