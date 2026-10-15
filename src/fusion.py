# -*- coding: utf-8 -*-
"""多帧时域融合：把「单帧一次性判断」升级为「多帧投票 + 加权平均」。

动机（系统性弱点）
------------------
现有架构是**开环**的：一次截图 → 一次判断 → 输出。这带来两个结构性问题：
  1. 单帧噪声无法抑制：抗锯齿/压缩让球心有 ~0.45px 抖动，没有第二次观测去平掉它；
  2. 瞬时干扰无法排除：球杆扫过、瞄准圈、发光特效只在个别帧出现，
     单帧判断会把它们当成球（实测白球旁的瞄准圈就曾造成误检）。

时间预算是 8s，而单帧识别约 1.1s —— 有充足余量做多次观测。
本模块把 N 帧的检测结果做**投票 + 加权平均**：
  * 只在 ≥ min_votes 帧中出现的球才算数 → 干掉瞬时伪检；
  * 球心按置信度加权平均 → 噪声按 √N 衰减。

代价优化
--------
半径是游戏渲染的属性、跨帧恒定，所以第 1 帧做完整扫描拿到半径后，
其余帧直接复用（跳过半径扫描），单帧耗时从 1.1s 降到约 0.5s。
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def fuse_detections(
    frames: list[list[dict]],
    min_votes: int | None = None,
    tol: float = 8.0,
) -> list[dict]:
    """融合多帧检测结果。

    frames: 每帧一个列表，元素是含 x, y, r, color, conf 的 dict。
    min_votes: 至少出现在多少帧才算真球；默认 ceil(0.6 * 帧数)。
    tol: 判定「同一颗球」的位置容差（像素）。球局静止，故可取得较紧。
        台球相切时球心距 = 2r ≈ 40px，容差 8px 不会把两颗球并到一起。

    返回：融合后的球列表，附带 votes 字段（出现帧数）。
    """
    if not frames:
        return []
    valid = [f for f in frames if f]
    if not valid:
        return []
    if min_votes is None:
        min_votes = max(2, int(np.ceil(0.6 * len(frames))))

    # 把全部检测摊平，记录来自哪一帧
    dets: list[tuple[int, dict]] = []
    for fi, fr in enumerate(frames):
        for d in fr:
            dets.append((fi, d))
    n = len(dets)
    if n == 0:
        return []

    # 并查集聚类：同一帧内的两个检测不合并（NMS 已保证同帧不重复）
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        fi, di = dets[i]
        for j in range(i + 1, n):
            fj, dj = dets[j]
            if fi == fj:
                continue
            dx = di["x"] - dj["x"]
            dy = di["y"] - dj["y"]
            if dx * dx + dy * dy <= tol * tol:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[dict] = []
    for idxs in groups.values():
        vote_frames = {dets[i][0] for i in idxs}
        if len(vote_frames) < min_votes:
            continue
        xs = np.array([dets[i][1]["x"] for i in idxs], dtype=float)
        ys = np.array([dets[i][1]["y"] for i in idxs], dtype=float)
        rs = np.array([dets[i][1].get("r", 0.0) for i in idxs], dtype=float)
        cs = [dets[i][1].get("color", "unknown") for i in idxs]
        cfs = np.array([dets[i][1].get("conf", 1.0) for i in idxs], dtype=float)
        w = np.maximum(cfs, 1e-6)
        out.append(
            {
                "x": float(np.average(xs, weights=w)),
                "y": float(np.average(ys, weights=w)),
                "r": float(np.median(rs)) if len(rs) else 0.0,
                "color": Counter(cs).most_common(1)[0][0],
                "conf": float(np.mean(cfs)),
                "votes": len(vote_frames),
                # 诊断：该球在帧间的离散度（越小越可信）
                "spread": float(np.sqrt(xs.var() + ys.var())),
            }
        )

    out.sort(key=lambda b: (b["y"], b["x"]))
    return out


def fuse_summary(frames: list[list[dict]], fused: list[dict]) -> dict:
    """给出融合前后的对比诊断，便于判断融合是否真的起作用。"""
    counts = [len(f) for f in frames]
    return {
        "frames": len(frames),
        "per_frame_counts": counts,
        "per_frame_max": max(counts) if counts else 0,
        "fused": len(fused),
        "rejected": (max(counts) if counts else 0) - len(fused),
        "mean_spread": float(np.mean([b["spread"] for b in fused])) if fused else 0.0,
    }
