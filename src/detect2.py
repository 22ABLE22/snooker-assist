# -*- coding: utf-8 -*-
"""球体识别 v2：颜色无关候选 + 等半径圆盘匹配滤波。

设计动机（相对旧 detect.py 的根本区别）
--------------------------------------
旧方案对 8 种颜色分别做 HSV 阈值切分，病根：
  * 阈值与台呢皮肤、光照强耦合，换皮肤即崩；
  * 绿球在绿台呢上、红球在红台呢上直接消失；
  * 球互相接触时连通域合并，只能算出一颗球；
  * 阴影/高光拉偏质心，半径靠外接矩形估，误差大。

新方案候选生成完全不看颜色，只用「与局部台呢的感知色差」：
  1. Lab 空间降采样中值估计局部台呢背景 → 感知色彩距离响应 d；
  2. d 阈值化得「非台呢」掩码（含球、库边、瞄准线、水印等一切异物）；
  3. **关键一步**：所有球半径相同 → 用半径 r 的圆盘对掩码做匹配滤波，
     得到「圆盘覆盖率」图。真球 ≈1.0，细长结构（库边/线）必然很低，
     于是无需任何形状启发式即可剔除噪声；
  4. 覆盖率的局部极大值 = 球心；再用掩码加权质心做亚像素精修；
  5. 颜色分类：内盘中位 Lab 对参考调色板做最近邻 + 置信度。

半径 r 由数据自身估计（连通域等效半径的众数），不写死。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------- 参考调色板（BGR，可被 calibrate 覆盖）----------
PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    "cue": (246, 246, 244),
    "red": (44, 44, 200),
    "yellow": (48, 214, 238),
    "green": (52, 150, 36),
    "brown": (32, 74, 126),
    "blue": (180, 78, 44),
    "pink": (196, 158, 236),
    "black": (26, 25, 24),
}

_PALETTE_LAB: dict[str, np.ndarray] = {}


def palette_lab() -> dict[str, np.ndarray]:
    global _PALETTE_LAB
    if not _PALETTE_LAB:
        for name, bgr in PALETTE_BGR.items():
            px = np.uint8([[list(bgr)]])
            _PALETTE_LAB[name] = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return _PALETTE_LAB


def bgr_to_lab(bgr) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def lab_to_bgr(lab) -> tuple[int, int, int]:
    b, g, r = cv2.cvtColor(np.uint8([[list(np.round(lab))]]), cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


@dataclass
class Ball2:
    x: float          # 台呢裁剪内像素坐标
    y: float
    r: float
    color: str
    conf: float = 0.5
    lab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cov: float = 0.0  # 圆盘覆盖率

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class State2:
    width: int
    height: int
    felt_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    crop_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    balls: list[Ball2] = field(default_factory=list)
    error: str = ""
    diag: dict = field(default_factory=dict)

    def ok(self) -> bool:
        _, _, w, h = self.felt_rect
        return w > 0 and h > 0 and not self.error

    def cue_ball(self) -> Ball2 | None:
        c = [b for b in self.balls if b.color == "cue"]
        return max(c, key=lambda b: b.conf) if c else None


# ---------------------------------------------------------------- 台呢定位
def find_felt(
    img_bgr: np.ndarray, hue_lo: int = 40, hue_hi: int = 88, min_s: int = 55
) -> tuple[int, int, int, int] | None:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((H >= hue_lo) & (H <= hue_hi) & (S > min_s) & (V > 40)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w < 120 or h < 80:
        return None
    return int(x), int(y), int(w), int(h)


# ------------------------------------------------- 局部台呢背景（Lab）
def estimate_background_lab(lab: np.ndarray, down: int = 8, ksize: int = 15) -> np.ndarray:
    """降采样中值估计局部背景：比全分辨率 medianBlur 快一个量级。"""
    h, w = lab.shape[:2]
    sw, sh = max(8, w // down), max(8, h // down)
    small = cv2.resize(lab, (sw, sh), interpolation=cv2.INTER_AREA)
    k = min(ksize, max(3, (min(sw, sh) // 3) | 1))
    if k % 2 == 0:
        k += 1
    chans = [cv2.medianBlur(small[:, :, c], k) for c in range(3)]
    bg_small = cv2.merge(chans)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def response_map(
    crop_bgr: np.ndarray, down: int = 8, ksize: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (Lab 感知距离响应 float32, Lab 图)。"""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    bg = estimate_background_lab(lab, down=down, ksize=ksize)
    diff = lab.astype(np.float32) - bg.astype(np.float32)
    d = np.sqrt((diff ** 2).sum(axis=2))
    return d, lab


def disk_annulus_kernels(r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (disk 归一化核, annulus 归一化核, 有效半径)。

    score = mean(disk) - mean(annulus) 是经典斑点检测器：
      * 紧凑圆斑 → disk 内高、annulus 内低 → score 高；
      * 细长线（库边/瞄准线/球杆）→ disk 被稀释 → score 低；
      * 台呢纹理噪声 → 两者都低 → score≈0。
    天然解决「线状物误检」和「台呢渐变」，且不需要二值阈值。
    """
    rr = max(1.5, float(r))
    ri = int(round(rr))
    ro = int(round(rr * 1.75))
    size = 2 * ro + 1
    disk = np.zeros((size, size), np.float32)
    cv2.circle(disk, (ro, ro), ri, 1.0, -1, lineType=cv2.LINE_8)
    ring = np.zeros((size, size), np.float32)
    cv2.circle(ring, (ro, ro), ro, 1.0, -1, lineType=cv2.LINE_8)
    ann = np.clip(ring - disk, 0, 1)
    ds, as_ = float(disk.sum()), float(ann.sum())
    if ds <= 0:
        disk[ro, ro] = 1.0
        ds = 1.0
    if as_ <= 0:
        ann[:] = 0.0
        ann[0, 0] = 1.0
        as_ = 1.0
    return disk / ds, ann / as_, rr


def brightness_std_map(crop_bgr: np.ndarray, r: float) -> np.ndarray:
    """局部亮度标准差图（明暗梯度通道）。

    球是球体，必有高光与明暗渐变 → Vstd 高（实测真球 27~40）；
    台呢是平的 → Vstd≈0.4。这条通道能发现「与台呢同色」的球
    （绿球在绿台呢上、红球在红台呢上），弥补纯色彩通道的盲区。
    """
    V = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    kd, _ka, _ = disk_annulus_kernels(r)
    m1 = cv2.filter2D(V, -1, kd, borderType=cv2.BORDER_REPLICATE)
    m2 = cv2.filter2D(V * V, -1, kd, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def joint_score(col: np.ndarray, vstd: np.ndarray) -> np.ndarray:
    """颜色通道 × 明暗通道的几何平均。

    单通道都无法把贴库绿球与台呢纹理分开（球 min 30.5 < 台呢 max 56.1），
    但真球在两通道**同时**显著非零，台呢噪声是随机的、极少同时高。
    几何平均对「一个通道高、另一个低」的情形惩罚最重，实测最稳。
    """
    return np.sqrt(np.clip(col, 0.0, None) * np.clip(vstd, 0.0, None))


def disk_fill_ratio(
    d: np.ndarray, cx: float, cy: float, r: float, thr: float = 25.0, frac: float = 0.95
) -> float:
    """球盘内「非台呢」像素占比 —— 区分真球与库边/高光的决定性判据。

    实测（19 颗真球 vs 8 个库边/高光误检）：
      * 真球盘内 d>25 占比 = 100.0%（全部 19 颗无一例外）；
      * 库边/高光误检 = 43.6% ~ 75.4%（中位 ~66%）。
    物理含义：真球是实心圆盘，整个球盘都偏离台呢；
    库边是长条，只从球盘一侧切过；高光是球面上一小块亮斑。
    用 frac=0.95 取盘内 95% 半径区域，避开边缘抗锯齿。
    """
    h, w = d.shape[:2]
    rad = max(2, int(round(r * frac)))
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad + 1)
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * frac) ** 2
    n = int(sel.sum())
    if n < 20:
        return 0.0
    return float((d[y0:y1, x0:x1][sel] > thr).mean())


def ball_evidence(
    lab_med: np.ndarray,
    felt_lab: np.ndarray | None,
    vstd: float,
    *,
    min_felt_dist: float = 20.0,
    min_vstd: float = 12.0,
    min_conf: float = 0.55,
) -> tuple[bool, float]:
    """判定「这确实是一颗球」的证据（**或**关系）+ 置信度门限。

    球偏离台呢只有两种成因，任一成立即可：
      A. 颜色明显不同于台呢（色差 > 20）—— 绝大多数球；
      B. 明暗渐变明显（Vstd > 12）—— 与台呢同色的球（绿球），
         以及合成/纯色填充的球（Vstd 近 0，靠 A 通过）。

    为什么用「或」而不是「与」：
      真实游戏球**两个特征都强**（色差 88~120，Vstd 34~58），
      所以 AND 在真实截图上看起来正常；但合成 demo 是纯色填充球
      （色差 101~150，Vstd 0~12），AND 会把它们全部误杀。
      实测在 `disk_fill_ratio >= 0.85` 之后，「或」在真实帧 19/19、
      合成 5/5 均正确。

    `min_conf` 用于排除台呢水印「腾讯桌球」的弧形亮线：
      弧线穿过球盘时覆盖率能勉强过 0.85，但其颜色与台呢接近、
      明暗渐变也弱，两个裕度都小 → 置信度低。
      实测该误检 conf=0.43，而全部真球 conf=0.90~0.99，界限清晰。
    """
    if felt_lab is None:
        return True, 0.0
    d_felt = float(np.linalg.norm(lab_med - felt_lab))
    ok = (d_felt > min_felt_dist) or (vstd > min_vstd)
    # 置信度：取两者中较大的裕度（任一成立即给分）
    m1 = d_felt / max(min_felt_dist, 1e-6)
    m2 = vstd / max(min_vstd, 1e-6)
    margin = max(m1, m2)
    conf = float(np.clip((margin - 1.0) / 1.5, 0.0, 1.0))
    if ok and conf < min_conf:
        ok = False
    return ok, conf


def blob_score(d: np.ndarray, r: float, felt_tau: float = 0.0) -> np.ndarray:
    """圆盘-环形匹配滤波响应。

    felt_tau<=0（默认）：朴素 `mean(disk) - mean(annulus)`。
        实测其 top-20 峰值和在 r=18 明确见顶，**半径扫描专用**。
    felt_tau>0：环形均值只统计「像台呢」的像素（权重 exp(-d/tau)）。
        朴素写法对**贴库球**失效：球紧挨库边时环形区含库边（同样高响应），
        把环形均值抬高，得分被压到噪声水平（实测真绿球 18.9 vs 真球 57~81），
        导致漏检。加权后库边像素 w≈0 自动排除。
    """
    kd, ka, _ = disk_annulus_kernels(r)
    dd = d.astype(np.float32)
    inner = cv2.filter2D(dd, -1, kd, borderType=cv2.BORDER_REPLICATE)
    if felt_tau <= 0:
        outer = cv2.filter2D(dd, -1, ka, borderType=cv2.BORDER_REPLICATE)
        return inner - outer
    w = np.exp(-np.clip(dd, 0.0, None) / max(1e-6, felt_tau)).astype(np.float32)
    num = cv2.filter2D(dd * w, -1, ka, borderType=cv2.BORDER_REPLICATE)
    den = cv2.filter2D(w, -1, ka, borderType=cv2.BORDER_REPLICATE)
    return inner - num / np.maximum(den, 1e-6)


def refine_radius_by_distance_transform(
    d: np.ndarray,
    balls: list[Ball2],
    r_guess: float,
    mask_thr: float = 11.0,
    lo_ratio: float = 0.75,
    hi_ratio: float = 1.35,
) -> tuple[float, list[float]]:
    """用距离变换直接量出球半径，修正匹配滤波的系统性低估。

    匹配滤波扫描给出的半径偏小（实测 17~18px），而真值约 19.8px：
    判据峰值会被拉到偏小的半径上。半径偏小会直接歪掉几何
    （幽灵球位置、库边内移量、走廊宽度），故必须校正。

    距离变换是**直接几何测量**：球心处 dt 值 = 球心到最近非台呢像素距离
    = 球半径。掩码阈值经标定：以「过球心弦长」为真值比对，
    阈值 10~12 时误差 ≤0.02px（阈值 20 会低估 0.4~0.6px，
    因为球边缘的抗锯齿落在 10~20 之间，被切掉了）。
    """
    if not balls:
        return float(r_guess), []
    m = (d > mask_thr).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    h, w = dt.shape[:2]
    vals: list[float] = []
    for b in balls:
        cx, cy = int(round(b.x)), int(round(b.y))
        rad = int(max(4, r_guess * 1.6))
        x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
        y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = dt[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        peak = float(sub.max())
        # 只接受落在合理区间的测量，避免把粘连团块或库边算进来
        if r_guess * lo_ratio <= peak <= r_guess * hi_ratio:
            vals.append(peak)
    if len(vals) < 3:
        return float(r_guess), vals
    return float(np.median(vals)), vals


def estimate_radius(
    d: np.ndarray, felt_h: int, crop_bgr: np.ndarray | None = None
) -> tuple[float, dict]:
    """用数据自身估计统一球半径（粗到精，兼顾速度与精度）。

    判据：正确半径下，圆盘-环形匹配滤波会让「最亮的一批峰」集体达到最强。
    实测 r=18 处 p99.9 与 top-20 峰值和同时见顶（真球 r_eq≈18.7），
    而 gain/p99 之类相对判据会被台呢纹理噪声带偏，故弃用。

    性能：全分辨率逐半径扫描约 720ms，是整条流水线的瓶颈。
    改为**半分辨率粗扫 + 全分辨率精修**：粗扫把候选缩到 3 个，
    精修只需 3 次全分辨率滤波，总耗时降到约 1/5，结果与全扫一致。

    crop_bgr 给定时，粗扫阶段额外纳入联合图（颜色×明暗），
    使「与台呢同色的球」也参与判据。
    """
    lo = max(4, int(felt_h * 0.010))
    hi = max(lo + 2, int(felt_h * 0.045))
    info: dict = {"range": [lo, hi], "scan": []}

    # ---- 粗扫：1/4 分辨率 ----
    # 半径判据只需要「哪个半径附近最强」，1/4 分辨率足够定位峰值位置；
    # 精度由随后的全分辨率精修保证。1/4 比 1/2 又快约 3 倍。
    small = cv2.resize(d, (max(8, d.shape[1] // 4), max(8, d.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    small_bgr = None
    if crop_bgr is not None:
        small_bgr = cv2.resize(crop_bgr, (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_AREA)
    r_lo = max(2, lo // 4)
    r_hi = max(r_lo + 1, hi // 4)
    scored: list[tuple[float, int]] = []
    for rs in range(r_lo, r_hi + 1):
        if small_bgr is not None:
            s = joint_score(blob_score(small, float(rs)),
                            brightness_std_map(small_bgr, float(rs)))
        else:
            s = blob_score(small, float(rs))
        p999 = float(np.percentile(s, 99.9))
        top20 = float(np.sort(s.reshape(-1))[::-1][:20].sum())
        scored.append((top20 + p999 * 20.0, rs))
    scored.sort(key=lambda t: -t[0])
    # 取粗扫前 2 名，映射回全分辨率候选半径（±2 覆盖 4 倍缩放的量化误差）
    cands: list[int] = []
    for _v, rs in scored[:2]:
        for rf in range(rs * 4 - 2, rs * 4 + 3):
            if lo <= rf <= hi and rf not in cands:
                cands.append(rf)

    # ---- 精修：全分辨率，只测少量候选 ----
    best_r, best_val = float(cands[0]), -1.0
    for r in cands:
        s = blob_score(d, float(r))
        p999 = float(np.percentile(s, 99.9))
        top20 = float(np.sort(s.reshape(-1))[::-1][:20].sum())
        val = top20 + p999 * 20.0
        info["scan"].append({"r": r, "p999": round(p999, 2), "top20": round(top20, 1)})
        if val > best_val:
            best_val, best_r = val, float(r)

    info["scan"].sort(key=lambda row: row["r"])
    info["best_r"] = best_r
    info["best_val"] = round(best_val, 1)
    return best_r, info


def _local_maxima(img: np.ndarray, min_sep: float, min_val: float) -> list[tuple[int, int, float]]:
    """局部极大值峰。min_sep = 两个峰之间允许的**最小距离**。

    注意：cv2.dilate 只在核**半径**内抑制较低像素，所以要让 min_sep
    真正表达「最小间距」，核边长必须是 2*min_sep+1。
    早期版本用 k=min_sep（半径 min_sep/2），实测 min_sep=20.7 时
    相距 11px 的两个等高峰都能存活，参数名与行为不符（审计 S6）。

    性能：早期用 `np.nonzero(labels == i)` 逐标签扫描，
    复杂度 O(标签数 × 图像面积)，在 1300×700 上单次约 290ms。
    改为「按值降序、每标签只取首个」的向量化写法，复杂度只与峰像素数相关。
    """
    k = int(2 * max(1.0, round(min_sep))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dmax = cv2.dilate(img, kernel)
    pk = ((img >= dmax - 1e-6) & (img >= min_val)).astype(np.uint8)
    n, labels = cv2.connectedComponents(pk, connectivity=8)
    if n <= 1:
        return []

    ys, xs = np.nonzero(labels)
    if len(xs) == 0:
        return []
    lab = labels[ys, xs]
    vals = img[ys, xs]
    # 按值降序；每个标签只保留第一个（即该标签内的最大值位置）
    order = np.argsort(-vals, kind="stable")
    lab_sorted = lab[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = lab_sorted[1:] != lab_sorted[:-1]
    sel = order[first]
    return [(int(xs[i]), int(ys[i]), float(vals[i])) for i in sel]


def _inner_stats(
    lab: np.ndarray,
    vplane: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    frac: float = 0.60,
) -> tuple[np.ndarray | None, float]:
    """返回 (内盘中位 Lab, 内盘 HSV-V 标准差)。

    vplane 必须是 **HSV 的 V 通道**（不是 BGR 的 R 通道）。
    早期版本误用 `bgr[:,:,2]`（即 R），对饱和蓝/绿球 R 几乎恒定，
    导致 Vstd 被低估（实测饱和蓝球 R-std 2.5 vs V-std 15.9，差 6.5 倍），
    证据门会按颜色选择性地误杀球。此处统一用真亮度 V，
    与 brightness_std_map 保持同一物理量。
    """
    h, w = lab.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r + 1))
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * frac) ** 2
    if int(sel.sum()) < 5:
        return None, 0.0
    px_lab = lab[y0:y1, x0:x1][sel]
    px_v = vplane[y0:y1, x0:x1][sel]
    med = np.median(px_lab, axis=0).astype(np.float32)
    vstd = float(np.std(px_v.astype(np.float64)))
    return med, vstd


def classify_lab(lab_med: np.ndarray, felt_lab: np.ndarray | None = None) -> tuple[str, float]:
    """Lab 最近邻分类 + 置信度。

    台呢同色歧义处理：绿调色板到台呢的距离天生很小（实测 25.5，其他色 61~137），
    因此若最佳匹配是 green 且与台呢极近，需要额外证据（Vstd）才认，见调用处。
    """
    pal = palette_lab()
    dists = {n: float(np.linalg.norm(lab_med - v)) for n, v in pal.items()}
    order = sorted(dists.items(), key=lambda kv: kv[1])
    best, d1 = order[0]
    d2 = order[1][1] if len(order) > 1 else d1 + 40.0
    margin = (d2 - d1) / max(d2, 1e-6)
    conf = float(np.clip(margin * 2.5, 0.05, 0.99))
    if felt_lab is not None:
        d_felt = float(np.linalg.norm(lab_med - felt_lab))
        # 与台呢几乎同色 → 只有极强证据才当球
        if d_felt < 12.0:
            conf *= 0.25
    return best, conf


# ------------------------------------------------------------------ 主检测
POCKET_POS = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0))


def detect_pocket_centers(crop_bgr: np.ndarray) -> list[tuple[float, float]]:
    """在 6 个预期袋位附近搜索暗洞，返回实测袋口中心（裁剪坐标）。

    必须实测而非用 felt 矩形角：实测袋心相对矩形角内缩约 1.3 个球半径，
    按矩形角建模会把袋口暗洞漏掉，进而误检成"黑球"。
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    h, w = V.shape[:2]
    win = max(12, int(min(w, h) * 0.07))
    v_lo = float(np.percentile(V, 3))
    thr = max(45.0, v_lo + 18.0)
    centers: list[tuple[float, float]] = []
    for nx, ny in POCKET_POS:
        cx, cy = int(nx * (w - 1)), int(ny * (h - 1))
        x0, x1 = max(0, cx - win), min(w, cx + win + 1)
        y0, y1 = max(0, cy - win), min(h, cy + win + 1)
        sub = V[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        m = (sub < thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            j = max(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
            if int(stats[j, cv2.CC_STAT_AREA]) >= 6:
                centers.append((float(x0 + cents[j][0]), float(y0 + cents[j][1])))
                continue
        # 兜底：取窗口内最暗点
        jj = np.unravel_index(int(np.argmin(sub)), sub.shape)
        centers.append((float(x0 + jj[1]), float(y0 + jj[0])))
    return centers


def play_rect_from_pockets(
    pockets: list[tuple[float, float]], w: int, h: int
) -> tuple[float, float, float, float]:
    """由四角袋心反推「可放球的比赛区」矩形。

    袋心近似位于台呢内沿角点，故四角袋心的外接矩形即球心可及范围。
    实测贴库球会略微「越界」（球心到边距离比半径小约 3px），
    原因是袋心略偏库边外侧。这个偏差**不在此处修正** ——
    若在这里内缩，会连带把贴库球裁出检测范围（实测丢球）。
    偏差改由 detect.py 在输出坐标系时处理，见该文件说明。
    """
    if len(pockets) < 6:
        return (0.0, 0.0, float(w), float(h))
    pts = np.array(pockets, dtype=float)
    corners = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
    sel = []
    for cx, cy in corners:
        j = int(np.argmin(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
        sel.append(pts[j])
    sel = np.array(sel)
    x0, x1 = float(sel[:, 0].min()), float(sel[:, 0].max())
    y0, y1 = float(sel[:, 1].min()), float(sel[:, 1].max())
    if x1 - x0 < w * 0.5 or y1 - y0 < h * 0.5:
        return (0.0, 0.0, float(w), float(h))
    return (x0, y0, x1 - x0, y1 - y0)


def pocket_exclusion_mask(
    shape: tuple[int, int], pockets: list[tuple[float, float]], ball_r: float, ratio: float = 1.9
) -> np.ndarray:
    """袋口附近的排除掩码（用于半径扫描时屏蔽伪响应）。"""
    m = np.zeros(shape, dtype=np.uint8)
    for px, py in pockets:
        cv2.circle(m, (int(round(px)), int(round(py))), int(round(ball_r * ratio)), 255, -1)
    return m


def reject_border_and_pockets(
    balls: list[Ball2],
    w: int,
    h: int,
    r: float,
    play_rect: tuple[float, float, float, float] | None = None,
    pockets: list[tuple[float, float]] | None = None,
) -> list[Ball2]:
    """剔除台边外与袋口处的伪球。

    台呢裁剪含库边与装饰菱形，袋口是暗洞，二者都产生强斑点响应。
    真球球心不会越过比赛区边界，也不会落进袋口。
    """
    keep: list[Ball2] = []
    for b in balls:
        # 1) 比赛区（由袋心反推）
        if play_rect is not None:
            px0, py0, pw, ph = play_rect
            if not (px0 - r * 0.35 <= b.x <= px0 + pw + r * 0.35
                    and py0 - r * 0.35 <= b.y <= py0 + ph + r * 0.35):
                continue
        else:
            lim = r * 1.02
            if b.x < lim or b.y < lim or b.x > w - lim or b.y > h - lim:
                continue
        # 2) 袋口危险区（实测袋心）
        # 实测四角袋心约在 (17,17)/(1289,17)/(17,678)/(1289,678)，
        # 但袋口暗洞与袋唇延伸可达 ~2.2R，故余量取 2.2R。
        if pockets:
            near = False
            for px, py in pockets:
                if (b.x - px) ** 2 + (b.y - py) ** 2 < (r * 2.2) ** 2:
                    near = True
                    break
            if near:
                continue
        keep.append(b)
    return keep


def _gap_threshold(vals: np.ndarray, r_est: float, felt_h: int) -> float:
    """在峰得分排序曲线上找自然断层，同时用斯诺克球数上限做护栏。

    真球峰高且成簇；噪声峰低而密。取相邻峰的最大相对跌落点作为分界。
    注意：本函数只做**粗筛**，真正是否算球由 ball_evidence 的双重证据决定，
    故门槛取得宽松（不高于 0.35 * 最高峰），避免漏掉贴库/暗色球。
    """
    if len(vals) == 0:
        return 1e9
    if len(vals) <= 3:
        return float(vals.min()) * 0.5

    head = vals[:60]
    ratios = head[:-1] / np.maximum(head[1:], 1e-6)
    j = int(np.argmax(ratios))
    gap_thr = float(head[j + 1])

    # 斯诺克最多 22 颗球；给余量到 40，避免把真球截断
    max_balls = 40
    if len(head) > max_balls:
        gap_thr = max(gap_thr, float(head[max_balls]))

    # 粗筛地板：最高峰的 35%。过严会漏掉贴库绿球（实测其 Vstd 60 属高值，
    # 但暗色/被遮挡球可能较低），故宁可宽松，由证据门把关。
    floor = 0.35 * float(vals[0])
    return float(max(gap_thr, floor, 4.0))


def detect_balls_v2(
    crop_bgr: np.ndarray,
    *,
    exp_r: float | None = None,
    score_min: float | None = None,
    play_override: tuple[float, float, float, float] | None = None,
    debug: dict | None = None,
) -> list[Ball2]:
    """斑点检测主流程。不再依赖二值阈值分割球。

    play_override: 若调用方已知比赛区（crop 坐标系），直接使用，不再由袋心反推。
        当调用方传进来的矩形**本身就是比赛区**（例如 overlay 锁定的框）时，
        若仍按袋心反推，会因为裁剪内缩而算出一个更小的矩形，
        把贴近原边界的球裁掉（实测丢 2 颗）。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]

    d, lab = response_map(crop_bgr)
    pockets = detect_pocket_centers(crop_bgr)
    if play_override is not None:
        play_rect = (float(play_override[0]), float(play_override[1]),
                     float(play_override[2]), float(play_override[3]))
    else:
        play_rect = play_rect_from_pockets(pockets, w, h)
    excl = pocket_exclusion_mask((h, w), pockets, max(6.0, h * 0.02), ratio=2.4)

    r_est = float(exp_r) if exp_r else None
    r_info: dict = {}
    if r_est is None:
        # 半径扫描屏蔽袋口：袋口暗洞响应会随半径变化而干扰判据。
        # 扫描用**朴素** blob_score：实测其 top-20 峰值和在 r=18 明确见顶；
        # 若改用联合图，判据会被同色盲区/贴库球带偏（实测误选 r=10）。
        d_scan = d.copy()
        d_scan[excl > 0] = 0.0
        r_est, r_info = estimate_radius(d_scan, h)
    r_est = float(np.clip(r_est, 3.0, min(h, w) * 0.08))

    col = blob_score(d, r_est, felt_tau=20.0)   # 贴库球稳健，用于定位
    vstd_map = brightness_std_map(crop_bgr, r_est)
    # HSV 的 V 通道（真亮度）。必须与 brightness_std_map 用同一物理量，
    # 不能误用 BGR 的 R 通道（见 _inner_stats 注释 / 审计 S1）。
    vplane = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    # 袋口/台外区域一律作废（必须在找峰之前，否则袋口暗洞会生成伪峰）
    col = col.copy()
    vstd_map = vstd_map.copy()
    for px, py in pockets:
        cv2.circle(col, (int(round(px)), int(round(py))), int(round(r_est * 1.8)), 0.0, -1)
        cv2.circle(vstd_map, (int(round(px)), int(round(py))), int(round(r_est * 1.8)), 0.0, -1)
    if play_rect != (0.0, 0.0, float(w), float(h)):
        px0, py0, pw, ph = play_rect
        m = np.zeros((h, w), np.uint8)
        m[int(py0) : int(py0 + ph) + 1, int(px0) : int(px0 + pw) + 1] = 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        col[m == 0] = 0.0
        vstd_map[m == 0] = 0.0

    # ── 候选峰生成：双通道，取长补短 ──────────────────────────────────
    #  * 色彩通道（col）：对「与台呢不同色」的球定位精准（实测 <1.5px）；
    #    但对同色球（绿球）无峰 —— 真绿球 col=41.8，反被紧邻库边伪峰 47.5 压制。
    #  * Vstd 通道：对任何颜色的球都锐利（球 25~85 vs 台呢 0.4），
    #    但会被高光/球杆/水印骗出伪峰。
    # 两路峰全部送入后续**双重门限**裁决（覆盖率 + 颜色/明暗证据），
    # 不再用脆弱的 score_min 硬门槛 —— 实测贴库绿球在色彩通道上
    # 得分天生偏低，任何全局阈值都会把它连同噪声一起切掉。
    #
    # min_sep 传 2r：_local_maxima 的 cv2.dilate 只在**核半径**内抑制，
    # 要让参数真正表达「球心间距至少 2r」，须传 2r（见 S6）。
    sep = max(4.0, r_est * 1.0)
    col_peaks = _local_maxima(col, min_sep=sep, min_val=1.0)
    vstd_peaks = _local_maxima(vstd_map, min_sep=sep, min_val=6.0)

    peaks: list[tuple[int, int, float]] = []
    seen: list[tuple[int, int]] = []
    for px, py, sv in sorted(col_peaks + vstd_peaks, key=lambda p: -p[2]):
        if any((px - tx) ** 2 + (py - ty) ** 2 < (r_est * 1.0) ** 2 for tx, ty in seen):
            continue
        seen.append((px, py))
        peaks.append((px, py, sv))
    score_min = 0.0  # 不再使用；由下方双重门限裁决

    # 台呢参考色（用于颜色分类时压低「与台呢同色」的置信度）
    flat = d.reshape(-1)
    cut = np.percentile(flat, 40)
    felt_px = lab.reshape(-1, 3)[flat <= cut]
    felt_lab = np.median(felt_px, axis=0).astype(np.float32) if len(felt_px) else None

    balls: list[Ball2] = []
    for px, py, sval in peaks:
        lab_med, vstd = _inner_stats(lab, vplane, px, py, r_est, frac=0.62)
        if lab_med is None:
            continue
        # ── 门限一：盘内非台呢覆盖率 ──
        # 真球是实心圆盘，整个盘都偏离台呢；库边只从盘一侧切过，高光是球面小块。
        # 实测分布（含贴库绿球）：真球 88.8% ~ 100.0%，库边/高光误检 43.6% ~ 75.4%。
        # 0.85 落在这条清晰的空隙里。
        fill = disk_fill_ratio(d, px, py, r_est)
        if fill < 0.85:
            continue
        # ── 门限二：色差 + 明暗渐变 ──
        # 兜住覆盖率过线但颜色/亮度异常的候选。Vstd 用与响应图一致的整球尺度。
        vstd_full = float(vstd_map[py, px])
        ok, ev_conf = ball_evidence(lab_med, felt_lab, max(vstd, vstd_full))
        if not ok:
            continue
        color, conf = classify_lab(lab_med, felt_lab)
        conf = float(min(0.99, max(conf, ev_conf * 0.9)))
        # 用局部色彩距离加权精修球心（限幅，防止被库边拖走）
        cx, cy = _refine_by_response(d, px, py, r_est)
        balls.append(
            Ball2(
                x=float(cx), y=float(cy), r=float(r_est), color=color, conf=float(conf),
                lab=(float(lab_med[0]), float(lab_med[1]), float(lab_med[2])), cov=float(sval),
            )
        )

    balls = nms(balls, min_dist=max(4.0, r_est * 1.35))
    balls = reject_border_and_pockets(balls, w, h, r_est, play_rect=play_rect, pockets=pockets)

    # ── 半径校正：匹配滤波系统性低估约 10%，用距离变换直接量出真半径 ──
    # 半径偏小会歪掉幽灵球位置/库边内移/走廊宽度，故必须校正。
    r_dt, r_samples = refine_radius_by_distance_transform(d, balls, r_est)
    if r_samples:
        for b in balls:
            b.r = float(r_dt)

    balls.sort(key=lambda b: (b.y, b.x))

    if debug is not None:
        debug.update(
            {"exp_r": float(r_est), "r_info": r_info, "r_final": float(r_dt),
             "r_dt_samples": [round(v, 2) for v in r_samples],
             "mask": (col > 0).astype(np.uint8),
             "resp": d, "lab": lab, "col": col, "vstd": vstd_map,
             "score_min": float(score_min),
             "pockets": pockets, "play_rect": play_rect,
             "felt_lab": None if felt_lab is None else felt_lab.tolist()}
        )
    return balls


def _refine_by_response(
    d: np.ndarray, cx: float, cy: float, r: float, max_shift: float = 0.35
) -> tuple[float, float]:
    """在半径邻域内用色彩距离加权求质心，抑制阴影/高光偏移。

    必须限制位移上限（max_shift × r）：贴库球旁边就是库边，库边的色彩距离
    远高于球本身，不加限制会把球心拖到库边上（实测真绿球被从 (437,452)
    拖到 (465,457)，偏 37.6px，Vstd 随之从 39.4 掉到 3.3 → 被证据门误杀）。
    """
    h, w = d.shape[:2]
    rad = int(np.ceil(r)) + 2
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad + 1)
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return cx, cy
    sub = d[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 1.15) ** 2
    if int(sel.sum()) < 6:
        return cx, cy
    wts = sub[sel]
    base = float(np.median(wts))
    wts = np.maximum(wts - base * 0.5, 0.0) + 1e-6
    nx = float(np.average(xx[sel], weights=wts))
    ny = float(np.average(yy[sel], weights=wts))
    dx, dy = nx - cx, ny - cy
    dist = float(np.hypot(dx, dy))
    lim = max(1.0, r * max_shift)
    if dist > lim:
        nx = cx + dx / dist * lim
        ny = cy + dy / dist * lim
    return nx, ny


def nms(balls: list[Ball2], min_dist: float) -> list[Ball2]:
    balls = sorted(balls, key=lambda b: -b.cov)
    kept: list[Ball2] = []
    for b in balls:
        if all((b.x - k.x) ** 2 + (b.y - k.y) ** 2 >= min_dist ** 2 for k in kept):
            kept.append(b)
    return kept


# ------------------------------------------------------------------ 顶层
def analyze_frame2(
    img_bgr: np.ndarray,
    felt_override: tuple[int, int, int, int] | None = None,
    *,
    exp_r: float | None = None,
    inset: int = 3,
    debug: dict | None = None,
) -> State2:
    """检测一帧。

    felt_override 语义：**已经是比赛区**（可放球范围，四角即角袋中心）。
        此时不再由袋心反推比赛区，直接以它为坐标系 ——
        因为调用方（如 overlay 的 F7 锁定框）传进来的就是比赛区，
        再反推会因裁剪内缩而算出更小的矩形，裁掉贴边球。
    """
    h, w = img_bgr.shape[:2]
    play_override = None
    if felt_override is not None:
        fx, fy, fw, fh = (int(v) for v in felt_override)
    else:
        fr = find_felt(img_bgr)
        if fr is None:
            return State2(width=w, height=h, error="未检测到台呢")
        fx, fy, fw, fh = fr

    fx = max(0, min(fx, w - 20))
    fy = max(0, min(fy, h - 20))
    fw = max(20, min(fw, w - fx))
    fh = max(20, min(fh, h - fy))

    if felt_override is not None:
        # 锁定模式：crop 就是比赛区本身，坐标系 = crop 原点
        cx0, cy0 = fx, fy
        cx1, cy1 = fx + fw, fy + fh
        play_override = (0.0, 0.0, float(fw), float(fh))
    else:
        cx0, cy0 = fx + inset, fy + inset
        cx1, cy1 = fx + fw - inset, fy + fh - inset

    crop = img_bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return State2(width=w, height=h, felt_rect=(fx, fy, fw, fh), error="台呢裁剪为空")

    balls = detect_balls_v2(
        crop, exp_r=exp_r, play_override=play_override, debug=debug
    )
    return State2(
        width=w, height=h,
        felt_rect=(fx, fy, fw, fh),
        crop_rect=(cx0, cy0, crop.shape[1], crop.shape[0]),
        balls=balls,
    )


# ------------------------------------------------------- 可视化 / 文本视图
_DRAW = {
    "cue": (255, 255, 255), "red": (0, 0, 255), "yellow": (0, 255, 255),
    "green": (0, 200, 0), "brown": (0, 80, 160), "blue": (255, 80, 0),
    "pink": (255, 120, 220), "black": (60, 60, 60), "unknown": (180, 180, 180),
}
_LETTER = {
    "cue": "W", "red": "r", "yellow": "y", "green": "g", "brown": "n",
    "blue": "b", "pink": "m", "black": "K", "unknown": "?",
}


def draw_state2(img_bgr: np.ndarray, st: State2, tag: str = "v2") -> np.ndarray:
    out = img_bgr.copy()
    if not st.ok():
        cv2.putText(out, st.error or "no table", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    fx, fy, fw, fh = st.felt_rect
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
    cx0, cy0 = st.crop_rect[0], st.crop_rect[1]
    for i, b in enumerate(st.balls, 1):
        px, py = int(round(cx0 + b.x)), int(round(cy0 + b.y))
        col = _DRAW.get(b.color, (180, 180, 180))
        cv2.circle(out, (px, py), int(round(b.r)), col, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(out, str(i), (px + int(b.r) + 2, py - int(b.r)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (px + int(b.r) + 2, py - int(b.r)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{tag}: {len(st.balls)} balls  r={st.balls[0].r:.1f}px" if st.balls else f"{tag}: 0 balls",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def ascii_state(st: State2, cols: int = 108) -> str:
    if not st.ok():
        return f"<{st.error}>"
    _cx0, _cy0, cw, ch = st.crop_rect
    cell = max(1, cw // cols)
    rows = max(1, ch // cell)
    grid = [[" "] * cols for _ in range(rows)]
    for b in st.balls:
        i = min(cols - 1, int(b.x / cell))
        j = min(rows - 1, int(b.y / cell))
        grid[j][i] = _LETTER.get(b.color, "?")
    return "\n".join("".join(r) for r in grid)


def color_counts(st: State2) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in st.balls:
        out[b.color] = out.get(b.color, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------ CLI
def _main(argv: list[str]) -> int:
    from pathlib import Path

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    # 默认用样本图；样本目录在项目根的 samples/
    default_img = project_root / "samples" / "recognize_12frames" / "frame_01_raw.png"
    p = args[0] if args else str(default_img)
    img = cv2.imread(p)
    if img is None:
        print(f"无法读取 {p}")
        print("用法: python src/detect2.py <图片路径>")
        return 1
    print(f"图像 {img.shape[1]}x{img.shape[0]}  ({p})")

    dbg: dict = {}
    st = analyze_frame2(img, debug=dbg)
    if not st.ok():
        print(f"失败: {st.error}")
        return 1
    fw, fh = st.felt_rect[2], st.felt_rect[3]
    print(f"台呢 rect={st.felt_rect} aspect={fw/max(fh,1):.3f}")
    print(f"crop={st.crop_rect}")
    print(f"半径扫描选中 r={dbg['exp_r']:.2f}px -> 距离变换校正 r={dbg.get('r_final', dbg['exp_r']):.2f}px"
          f"   (扫描范围 {dbg['r_info'].get('range')})")
    if dbg["r_info"].get("scan"):
        scan = dbg["r_info"]["scan"]
        mx = max(r["top20"] for r in scan) or 1.0
        print("  半径扫描（top20 峰值和，越高越像一组等大球）:")
        for row in scan:
            mark = " <== 选中" if row["r"] == dbg["r_info"]["best_r"] else ""
            bar = "#" * max(0, int(row["top20"] / mx * 60))
            print(f"    r={row['r']:>3} top20={row['top20']:>8.1f} p999={row['p999']:>7.2f} {bar}{mark}")
    print(f"\n检出 {len(st.balls)} 颗   颜色统计 {color_counts(st)}")
    print(f"\n  {'#':>3} {'x':>7} {'y':>7} {'r':>6} {'color':>7} {'conf':>5} {'cov':>5}  Lab")
    for i, b in enumerate(st.balls, 1):
        print(f"  {i:>3} {b.x:>7.1f} {b.y:>7.1f} {b.r:>6.2f} {b.color:>7} {b.conf:>5.2f} {b.cov:>5.2f}  "
              f"({b.lab[0]:.0f},{b.lab[1]:.0f},{b.lab[2]:.0f})")

    print("\n=== 检测布局文本视图 ===")
    print(ascii_state(st))

    out = project_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "v2_annotated.png"), draw_state2(img, st))
    cv2.imwrite(str(out / "v2_mask.png"), dbg["mask"] * 255)
    print(f"\n已存 {out / 'v2_annotated.png'} / {out / 'v2_mask.png'}")
    if "--dump-lab" in flags:
        print("\n=== 检出球 Lab 值（用于校准调色板）===")
        for i, b in enumerate(st.balls, 1):
            bgr = lab_to_bgr(np.array(b.lab))
            print(f"  {i:>3} {b.color:>7} Lab=({b.lab[0]:6.1f},{b.lab[1]:6.1f},{b.lab[2]:6.1f}) BGR={bgr}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
