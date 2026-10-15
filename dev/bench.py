# -*- coding: utf-8 -*-
"""评测台：多帧抓取 → 新旧检测器并排跑 → 输出标注图 + 打分表。

用法:
  python bench.py --frames 6 --interval 2.5
  python bench.py --image samples/recognize_12frames/frame_01_raw.png     # 只测已有图

输出（output/dev/bench_<时间戳>/）:
  frame_XX_raw.png        原始窗口截图
  frame_XX_old.png        旧 detect.py 标注
  frame_XX_new.png        新 detect2.py 标注
  frame_XX_side.png       左右并排（供打分）
  frame_XX.json           两套方案的球心/颜色/半径
  score.md                打分表
  report.txt              控制台摘要
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import capture  # noqa: E402
import detect2 as D2  # noqa: E402

PROBE = ROOT / "output/dev"


# ------------------------------------------------------------ 窗口定位（修正版）
def find_game_window() -> dict | None:
    """只认可见窗口，且优先标题精确匹配。

    旧 capture.find_window_rect 会选到不可见的同名窗口（实测 hwnd=395630
    visible=False 面积更大），导致 PrintWindow 拿到空白。此处修正。
    """
    user32 = ctypes.windll.user32
    rows: list[dict] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if not ("腾讯桌球" in title or "T-BALL" in title.upper() or "桌球" in title):
            return True
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < 400 or h < 300:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rows.append({"hwnd": int(hwnd), "title": title,
                     "rect": [r.left, r.top, w, h], "area": w * h})
        return True

    user32.EnumWindows(cb, 0)
    if not rows:
        return None
    rows.sort(key=lambda x: -x["area"])
    return rows[0]


def grab_frame(hwnd: int) -> np.ndarray | None:
    """PrintWindow 抓窗口内容（绕过遮挡）。"""
    return capture.print_window_bgr(hwnd)


# ------------------------------------------------------------ 旧检测器适配
def run_old(img: np.ndarray) -> dict:
    """跑**旧**检测器（detect_legacy.py）。

    注意：`detect.py` 现在已是新引擎的接口层（内部调用 detect2），
    所以基线必须显式指向 detect_legacy.py，否则会拿新方案和自己比。
    """
    import detect_legacy as D1

    try:
        st = D1.analyze_frame(img)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "balls": [], "felt": None}
    fx, fy, fw, fh = st.felt_rect
    balls = []
    for b in st.balls:
        balls.append({
            "x": round(fx + b.x, 1), "y": round(fy + b.y, 1),
            "r": round(b.r, 1), "color": b.color, "conf": round(b.score, 2),
        })
    return {
        "felt": [int(fx), int(fy), int(fw), int(fh)],
        "n": len(balls), "balls": balls,
        "counts": _counts([b["color"] for b in balls]),
        "error": st.error or "",
    }


def run_shipped(img: np.ndarray) -> dict:
    """跑**实际出厂**路径：detect.analyze_frame（新引擎的接口层）。

    这才是 overlay/main/analyze 真正调用的入口，
    含坐标变换、袋口归一化、球心夹取等适配逻辑，
    因此评测必须走这里，而不是直接调 detect2。
    """
    import detect as D

    try:
        st = D.analyze_frame(img)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "balls": [], "felt": None}
    fx, fy, fw, fh = st.felt_rect
    balls = []
    for b in st.balls:
        balls.append({
            "x": round(fx + b.x, 1), "y": round(fy + b.y, 1),
            "r": round(b.r, 1), "color": b.color, "conf": round(b.score, 2),
        })
    return {
        "felt": [int(fx), int(fy), int(fw), int(fh)],
        "n": len(balls), "balls": balls,
        "counts": _counts([b["color"] for b in balls]),
        "error": st.error or "",
    }


def run_new(img: np.ndarray) -> dict:
    dbg: dict = {}
    try:
        st = D2.analyze_frame2(img, debug=dbg)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "balls": [], "felt": None}
    cx0, cy0 = st.crop_rect[0], st.crop_rect[1]
    balls = []
    for b in st.balls:
        balls.append({
            "x": round(cx0 + b.x, 1), "y": round(cy0 + b.y, 1),
            "r": round(b.r, 1), "color": b.color, "conf": round(b.conf, 2),
            "score": round(b.cov, 2),
            "lab": [round(v, 1) for v in b.lab],
        })
    fx, fy, fw, fh = st.felt_rect
    return {
        "felt": [int(fx), int(fy), int(fw), int(fh)],
        "n": len(balls), "balls": balls,
        "counts": _counts([b["color"] for b in balls]),
        "radius": round(dbg.get("exp_r", 0), 2),
        "score_min": round(dbg.get("score_min", 0), 2),
        "error": st.error or "",
    }


def _counts(colors: list[str]) -> dict:
    out: dict[str, int] = {}
    for c in colors:
        out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------ 标注绘制
def draw_old(img: np.ndarray, res: dict) -> np.ndarray:
    out = img.copy()
    felt = res.get("felt")
    if felt and felt[2] > 0:
        x, y, w, h = felt
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
    for i, b in enumerate(res.get("balls", []), 1):
        px, py, r = int(b["x"]), int(b["y"]), int(b["r"])
        col = D2._DRAW.get(b["color"], (180, 180, 180))
        cv2.circle(out, (px, py), r, col, 2, lineType=cv2.LINE_AA)
        cv2.putText(out, str(i), (px + r + 2, py - r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (px + r + 2, py - r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"OLD detect.py: {res.get('n', 0)} balls", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def draw_new(img: np.ndarray, res: dict) -> np.ndarray:
    out = img.copy()
    felt = res.get("felt")
    if felt and felt[2] > 0:
        x, y, w, h = felt
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for i, b in enumerate(res.get("balls", []), 1):
        px, py, r = int(b["x"]), int(b["y"]), int(b["r"])
        col = D2._DRAW.get(b["color"], (180, 180, 180))
        cv2.circle(out, (px, py), r, col, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 9, 1)
        cv2.putText(out, str(i), (px + r + 2, py - r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (px + r + 2, py - r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"NEW detect2.py: {res.get('n', 0)} balls  r={res.get('radius', 0)}px", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1] + 8

    def pad(im):
        if im.shape[0] == h:
            return im
        return cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    canvas = np.full((h, w, 3), 20, np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] + 8 : a.shape[1] + 8 + b.shape[1]] = b
    return canvas


# ------------------------------------------------------------ 逐球放大切片
def zoom_strip(
    img: np.ndarray, balls: list[dict], *, cols: int = 6, cell: int = 132, pad_px: int = 34
) -> np.ndarray:
    """把每颗球周围裁下来放大，拼成一张对照图，便于逐颗核对。

    每格：原始放大图 + 圈 + 编号 + 颜色 + 置信度。
    """
    if not balls:
        return np.full((cell, cell, 3), 30, np.uint8)
    h, w = img.shape[:2]
    scale = 3
    inner = cell - 26
    rows = (len(balls) + cols - 1) // cols
    canvas = np.full((rows * cell + 26, cols * cell, 3), 26, np.uint8)

    for k, b in enumerate(balls):
        r = int(k // cols)
        c = int(k % cols)
        cx, cy = int(b["x"]), int(b["y"])
        x0, y0 = max(0, cx - pad_px), max(0, cy - pad_px)
        x1, y1 = min(w, cx + pad_px), min(h, cy + pad_px)
        patch = img[y0:y1, x0:x1].copy()
        if patch.size == 0:
            continue
        big = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        bh, bw = big.shape[:2]
        ox, oy = (inner - bw) // 2, (inner - bh) // 2
        tile = np.full((inner, inner, 3), 26, np.uint8)
        if bh <= inner and bw <= inner:
            tile[oy : oy + bh, ox : ox + bw] = big
        else:
            big = cv2.resize(big, (inner, inner), interpolation=cv2.INTER_AREA)
            tile = big
        # 球心十字 + 半径圈（按放大倍数）
        ccx = ox + (cx - x0) * scale
        ccy = oy + (cy - y0) * scale
        rad = int(b["r"] * scale)
        col = D2._DRAW.get(b["color"], (180, 180, 180))
        cv2.circle(tile, (ccx, ccy), rad, col, 1, lineType=cv2.LINE_AA)
        cv2.drawMarker(tile, (ccx, ccy), (0, 0, 0), cv2.MARKER_CROSS, 7, 1)

        y_base = r * cell
        x_base = c * cell
        canvas[y_base + 13 : y_base + 13 + inner, x_base : x_base + inner] = tile
        label = f"#{k+1} {b['color']} {b.get('conf', 0):.2f}"
        cv2.putText(canvas, label, (x_base + 3, y_base + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x_base, y_base), (x_base + cell - 1, y_base + cell + 12), (70, 70, 70), 1)
    return canvas


def zoom_at(img: np.ndarray, gx: float, gy: float, pad_px: int = 34, scale: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = int(gx), int(gy)
    x0, y0 = max(0, cx - pad_px), max(0, cy - pad_px)
    x1, y1 = min(w, cx + pad_px), min(h, cy + pad_px)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return np.full((10, 10, 3), 26, np.uint8)
    big = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(big, ((cx - x0) * scale, (cy - y0) * scale), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    return big


# ------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6, help="抓取帧数")
    ap.add_argument("--interval", type=float, default=2.5, help="帧间隔秒")
    ap.add_argument("--image", default=None, help="只测一张已有图")
    ap.add_argument("--tag", default=None, help="输出目录后缀")
    args = ap.parse_args()

    ts = time.strftime("%m%d_%H%M%S")
    tag = f"{ts}_{args.tag}" if args.tag else ts
    outdir = PROBE / f"bench_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    say("=" * 78)
    say(f"评测台 输出目录: {outdir}")
    say("=" * 78)

    frames: list[tuple[str, np.ndarray]] = []

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            say(f"无法读取 {args.image}")
            return 1
        frames.append((Path(args.image).stem, img))
    else:
        win = find_game_window()
        if win is None:
            say("未找到可见的「腾讯桌球」窗口。请确认小程序已打开。")
            return 1
        say(f"游戏窗口: hwnd={win['hwnd']} rect={win['rect']} title={win['title']!r}")
        for i in range(args.frames):
            img = grab_frame(win["hwnd"])
            if img is None:
                say(f"  第 {i+1} 帧抓取失败")
                continue
            say(f"  第 {i+1} 帧: {img.shape[1]}x{img.shape[0]} mean={img.mean():.1f} std={img.std():.1f}")
            frames.append((f"frame_{i+1:02d}", img))
            if i < args.frames - 1:
                time.sleep(args.interval)

    if not frames:
        say("没有可用帧")
        return 1

    records = []
    say("")
    for name, img in frames:
        say("-" * 78)
        say(f"[{name}]  {img.shape[1]}x{img.shape[0]}")
        old = run_old(img)
        new = run_new(img)
        say(f"  旧 detect.py : {old.get('n', 0):>3} 颗  {old.get('counts', {})}"
            + (f"  ERR={old['error']}" if old.get("error") else ""))
        say(f"  新 detect2.py: {new.get('n', 0):>3} 颗  {new.get('counts', {})}"
            + (f"  ERR={new['error']}" if new.get("error") else "")
            + f"  r={new.get('radius')}px")

        a = draw_old(img, old)
        b = draw_new(img, new)
        cv2.imwrite(str(outdir / f"{name}_raw.png"), img)
        cv2.imwrite(str(outdir / f"{name}_old.png"), a)
        cv2.imwrite(str(outdir / f"{name}_new.png"), b)
        cv2.imwrite(str(outdir / f"{name}_side.png"), side_by_side(a, b))
        # 逐球放大切片：便于逐颗核对颜色/圆心
        cv2.imwrite(str(outdir / f"{name}_new_zoom.png"), zoom_strip(img, new.get("balls", [])))
        cv2.imwrite(str(outdir / f"{name}_old_zoom.png"), zoom_strip(img, old.get("balls", [])))

        rec = {"name": name, "size": [img.shape[1], img.shape[0]],
               "old": old, "new": new}
        (outdir / f"{name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        records.append(rec)

    (outdir / "all.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "report.txt").write_text("\n".join(log), encoding="utf-8")

    # ---------------- 打分表 ----------------
    md: list[str] = []
    md.append(f"# 球体识别打分表  `{tag}`\n")
    md.append("对每一帧，请给**新方案** `detect2.py` 打分。\n")
    md.append("**打分方法**：打开 `frame_XX_side.png`（左=旧方案，右=新方案），")
    md.append("对照真实球局数一数。\n")
    md.append("\n评分栏建议填：")
    md.append("- **漏检数**：真实存在但没画圈的球有几个")
    md.append("- **误检数**：画了圈但那里没有球的有几个")
    md.append("- **颜色错**：圈对了但颜色标错的有几个")
    md.append("- **圆心偏**：圈位置明显偏离球心的有几个")
    md.append("- **总分**：0–10（10 = 完全正确）\n")
    md.append("| 帧 | 旧方案球数 | 新方案球数 | 漏检 | 误检 | 颜色错 | 圆心偏 | 总分 | 备注 |")
    md.append("|----|-----------|-----------|------|------|--------|--------|------|------|")
    for r in records:
        md.append(f"| {r['name']} | {r['old'].get('n', 0)} | {r['new'].get('n', 0)} "
                  f"|  |  |  |  |  |  |")
    md.append("\n## 各帧新方案明细\n")
    for r in records:
        md.append(f"### {r['name']}  （新方案 {r['new'].get('n',0)} 颗，r={r['new'].get('radius')}px）\n")
        md.append("| # | x | y | r | 颜色 | 置信 | 得分 | 你认为实际颜色 |")
        md.append("|---|---|---|---|------|------|------|----------------|")
        for i, b in enumerate(r["new"].get("balls", []), 1):
            md.append(f"| {i} | {b['x']} | {b['y']} | {b['r']} | {b['color']} "
                      f"| {b['conf']} | {b.get('score')} |  |")
        md.append("")
        md.append(f"旧方案同帧: {r['old'].get('counts', {})}  共 {r['old'].get('n',0)} 颗\n")
    (outdir / "score.md").write_text("\n".join(md), encoding="utf-8")

    say("")
    say("=" * 78)
    say(f"已输出到: {outdir}")
    say(f"  打分表: {outdir / 'score.md'}")
    say(f"  并排图: frame_XX_side.png  (左=旧 右=新)")
    say(f"  逐球放大: frame_XX_new_zoom.png")
    say("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
