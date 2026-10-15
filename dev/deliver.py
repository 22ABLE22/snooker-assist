# -*- coding: utf-8 -*-
"""最终交付：多帧球体识别结果 + 自检 + 打分表。

用法:
  python deliver.py                 # 抓 6 帧实时画面
  python deliver.py --frames 10 --interval 1.2

产出（output/dev/final_<时间戳>/）:
  frame_XX_compare.png   左=旧 detect.py  右=新 detect2.py
  frame_XX_zoom.png      新方案逐球 3 倍放大（核对颜色/圆心）
  frame_XX_balls.png     新方案逐球 3 倍放大（含编号）
  frame_XX_raw.png       原始截图
  frame_XX.json          球心/半径/颜色/置信度
  SCORE.md               打分表（请在此填分）
  SELFCHECK.txt          自检报告（规则冲突、帧间稳定性）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import bench  # noqa: E402
import detect2 as D2  # noqa: E402

PROBE = ROOT / "output/dev"


def selfcheck(recs: list[dict]) -> list[str]:
    """规则自检：斯诺克球数与颜色约束。"""
    out: list[str] = []
    out.append("=" * 74)
    out.append("自检报告")
    out.append("=" * 74)

    # 1) 帧间稳定性
    out.append("\n[1] 帧间稳定性（球数 / 半径 / 颜色分布）")
    ns = [r["new"].get("n", 0) for r in recs]
    rs = [r["new"].get("radius") for r in recs]
    out.append(f"    球数: {ns}   极差={max(ns)-min(ns) if ns else 0}")
    out.append(f"    半径: {rs}   极差={max(rs)-min(rs) if rs else 0}")
    if len(set(ns)) == 1 and len(set(rs)) == 1:
        out.append("    判定: 稳定 [OK]")
    else:
        out.append("    判定: 不稳定 [FAIL] (需排查)")

    # 2) 规则约束
    out.append("\n[2] 斯诺克规则约束")
    for r in recs:
        cnt = r["new"].get("counts", {})
        issues = []
        if cnt.get("black", 0) > 1:
            issues.append(f"黑球 {cnt['black']} 颗 > 1")
        if cnt.get("red", 0) > 15:
            issues.append(f"红球 {cnt['red']} 颗 > 15")
        for c in ("cue", "yellow", "green", "brown", "blue", "pink", "black"):
            if cnt.get(c, 0) > 1:
                issues.append(f"{c} {cnt[c]} 颗 > 1")
        tag = "OK" if not issues else "!! " + "; ".join(issues)
        out.append(f"    {r['name']}: {cnt}  -> {tag}")

    # 3) 新旧差异
    out.append("\n[3] 新旧方案差异")
    for r in recs:
        o, n = r["old"].get("n", 0), r["new"].get("n", 0)
        d = n - o
        out.append(f"    {r['name']}: 旧={o} 新={n} 差={d:+d}")

    # 4) 逐球置信度
    out.append("\n[4] 新方案低置信度球（需人工确认）")
    any_low = False
    for r in recs:
        for i, b in enumerate(r["new"].get("balls", []), 1):
            if b.get("conf", 1.0) < 0.6 or b.get("score", 99) < 40:
                any_low = True
                out.append(f"    {r['name']} #{i} {b['color']} "
                           f"conf={b.get('conf')} score={b.get('score')} "
                           f"@({b['x']:.0f},{b['y']:.0f})")
    if not any_low:
        out.append("    无（全部 conf>=0.6 且 score>=40）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    ts = time.strftime("%m%d_%H%M%S")
    outdir = PROBE / f"final_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    say("=" * 74)
    say(f"球体识别交付包: {outdir}")
    say("=" * 74)

    frames: list[tuple[str, np.ndarray]] = []
    if args.image:
        im = cv2.imread(args.image)
        if im is None:
            say(f"无法读取 {args.image}")
            return 1
        frames.append((Path(args.image).stem, im))
    else:
        win = bench.find_game_window()
        if win is None:
            say("未找到可见的「腾讯桌球」窗口。请确认小程序已打开并可见。")
            return 1
        say(f"游戏窗口: hwnd={win['hwnd']} rect={win['rect']}")
        for i in range(args.frames):
            im = bench.grab_frame(win["hwnd"])
            if im is None:
                say(f"  第 {i+1} 帧抓取失败")
                continue
            frames.append((f"frame_{i+1:02d}", im))
            say(f"  第 {i+1} 帧 {im.shape[1]}x{im.shape[0]}")
            if i < args.frames - 1:
                time.sleep(args.interval)

    if not frames:
        say("无可用帧")
        return 1

    recs = []
    for name, img in frames:
        old = bench.run_old(img)
        new = bench.run_shipped(img)
        say(f"[{name}] 旧 {old.get('n',0)} 颗 | 新 {new.get('n',0)} 颗 "
            f"{new.get('counts', {})}  r={new.get('radius')}px")

        cv2.imwrite(str(outdir / f"{name}_raw.png"), img)
        cv2.imwrite(str(outdir / f"{name}_compare.png"),
                    bench.side_by_side(bench.draw_old(img, old), bench.draw_new(img, new)))
        cv2.imwrite(str(outdir / f"{name}_zoom.png"),
                    bench.zoom_strip(img, new.get("balls", [])))
        cv2.imwrite(str(outdir / f"{name}_balls.png"),
                    bench.zoom_strip(img, new.get("balls", []), cols=5, cell=160, pad_px=30))
        rec = {"name": name, "size": [img.shape[1], img.shape[0]], "old": old, "new": new}
        (outdir / f"{name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        recs.append(rec)

    (outdir / "all.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 打分表 ----------------
    md = [
        f"# 球体识别打分表\n",
        f"生成时间 `{ts}`，共 {len(recs)} 帧。\n",
        "## 怎么看\n",
        "1. 打开 `frame_XX_compare.png`：**左边是旧 detect.py，右边是新 detect2.py**。",
        "2. 打开 `frame_XX_zoom.png`：新方案每颗球的 **3 倍放大图**，逐颗核对颜色和圆心。",
        "3. 打开 `frame_XX_raw.png`：原始截图，用来数清真实有几颗球。\n",
        "## 请填这张表\n",
        "| 帧 | 新方案球数 | 漏检 | 误检 | 颜色错 | 圆心偏 | 总分(0-10) | 备注 |",
        "|----|-----------|------|------|--------|--------|-----------|------|",
    ]
    for r in recs:
        md.append(f"| {r['name']} | {r['new'].get('n',0)} |  |  |  |  |  |  |")
    md.append("\n> 漏检 = 有球但没圈；误检 = 圈了但没球；颜色错 = 圈对但颜色错；圆心偏 = 圈偏离球心。\n")

    md.append("\n## 各帧逐球明细（新方案）\n")
    for r in recs:
        nw = r["new"]
        md.append(f"### {r['name']}\n")
        md.append(f"- 新方案: **{nw.get('n',0)} 颗**，统一半径 {nw.get('radius')}px，"
                  f"颜色分布 `{nw.get('counts', {})}`")
        md.append(f"- 旧方案: {r['old'].get('n',0)} 颗，`{r['old'].get('counts', {})}`\n")
        md.append("| # | x | y | r | 颜色 | 置信 | 斑点分 | 实际颜色(请填) |")
        md.append("|---|---|---|---|------|------|--------|----------------|")
        for i, b in enumerate(nw.get("balls", []), 1):
            md.append(f"| {i} | {b['x']:.0f} | {b['y']:.0f} | {b['r']:.0f} | {b['color']} "
                      f"| {b['conf']:.2f} | {b.get('score',0):.1f} |  |")
        md.append("")
    (outdir / "SCORE.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- 自检 ----------------
    chk = selfcheck(recs)
    (outdir / "SELFCHECK.txt").write_text("\n".join(chk), encoding="utf-8")
    say("")
    for line in chk:
        say(line)

    (outdir / "report.txt").write_text("\n".join(log), encoding="utf-8")
    say("")
    say("=" * 74)
    say(f"交付目录: {outdir}")
    say(f"  打分表   : SCORE.md")
    say(f"  对比图   : frame_XX_compare.png (左旧右新)")
    say(f"  逐球放大 : frame_XX_zoom.png / frame_XX_balls.png")
    say(f"  原始截图 : frame_XX_raw.png")
    say(f"  数据     : frame_XX.json / all.json")
    say(f"  自检     : SELFCHECK.txt")
    say("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
