# -*- coding: utf-8 -*-
"""腾讯桌球 / 斯诺克解球辅助 —— 命令行入口

用法:
  python main.py analyze <图片路径> [--target red] [--mode escape|pot|both]
  python main.py snap              # 截当前窗口并分析（解球）
  python main.py snap --pot        # 截图并给出进球线路
  python main.py calibrate         # 显示截取区域配置提示
  python main.py demo              # 用合成球局演示算法
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from analyze import analyze, draw_result, summarize
from capture import grab_default_table, grab_game_window, load_image, save_snapshot
from config import OUT_DIR, TABLE_REGION


def _resolve_target(args) -> str | None:
    return getattr(args, "target", None)


def _run_analysis(img, args, tag: str):
    mode = "pot" if getattr(args, "pot", False) else getattr(args, "mode", "escape")
    if getattr(args, "both", False):
        mode = "both"
    result = analyze(
        img,
        target_color=_resolve_target(args),
        mode=mode,
        max_cushions=getattr(args, "cushions", None),
    )
    annotated = draw_result(img, result, top_k=getattr(args, "topk", 3))
    out_path = OUT_DIR / f"result_{tag}_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(str(out_path), annotated)
    snap = OUT_DIR / f"input_{tag}_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(str(snap), img)
    print(summarize(result))
    print()
    print(f"[保存] 输入: {snap}")
    print(f"[保存] 标注: {out_path}")
    # 预览（--no-gui 跳过）
    if not getattr(args, "no_gui", False):
        try:
            scale = 1280 / max(annotated.shape[1], 1)
            if scale < 1:
                disp = cv2.resize(annotated, None, fx=scale, fy=scale)
            else:
                disp = annotated
            cv2.imshow("Snooker Assist", disp)
            print("按任意键关闭预览…")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"(预览窗口失败: {e})")
    return result


def cmd_analyze(args):
    img = load_image(args.image)
    return _run_analysis(img, args, "file")


def cmd_snap(args):
    if args.region:
        # region: left,top,width,height
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) != 4:
            print("region 格式: left,top,width,height")
            return 2
        region = dict(zip(["left", "top", "width", "height"], parts))
        from capture import grab_screen

        img = grab_screen(region)
    elif args.fullscreen:
        from capture import grab_screen

        img = grab_screen()
    else:
        img, region = grab_game_window()
        print(f"截取窗口: {region}")
    return _run_analysis(img, args, "snap")


def cmd_calibrate(_args):
    print("当前 TABLE_REGION 配置（config.py）:")
    print(f"  {TABLE_REGION}")
    print()
    print("标定步骤：")
    print("1. 打开腾讯桌球，摆好球局")
    print("2. 运行:  python main.py snap --fullscreen")
    print("3. 查看 output/ 下标注图，黄框即当前 felt 检测结果")
    print("4. 若不准，修改 config.py 的 TABLE_REGION，只框住台面（含库边），避开左右侧 UI")
    print("5. 也可每次用:  python main.py snap --region 140,200,1680,840")
    return 0


def cmd_demo(_args):
    """合成一局被挡住的解球局面，验证镜面算法。"""
    import numpy as np

    # 绿台
    W, H = 1600, 800
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 120, 50)
    # 库边
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (40, 70, 140), 24)

    felt = (40, 40, W - 80, H - 80)

    def put(nx, ny, color_bgr, r=18):
        x = int(felt[0] + nx * felt[2])
        y = int(felt[1] + ny * felt[3])
        cv2.circle(img, (x, y), r, color_bgr, -1)
        cv2.circle(img, (x, y), r, (255, 255, 255), 1)
        # 高光
        cv2.circle(img, (x - r // 3, y - r // 3), max(2, r // 4), (255, 255, 255), -1)

    # 白球在左下，目标红在右侧，中间被蓝球挡住直线
    put(0.15, 0.70, (255, 255, 255))  # cue
    put(0.75, 0.35, (40, 40, 230))    # target red
    put(0.45, 0.52, (220, 120, 30))   # blue blocker
    put(0.20, 0.25, (0, 220, 255))    # yellow
    put(0.80, 0.70, (30, 30, 30))     # black

    cv2.imwrite(str(OUT_DIR / "demo_input.png"), img)
    print("已生成演示图:", OUT_DIR / "demo_input.png")
    args = argparse.Namespace(image=str(OUT_DIR / "demo_input.png"), target="red", mode="escape", pot=False, both=False, cushions=2, topk=3, no_gui=True)
    return _run_analysis(img, args, "demo")


def build_parser():
    p = argparse.ArgumentParser(description="斯诺克/腾讯桌球解球辅助")
    sub = p.add_subparsers(dest="cmd")

    pa = sub.add_parser("analyze", help="分析图片文件")
    pa.add_argument("image", help="截图路径")
    _add_common(pa)
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("snap", help="截取当前屏幕/游戏窗口并分析")
    ps.add_argument("--region", help="left,top,width,height")
    ps.add_argument("--fullscreen", action="store_true", help="截全屏")
    _add_common(ps)
    ps.set_defaults(func=cmd_snap)

    pc = sub.add_parser("calibrate", help="标定说明")
    pc.set_defaults(func=cmd_calibrate)

    pd = sub.add_parser("demo", help="算法演示")
    pd.set_defaults(func=cmd_demo)
    return p


def _add_common(sp):
    sp.add_argument("--target", default=None, help="目标球颜色 red/yellow/green/brown/blue/pink/black")
    sp.add_argument("--mode", default="escape", choices=["escape", "pot", "both"], help="escape=解球碰目标, pot=进球线路, both=两者")
    sp.add_argument("--pot", action="store_true", help="等价 --mode pot")
    sp.add_argument("--both", action="store_true", help="等价 --mode both")
    sp.add_argument("--cushions", type=int, default=None, help="最大库数 1-3")
    sp.add_argument("--topk", type=int, default=3, help="绘制前几条方案")
    sp.add_argument("--no-gui", action="store_true", help="不弹出预览窗口，只保存图片并打印文字")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # 默认：尝试截图分析
        args = parser.parse_args(["snap"])
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
