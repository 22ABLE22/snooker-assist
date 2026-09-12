# -*- coding: utf-8 -*-
"""置顶悬浮标注窗 + 全局热键

用法:
  python overlay.py

热键（全局）:
  F8  —— 截图分析解球线路（碰目标）
  F9  —— 截图分析进球线路
  F10 —— 切换目标球颜色
  F11 —— 清除标注 / 切换透明
  F12 —— 退出

说明:
  - 悬浮窗覆盖在「腾讯桌球」窗口上，半透明显示路径，默认鼠标穿透
  - 不依赖 PIL；用 tkinter Canvas 画线
"""

from __future__ import annotations

import ctypes
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

# 保证可导入本目录模块
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze import AnalysisResult, analyze  # noqa: E402
from capture import find_window_rect, grab_game_window  # noqa: E402
from config import TARGET_COLORS  # noqa: E402

# ---------- Win32 热键（线程队列 hwnd=0，更可靠）----------

user32 = ctypes.windll.user32
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
VK_F8, VK_F9, VK_F10, VK_F11, VK_F12 = 0x77, 0x78, 0x79, 0x7A, 0x7B
HOTKEY_IDS = {"F8": 108, "F9": 109, "F10": 110, "F11": 111, "F12": 112}
# 委托在 RegisterHotKey 的线程上，消息进线程队列
_HOTKEY_HWND = wintypes.HWND(0)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", ctypes.c_uint),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


def register_hotkeys() -> dict[str, bool]:
    """注册全局热键到当前线程（hwnd=0）。返回各键是否成功。"""
    mods = MOD_NOREPEAT
    mapping = [
        ("F8", HOTKEY_IDS["F8"], VK_F8),
        ("F9", HOTKEY_IDS["F9"], VK_F9),
        ("F10", HOTKEY_IDS["F10"], VK_F10),
        ("F11", HOTKEY_IDS["F11"], VK_F11),
        ("F12", HOTKEY_IDS["F12"], VK_F12),
    ]
    ok: dict[str, bool] = {}
    for name, hid, vk in mapping:
        # hwnd=0 → 投递到调用线程消息队列
        success = bool(user32.RegisterHotKey(_HOTKEY_HWND, hid, mods, vk))
        if not success:
            # 不带 NOREPEAT 再试一次
            success = bool(user32.RegisterHotKey(_HOTKEY_HWND, hid, 0, vk))
        ok[name] = success
        if success:
            print(f"[hotkey] {name} 已注册")
        else:
            print(f"[warn] {name} 注册失败（可能被其他程序占用）")
    return ok


def unregister_hotkeys() -> None:
    for hid in HOTKEY_IDS.values():
        user32.UnregisterHotKey(_HOTKEY_HWND, hid)


def pump_hotkeys(callbacks: dict[int, callable]) -> int:
    """非阻塞取出线程队列中的 WM_HOTKEY。返回处理条数。"""
    msg = MSG()
    # hwnd=0：本线程队列
    n = 0
    while user32.PeekMessageW(ctypes.byref(msg), 0, WM_HOTKEY, WM_HOTKEY, 1):
        wid = int(msg.wParam)
        fn = callbacks.get(wid)
        if fn:
            fn()
            n += 1
    return n


def set_click_through(hwnd: int, enable: bool = True) -> None:
    """鼠标穿透悬浮窗。"""
    if not hwnd:
        return
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    style = user32.GetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW
    if enable:
        style |= WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    user32.SetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE, style)


# ---------- 悬浮窗 ----------

COLOR_HEX = {
    "cue": "#FFFFFF",
    "red": "#E02020",
    "yellow": "#FFE040",
    "green": "#30C060",
    "brown": "#A06030",
    "blue": "#3080FF",
    "pink": "#FF90C0",
    "black": "#202020",
    "unknown": "#AAAAAA",
}

PATH_COLORS = ["#FFE040", "#30D0FF", "#FF60E0"]


class OverlayApp:
    def __init__(self) -> None:
        self.target_idx = 0
        self.mode = "escape"  # escape | pot
        self.show = True
        self.last_result: AnalysisResult | None = None
        self.status = "F8解球 F9进球 F10换目标 F11显隐 F12退出"
        self.click_through = True

        self.root = tk.Tk()
        self.root.title("Snooker Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")

        # Windows 黑色透明
        try:
            self.root.wm_attributes("-transparentcolor", "black")
        except tk.TclError:
            pass
        try:
            self.root.attributes("-alpha", 0.92)
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._sync_to_game()
        self.root.update_idletasks()
        self.hwnd = self._get_hwnd()
        if self.hwnd:
            set_click_through(self.hwnd, self.click_through)

        # 热键：Win32 线程队列 + 可选 keyboard 库 + 窗口内按键
        self.hotkey_ok = register_hotkeys()
        self._kb = None
        try:
            import keyboard as kb  # type: ignore

            self._kb = kb
            kb.add_hotkey("f8", lambda: self.root.after(0, self.on_escape))
            kb.add_hotkey("f9", lambda: self.root.after(0, self.on_pot))
            kb.add_hotkey("f10", lambda: self.root.after(0, self.on_cycle_target))
            kb.add_hotkey("f11", lambda: self.root.after(0, self.on_toggle))
            kb.add_hotkey("esc", lambda: self.root.after(0, self.quit))
            print("[hotkey] keyboard 库已启用（双保险）")
        except Exception as e:
            print(f"[hotkey] keyboard 库不可用: {e}")

        for key, fn in (
            ("<F8>", self.on_escape),
            ("<F9>", self.on_pot),
            ("<F10>", self.on_cycle_target),
            ("<F11>", self.on_toggle),
            ("<F12>", self.quit),
            ("<Escape>", self.quit),
        ):
            self.root.bind_all(key, lambda e, f=fn: f())

        self.root.after(40, self._tick)

    def _get_hwnd(self) -> int | None:
        try:
            # tkinter widget 的父窗口才是顶层 HWND
            return int(user32.GetParent(self.root.winfo_id()))
        except Exception:
            try:
                return int(self.root.winfo_id())
            except Exception:
                return None

    def _sync_to_game(self) -> dict | None:
        info = find_window_rect()
        if not info:
            # 居中默认
            w, h = 900, 500
            x = (self.root.winfo_screenwidth() - w) // 2
            y = (self.root.winfo_screenheight() - h) // 2
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.grect = {"left": x, "top": y, "width": w, "height": h}
            return None
        # 略外扩，覆盖完整窗口
        pad = 0
        x, y = info["left"] - pad, info["top"] - pad
        w, h = info["width"] + 2 * pad, info["height"] + 2 * pad
        self.root.geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")
        self.grect = info
        return info

    @property
    def target_color(self) -> str:
        return TARGET_COLORS[self.target_idx % len(TARGET_COLORS)]

    def _tick(self) -> None:
        pump_hotkeys(
            {
                HOTKEY_IDS["F8"]: self.on_escape,
                HOTKEY_IDS["F9"]: self.on_pot,
                HOTKEY_IDS["F10"]: self.on_cycle_target,
                HOTKEY_IDS["F11"]: self.on_toggle,
                HOTKEY_IDS["F12"]: self.quit,
            }
        )
        # 跟随游戏窗口移动
        info = find_window_rect()
        if info:
            cur = self.root.geometry()
            want = f"{info['width']}x{info['height']}+{info['left']}+{info['top']}"
            if cur != want:
                try:
                    self.root.geometry(want)
                    self.grect = info
                except Exception:
                    pass
        self.root.after(40, self._tick)

    def on_escape(self) -> None:
        # F8：优先找进球，没有再给解球
        self.mode = "both"
        self._run()

    def on_pot(self) -> None:
        self.mode = "pot"
        self._run()

    def on_cycle_target(self) -> None:
        self.target_idx = (self.target_idx + 1) % len(TARGET_COLORS)
        self.status = f"目标={self.target_color}"
        self._redraw()
        # 若已有最近截图思路：直接重分析需新截图；这里只改目标，下次 F8/F9 生效

    def on_toggle(self) -> None:
        self.show = not self.show
        if not self.show:
            self.canvas.delete("all")
            self.status = "已隐藏标注（F11 恢复）"
            self._draw_status()
        else:
            self.status = "显示标注"
            self._redraw()

    def _run(self) -> None:
        try:
            img, meta = grab_game_window()
        except Exception as e:
            self.status = f"截图失败: {e}"
            self._draw_status()
            return
        self.status = f"分析中… [{meta.get('method','')}] 目标={self.target_color} 模式={self.mode}"
        self._draw_status()
        self.root.update()
        try:
            result = analyze(img, target_color=self.target_color, mode=self.mode, max_cushions=2)
        except Exception as e:
            self.status = f"分析失败: {e}"
            self.last_result = None
            self._redraw()
            return
        self.last_result = result
        n = len(result.plans)
        best = result.plans[0] if result.plans else None
        cush = "-".join(best.cushions) if best and best.cushions else (best.kind if best else "-")
        sc = f"{best.score:.0f}" if best is not None else "-"
        self.status = (
            f"{result.message} | 首选:{cush} 分{sc} | 目标={self.target_color}"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if not self.show:
            self._draw_status()
            return
        r = self.last_result
        if r is None or not r.state.ok():
            self._draw_status()
            return

        fx, fy, fw, fh = r.state.felt_rect
        # 悬浮窗坐标 = 游戏窗口客户区原点；截图坐标与之对应
        # grab 的图像原点是窗口区域，felt_rect 是图像内坐标，可直接映射
        ox, oy = 0, 0

        def to_xy(nx, ny):
            return (ox + fx + nx * fw, oy + fy + ny * fh)

        # 台面框（细）
        self.canvas.create_rectangle(fx, fy, fx + fw, fy + fh, outline="#00C8C8", width=1)

        # 口袋
        pr = max(8, int(fh * 0.036))
        for p in r.state.pockets:
            x, y = to_xy(p[0], p[1])
            self.canvas.create_oval(x - pr, y - pr, x + pr, y + pr, outline="#FF8C00", width=2)

        # 球
        br_px = max(5, int(r.ball_radius_norm * fh))
        for b in r.state.balls:
            col = COLOR_HEX.get(b.color, "#AAAAAA")
            cx, cy = fx + b.x, fy + b.y
            rr = max(4, int(b.r))
            self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=col, width=2)
            if b.color in ("cue", "blue", "black", "pink", "yellow"):
                self.canvas.create_text(
                    cx + rr + 2, cy, text=b.color, anchor="w", fill="#EEEEEE",
                    font=("Segoe UI", 8),
                )

        # 路径
        for i, plan in enumerate(r.plans[:3]):
            col = PATH_COLORS[i % len(PATH_COLORS)]
            pts = [to_xy(p[0], p[1]) for p in plan.points]
            flat = []
            for x, y in pts:
                flat.extend([x, y])
            if len(flat) >= 4:
                self.canvas.create_line(*flat, fill=col, width=3 if i == 0 else 2, smooth=False)
            # 库点
            for pt in pts[1:-1]:
                self.canvas.create_oval(pt[0] - 6, pt[1] - 6, pt[0] + 6, pt[1] + 6, outline="#00FF60", width=2)
            # 幽灵球
            gx, gy = to_xy(plan.impact_point[0], plan.impact_point[1])
            self.canvas.create_oval(gx - br_px, gy - br_px, gx + br_px, gy + br_px, outline="#FF40FF", width=2)
            # 标签
            label = f"#{i+1} {plan.kind} sc={plan.score:.0f}"
            self.canvas.create_text(
                pts[0][0] + 12, pts[0][1] - 14 - i * 16,
                text=label, anchor="w", fill=col, font=("Segoe UI", 10, "bold"),
            )

        self._draw_status()

    def _draw_status(self) -> None:
        # 半透明感：用深色底条
        try:
            w = self.root.winfo_width()
        except Exception:
            w = 800
        self.canvas.create_rectangle(0, 0, max(w, 200), 36, fill="#101010", outline="")
        self.canvas.create_text(
            10, 18,
            text=f"{self.status}  |  目标:{self.target_color}  点击穿透:{'开' if self.click_through else '关'}",
            anchor="w", fill="#F0F0F0", font=("Segoe UI", 11),
        )

    def quit(self) -> None:
        try:
            unregister_hotkeys()
        except Exception:
            pass
        try:
            if self._kb:
                self._kb.unhook_all_hotkeys()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        print("悬浮窗已启动")
        print("  F8 解球  F9 进球  F10 切换目标  F11 显隐  F12/Esc 退出")
        print("  鼠标默认穿透，可直接操作游戏")
        self.root.mainloop()


def main() -> int:
    app = OverlayApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
