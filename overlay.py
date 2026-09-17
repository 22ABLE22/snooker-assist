# -*- coding: utf-8 -*-
"""置顶悬浮标注窗 + 全局热键

用法:
  python overlay.py

热键（全局）:
  F6  —— 清除台面锁定（恢复自动检测）
  F7  —— 校准台面：拖四边/按钮微调，再按 F7 锁定
  F8  —— 截图分析（优先进球，其次解球）
  F9  —— 截图分析进球线路
  F10 —— 切换目标球颜色
  F11 —— 显示/隐藏标注
  F12 / Esc —— 退出

说明:
  - 悬浮窗覆盖在「腾讯桌球」窗口上，半透明显示路径，默认鼠标穿透
  - 校准时会临时关闭鼠标穿透；确认后 F8/F9 不再改台面大小
  - 不依赖 PIL；用 tkinter Canvas 画线
"""

from __future__ import annotations

import json
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
from config import POCKET_DRAW_RADIUS_RATIO, TARGET_COLORS  # noqa: E402
from guide import detect_aim_line_window  # noqa: E402

# ---------- Win32 热键（线程队列 hwnd=0，更可靠）----------

user32 = ctypes.windll.user32
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F10, VK_F11, VK_F12 = 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B
HOTKEY_IDS = {
    "F5": 105,
    "F6": 106,
    "F7": 107,
    "F8": 108,
    "F9": 109,
    "F10": 110,
    "F11": 111,
    "F12": 112,
}
FELT_LOCK_PATH = ROOT / "output" / "felt_lock.json"
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
        ("F5", HOTKEY_IDS["F5"], VK_F5),
        ("F6", HOTKEY_IDS["F6"], VK_F6),
        ("F7", HOTKEY_IDS["F7"], VK_F7),
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
        self.status = "F9进球+延长线 F8分析 F7校准 F6清锁定 F5显隐延长线 F10换目标"
        self.click_through = True
        self.felt_lock: tuple[float, float, float, float] | None = None
        self.calib = False
        self.calib_rect: list[float] = [0.1, 0.1, 0.8, 0.8]
        self._drag_edge: str | None = None
        self._img_size = (1, 1)
        self._load_felt_lock()
        # 延长线：仅在 F8/F9 分析时截一帧绘制（默认显示）
        self.live_guide = True
        self.guide_pts: list[tuple[float, float]] | None = None
        self._guide_busy = False
        self._last_felt: tuple[int, int, int, int] | None = None

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
            kb.add_hotkey("f5", lambda: self.root.after(0, self.on_toggle_guide))
            kb.add_hotkey("f6", lambda: self.root.after(0, self.on_clear_lock))
            kb.add_hotkey("f7", lambda: self.root.after(0, self.on_calibrate))
            kb.add_hotkey("f8", lambda: self.root.after(0, self.on_escape))
            kb.add_hotkey("f9", lambda: self.root.after(0, self.on_pot))
            kb.add_hotkey("f10", lambda: self.root.after(0, self.on_cycle_target))
            kb.add_hotkey("f11", lambda: self.root.after(0, self.on_toggle))
            kb.add_hotkey("esc", lambda: self.root.after(0, self.quit))
            print("[hotkey] keyboard 库已启用（双保险）")
        except Exception as e:
            print(f"[hotkey] keyboard 库不可用: {e}")

        for key, fn in (
            ("<F5>", self.on_toggle_guide),
            ("<F6>", self.on_clear_lock),
            ("<F7>", self.on_calibrate),
            ("<F8>", self.on_escape),
            ("<F9>", self.on_pot),
            ("<F10>", self.on_cycle_target),
            ("<F11>", self.on_toggle),
            ("<F12>", self.quit),
            ("<Escape>", self.quit),
            ("<Left>", lambda: self.nudge_edge("left", -2)),
            ("<Right>", lambda: self.nudge_edge("left", 2)),
            ("<Up>", lambda: self.nudge_edge("top", -2)),
            ("<Down>", lambda: self.nudge_edge("top", 2)),
        ):
            self.root.bind_all(key, lambda e, f=fn: f())

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

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
                HOTKEY_IDS["F5"]: self.on_toggle_guide,
                HOTKEY_IDS["F6"]: self.on_clear_lock,
                HOTKEY_IDS["F7"]: self.on_calibrate,
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
        # 不再后台实时抓帧（避免刷屏与抢热键）；延长线在 _run 里随 F8/F9 计算
        self.root.after(80, self._tick)

    def on_toggle_guide(self) -> None:
        """F5：显示/隐藏最近一次分析得到的延长线（不会自动循环截图）。"""
        self.live_guide = not self.live_guide
        if not self.live_guide:
            self.status = "延长线显示：关（F5 开）；按 F8/F9 会重新检测"
        else:
            self.status = "延长线显示：开；按 F9 在分析时一并画延长线"
        self._redraw()

    def _update_guide(self) -> None:
        """兼容旧入口：与 _run 共用，不再由 _tick 调用。"""
        self._run()

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

    def on_clear_lock(self) -> None:
        """F6：清除台面锁定，恢复每次自动检测。"""
        self.felt_lock = None
        self.calib = False
        self.click_through = True
        if self.hwnd:
            set_click_through(self.hwnd, True)
        try:
            if FELT_LOCK_PATH.exists():
                FELT_LOCK_PATH.unlink()
        except Exception:
            pass
        self.status = "已清除台面锁定，恢复自动检测"
        self._redraw()

    def on_calibrate(self) -> None:
        """F7：进入/确认台面校准。确认后本盘内不再改台面大小。"""
        if not self.calib:
            # 进入校准：先截一帧，用当前 felt 或自动检测初始化矩形
            self.calib = True
            self.click_through = False
            if self.hwnd:
                set_click_through(self.hwnd, False)
            try:
                img, _ = grab_game_window()
                self._img_size = (img.shape[1], img.shape[0])
                felt = None
                if self.felt_lock is not None:
                    felt = self._felt_from_lock(img.shape[1], img.shape[0])
                if felt is None:
                    from detect import detect_felt

                    felt = detect_felt(img)
                if felt is None:
                    w, h = img.shape[1], img.shape[0]
                    felt = (int(w * 0.12), int(h * 0.18), int(w * 0.76), int(h * 0.64))
                self.calib_rect = [float(v) for v in felt]
            except Exception as e:
                self.status = f"校准截图失败: {e}"
                self.calib = False
                self.click_through = True
                if self.hwnd:
                    set_click_through(self.hwnd, True)
                return
            self.status = "校准中：拖四边或方向键微调；再按 F7 确认锁定"
            self._redraw()
        else:
            # 确认锁定
            self._save_felt_lock()
            self.calib = False
            self.click_through = True
            if self.hwnd:
                set_click_through(self.hwnd, True)
            self.status = "台面已锁定（本盘内 F8/F9 不再改台面）；F6 可清除"
            self._redraw()

    def _load_felt_lock(self) -> None:
        try:
            if FELT_LOCK_PATH.exists():
                data = json.loads(FELT_LOCK_PATH.read_text(encoding="utf-8"))
                rel = data.get("rel")
                if rel and len(rel) == 4:
                    self.felt_lock = tuple(float(v) for v in rel)
        except Exception:
            self.felt_lock = None

    def _save_felt_lock(self) -> None:
        W, H = self._img_size
        if W <= 0 or H <= 0:
            return
        x, y, w, h = self.calib_rect
        rel = (x / W, y / H, w / W, h / H)
        self.felt_lock = rel
        try:
            FELT_LOCK_PATH.parent.mkdir(exist_ok=True)
            FELT_LOCK_PATH.write_text(
                json.dumps({"rel": list(rel), "img_w": W, "img_h": H}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[warn] 保存 felt_lock 失败: {e}")

    def _felt_from_lock(self, W: int, H: int) -> tuple[int, int, int, int] | None:
        if not self.felt_lock:
            return None
        rx, ry, rw, rh = self.felt_lock
        return (int(rx * W), int(ry * H), int(rw * W), int(rh * H))

    def nudge_edge(self, which: str, delta: float) -> None:
        if not self.calib:
            return
        x, y, w, h = self.calib_rect
        if which == "left":
            x2 = x + delta
            w2 = w - delta
            if w2 > 30:
                self.calib_rect = [x2, y, w2, h]
        elif which == "right":
            if w + delta > 30:
                self.calib_rect = [x, y, w + delta, h]
        elif which == "top":
            y2 = y + delta
            h2 = h - delta
            if h2 > 20:
                self.calib_rect = [x, y2, w, h2]
        elif which == "bottom":
            if h + delta > 20:
                self.calib_rect = [x, y, w, h + delta]
        self._redraw()

    def _edge_at(self, ex: float, ey: float) -> str | None:
        x, y, w, h = self.calib_rect
        tol = 18
        near = []
        if abs(ex - x) < tol and y - tol <= ey <= y + h + tol:
            near.append("left")
        if abs(ex - (x + w)) < tol and y - tol <= ey <= y + h + tol:
            near.append("right")
        if abs(ey - y) < tol and x - tol <= ex <= x + w + tol:
            near.append("top")
        if abs(ey - (y + h)) < tol and x - tol <= ex <= x + w + tol:
            near.append("bottom")
        if not near:
            return None
        # 取最近的
        def d(e):
            x, y, w, h = self.calib_rect
            return {
                "left": abs(ex - x),
                "right": abs(ex - (x + w)),
                "top": abs(ey - y),
                "bottom": abs(ey - (y + h)),
            }[e]

        return min(near, key=d)

    def _on_press(self, event) -> None:
        if not self.calib:
            return
        # 底部按钮
        bh = 36
        by = max(0, self.root.winfo_height() - bh)
        if event.y >= by:
            bw = 52
            step = 4
            btns = [
                ("左-", lambda: self.nudge_edge("left", -step)),
                ("左+", lambda: self.nudge_edge("left", step)),
                ("右-", lambda: self.nudge_edge("right", -step)),
                ("右+", lambda: self.nudge_edge("right", step)),
                ("上-", lambda: self.nudge_edge("top", -step)),
                ("上+", lambda: self.nudge_edge("top", step)),
                ("下-", lambda: self.nudge_edge("bottom", -step)),
                ("下+", lambda: self.nudge_edge("bottom", step)),
                ("确认", self.on_calibrate),
            ]
            for i, (label, fn) in enumerate(btns):
                bx = 8 + i * (bw + 4)
                if bx <= event.x <= bx + bw:
                    fn()
                    return
            return
        self._drag_edge = self._edge_at(event.x, event.y)

    def _on_drag(self, event) -> None:
        if not self.calib or not self._drag_edge:
            return
        x, y, w, h = self.calib_rect
        if self._drag_edge == "left":
            nx = min(max(0, event.x), x + w - 30)
            self.calib_rect = [nx, y, x + w - nx, h]
        elif self._drag_edge == "right":
            nw = max(30, event.x - x)
            self.calib_rect = [x, y, nw, h]
        elif self._drag_edge == "top":
            ny = min(max(0, event.y), y + h - 20)
            self.calib_rect = [x, ny, w, y + h - ny]
        elif self._drag_edge == "bottom":
            nh = max(20, event.y - y)
            self.calib_rect = [x, y, w, nh]
        self._redraw()

    def _on_release(self, _event) -> None:
        self._drag_edge = None

    def _on_click(self, event) -> None:
        """校准模式下点击按钮。"""
        if not self.calib:
            return
        # 按钮条在底部
        bh = 36
        by = max(0, self.root.winfo_height() - bh)
        if event.y < by:
            return
        x, y, w, h = self.calib_rect
        step = 4
        bw = 52
        btns = [
            ("左-", lambda: self.nudge_edge("left", -step)),
            ("左+", lambda: self.nudge_edge("left", step)),
            ("右-", lambda: self.nudge_edge("right", -step)),
            ("右+", lambda: self.nudge_edge("right", step)),
            ("上-", lambda: self.nudge_edge("top", -step)),
            ("上+", lambda: self.nudge_edge("top", step)),
            ("下-", lambda: self.nudge_edge("bottom", -step)),
            ("下+", lambda: self.nudge_edge("bottom", step)),
            ("确认", self.on_calibrate),
        ]
        for i, (label, fn) in enumerate(btns):
            bx = 8 + i * (bw + 4)
            if bx <= event.x <= bx + bw:
                fn()
                return

    def _run(self) -> None:
        try:
            img, meta = grab_game_window()
        except Exception as e:
            self.status = f"截图失败: {e}"
            self._draw_status()
            return
        self._img_size = (img.shape[1], img.shape[0])
        felt_ov = self._felt_from_lock(img.shape[1], img.shape[0])
        lock_note = "台面锁定" if felt_ov else "台面自动"
        self.status = (
            f"分析中… [{meta.get('method','')}] {lock_note} 目标={self.target_color} 模式={self.mode}"
        )
        self._draw_status()
        self.root.update()
        try:
            result = analyze(
                img,
                target_color=self.target_color,
                mode=self.mode,
                max_cushions=2,
                felt_override=felt_ov,
            )
        except Exception as e:
            self.status = f"分析失败: {e}"
            self.last_result = None
            self._redraw()
            return
        self.last_result = result
        if result.state.ok():
            self._last_felt = result.state.felt_rect
            # 本帧延长线：与分析同一张图，不另开循环
            try:
                cue = None
                if result.cue:
                    fx, fy, fw, fh = result.state.felt_rect
                    cue = (fx + result.cue[0] * fw, fy + result.cue[1] * fh)
                pts = detect_aim_line_window(img, result.state.felt_rect, cue=cue)
                self.guide_pts = pts if pts and len(pts) >= 2 else None
            except Exception:
                self.guide_pts = None
        n = len(result.plans)
        best = result.plans[0] if result.plans else None
        cush = "-".join(best.cushions) if best and best.cushions else (best.kind if best else "-")
        sc = f"{best.score:.0f}" if best is not None else "-"
        gnote = "延长线OK" if self.guide_pts else "无延长线"
        self.status = (
            f"{result.message} | 首选:{cush} 分{sc} | {gnote} | 目标={self.target_color}"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.calib:
            self._draw_calib()
            self._draw_status()
            return

        # F8/F9 分析时得到的延长线（F5 可隐藏）
        if self.live_guide and self.guide_pts and len(self.guide_pts) >= 2:
            a, b = self.guide_pts[0], self.guide_pts[1]
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill="#FFE040", width=3)
            self.canvas.create_oval(a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5, outline="#FFE040", width=2)

        if not self.show:
            self._draw_status()
            return
        r = self.last_result
        if r is None or not r.state.ok():
            self._draw_status()
            return

        fx, fy, fw, fh = r.state.felt_rect
        ox, oy = 0, 0

        def to_xy(nx, ny):
            return (ox + fx + nx * fw, oy + fy + ny * fh)

        self.canvas.create_rectangle(fx, fy, fx + fw, fy + fh, outline="#00C8C8", width=1)

        # 袋口：半径可在 config.POCKET_DRAW_RADIUS_RATIO 调
        pr = max(8, int(fh * POCKET_DRAW_RADIUS_RATIO))
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

    def _draw_calib(self) -> None:
        x, y, w, h = self.calib_rect
        self.canvas.create_rectangle(x, y, x + w, y + h, outline="#00FF88", width=3)
        # 边中点手柄
        handles = [
            (x, y + h / 2),
            (x + w, y + h / 2),
            (x + w / 2, y),
            (x + w / 2, y + h),
        ]
        for hx, hy in handles:
            self.canvas.create_rectangle(hx - 8, hy - 8, hx + 8, hy + 8, fill="#00FF88", outline="")
        # 袋口预览（按最终矩形生成）
        pr = max(8, int(h * POCKET_DRAW_RADIUS_RATIO))
        for nx, ny in ((0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)):
            px, py = x + nx * w, y + ny * h
            self.canvas.create_oval(px - pr, py - pr, px + pr, py + pr, outline="#FF8C00", width=2)
        # 按钮
        bh = 36
        by = max(0, self.root.winfo_height() - bh)
        self.canvas.create_rectangle(0, by, self.root.winfo_width(), by + bh, fill="#202020", outline="")
        labels = ["左-", "左+", "右-", "右+", "上-", "上+", "下-", "下+", "确认"]
        bw = 52
        for i, lab in enumerate(labels):
            bx = 8 + i * (bw + 4)
            self.canvas.create_rectangle(bx, by + 4, bx + bw, by + bh - 4, fill="#404040", outline="#888")
            self.canvas.create_text(bx + bw / 2, by + bh / 2, text=lab, fill="#FFF", font=("Segoe UI", 10))
        self.canvas.create_text(
            x + w / 2,
            max(16, y - 14),
            text="拖动四边 / 点按钮微调 / F7确认 / F6清除锁定",
            fill="#00FF88",
            font=("Segoe UI", 11),
        )

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
        print("  F9 进球分析（同时画出游戏瞄准延长线）")
        print("  F8 解球/进球综合分析")
        print("  F5 显示/隐藏最近延长线")
        print("  F6 清台面锁定  F7 校准/锁定台面")
        print("  F10 切换目标  F11 显隐  F12/Esc 退出")
        print("  不再后台实时抓帧；按 F8/F9 时截一帧并标注")
        self.root.mainloop()


def main() -> int:
    app = OverlayApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
