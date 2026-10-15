# -*- coding: utf-8 -*-
"""截图 / 窗口捕获"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mss
import numpy as np

from config import OUT_DIR, SNAPSHOT_PREFIX, TABLE_REGION, WINDOW_TITLE_HINTS

_DPI_AWARE_SET = False


def _ensure_dpi_aware() -> None:
    """避免 DPI 虚拟化导致 PrintWindow/GetClientRect 尺寸不一致。"""
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET:
        return
    _DPI_AWARE_SET = True
    try:
        import ctypes

        # Per-monitor V2
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def grab_screen(region: dict | None = None) -> np.ndarray:
    """截取全屏或指定区域，返回 BGR ndarray。"""
    _ensure_dpi_aware()
    with mss.mss() as sct:
        if region:
            monitor = {
                "left": int(region["left"]),
                "top": int(region["top"]),
                "width": int(region["width"]),
                "height": int(region["height"]),
            }
        else:
            monitor = sct.monitors[1]  # 主显示器
        shot = sct.grab(monitor)
        img = np.array(shot)
        # mss 返回 BGRA
        return img[:, :, :3].copy()


def grab_default_table() -> np.ndarray:
    """按 config.TABLE_REGION 截取台面。"""
    return grab_screen(TABLE_REGION)


def save_snapshot(img_bgr: np.ndarray, tag: str = "") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{SNAPSHOT_PREFIX}_{tag}_{ts}.png" if tag else f"{SNAPSHOT_PREFIX}_{ts}.png"
    path = OUT_DIR / name
    cv2.imwrite(str(path), img_bgr)
    return path


def _title_match_score(title: str, hints: list[str]) -> float:
    t = title.strip().lower()
    # 排除本工具/编辑器窗口，避免会话标题里含「腾讯桌球」时误匹配
    bad = ("mimo", "mimocode", "visual studio code", "pycharm", "记事本", "notepad")
    if any(b in t for b in bad):
        return 0.0
    if title.strip() in ("腾讯桌球", "T-BALL"):
        return 10.0
    for h in hints:
        if h and h.lower() in t:
            return 5.0 + max(0, 30 - len(title)) * 0.05
    return 0.0


def find_window_rect(title_hints: list[str] | None = None) -> dict | None:
    """尝试用 win32 找到目标窗口；失败则返回 None。返回含 hwnd。"""
    _ensure_dpi_aware()
    hints = title_hints or WINDOW_TITLE_HINTS
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found = {"best": None, "score": 0.0}

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            score = _title_match_score(title, hints)
            if score <= 0:
                return True
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w < 400 or h < 300:
                return True
            if score > found["score"]:
                found["score"] = score
                found["best"] = {
                    "left": rect.left,
                    "top": rect.top,
                    "width": w,
                    "height": h,
                    "title": title,
                    "hwnd": int(hwnd),
                }
            return True

        user32.EnumWindows(enum_proc, 0)
        return found["best"]
    except Exception:
        return None


def _bits_to_bgr(buf, w: int, h: int) -> np.ndarray | None:
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < w * h * 4:
        return None
    arr = arr[: w * h * 4].reshape((h, w, 4))
    return arr[:, :, :3].copy()


def _bmi_header(w: int, h: int):
    import ctypes
    from ctypes import wintypes

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    return bmi


def print_window_bgr(hwnd: int, width: int | None = None, height: int | None = None) -> np.ndarray | None:
    """PrintWindow 截窗口内容（BGR）。优先用窗口矩形物理像素尺寸。"""
    _ensure_dpi_aware()
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        hwnd = wintypes.HWND(hwnd)
        wr = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(wr)):
            return None
        w = wr.right - wr.left
        h = wr.bottom - wr.top
        if w <= 0 or h <= 0:
            return None

        # 兼容 DC 用桌面 DC，避免某些窗口 GetWindowDC 返回异常
        screenDC = user32.GetDC(0)
        if not screenDC:
            return None
        mfcDC = gdi32.CreateCompatibleDC(screenDC)
        if not mfcDC:
            user32.ReleaseDC(0, screenDC)
            return None
        bitmap = gdi32.CreateCompatibleBitmap(screenDC, w, h)
        if not bitmap:
            gdi32.DeleteDC(mfcDC)
            user32.ReleaseDC(0, screenDC)
            return None
        gdi32.SelectObject(mfcDC, bitmap)

        PW_RENDERFULLCONTENT = 2
        ok = user32.PrintWindow(hwnd, mfcDC, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(hwnd, mfcDC, 0)

        bmi = _bmi_header(w, h)
        buf = ctypes.create_string_buffer(w * h * 4)
        bits = gdi32.GetDIBits(mfcDC, bitmap, 0, h, buf, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mfcDC)
        user32.ReleaseDC(0, screenDC)

        if bits == 0:
            return None
        img = _bits_to_bgr(buf, w, h)
        if img is None:
            return None
        # 纯黑/全空视为失败
        if float(img.std()) < 2.0:
            return None
        return img
    except Exception:
        return None


def bitblt_window_bgr(hwnd: int) -> np.ndarray | None:
    """BitBlt 截窗口 DC（有时比 PrintWindow 更完整）。"""
    _ensure_dpi_aware()
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        hwnd = wintypes.HWND(hwnd)
        wr = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(wr)):
            return None
        w = wr.right - wr.left
        h = wr.bottom - wr.top
        if w <= 0 or h <= 0:
            return None

        hdc = user32.GetWindowDC(hwnd)
        if not hdc:
            return None
        mdc = gdi32.CreateCompatibleDC(hdc)
        if not mdc:
            user32.ReleaseDC(hwnd, hdc)
            return None
        bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
        if not bmp:
            gdi32.DeleteDC(mdc)
            user32.ReleaseDC(hwnd, hdc)
            return None
        gdi32.SelectObject(mdc, bmp)
        SRCCOPY = 0x00CC0020
        gdi32.BitBlt(mdc, 0, 0, w, h, hdc, 0, 0, SRCCOPY)

        bmi = _bmi_header(w, h)
        buf = ctypes.create_string_buffer(w * h * 4)
        bits = gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(hwnd, hdc)
        if bits == 0:
            return None
        img = _bits_to_bgr(buf, w, h)
        if img is None or float(img.std()) < 2.0:
            return None
        return img
    except Exception:
        return None


def raise_window(hwnd: int) -> None:
    _ensure_dpi_aware()
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.2)
    except Exception:
        pass


def screen_capture_window(info: dict) -> np.ndarray | None:
    """把窗口置前后按屏幕区域截取。"""
    try:
        if info.get("hwnd"):
            raise_window(int(info["hwnd"]))
        return grab_screen(
            {
                "left": info["left"],
                "top": info["top"],
                "width": info["width"],
                "height": info["height"],
            }
        )
    except Exception:
        return None


def felt_quality(img_bgr: np.ndarray) -> float:
    """评估截图像不像完整俯视球台：绿占比 + 长宽比接近 2:1 + 台面面积。"""
    if img_bgr is None or img_bgr.size == 0:
        return -1.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 55, 35), (95, 255, 210))
    green = float(mask.mean()) / 255.0
    if green < 0.05:
        return -1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return green * 0.1
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:
        return green * 0.1
    aspect = w / max(h, 1)
    # 标准台约 2.0；截残时 aspect 往往偏小
    aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 1.2)
    area_score = (w * h) / float(img_bgr.shape[0] * img_bgr.shape[1])
    # 矩形内绿色密度
    inner = mask[y : y + h, x : x + w]
    density = float(inner.mean()) / 255.0 if inner.size else 0.0
    return 0.35 * green + 0.35 * aspect_score + 0.20 * area_score + 0.10 * density


def grab_game_window() -> tuple[np.ndarray, dict]:
    """多路截取游戏窗口，自动选「台面最完整」的一张。"""
    _ensure_dpi_aware()
    info = find_window_rect()
    if not info or not info.get("hwnd"):
        full = grab_screen()
        h, w = full.shape[:2]
        return full, {"left": 0, "top": 0, "width": w, "height": h, "title": "fullscreen"}

    hwnd = int(info["hwnd"])
    candidates: list[tuple[str, np.ndarray]] = []

    img = print_window_bgr(hwnd)
    if img is not None:
        candidates.append(("printwindow", img))

    img = bitblt_window_bgr(hwnd)
    if img is not None:
        candidates.append(("bitblt", img))

    img = screen_capture_window(info)
    if img is not None:
        candidates.append(("screen", img))

    if not candidates:
        full = grab_screen()
        h, w = full.shape[:2]
        return full, {**info, "method": "fullscreen", "width": w, "height": h, "left": 0, "top": 0}

    scored = []
    for name, im in candidates:
        q = felt_quality(im)
        scored.append((q, name, im, im.shape[1], im.shape[0]))
        print(f"[capture] {name}: {im.shape[1]}x{im.shape[0]} quality={q:.3f}")

    scored.sort(key=lambda t: -t[0])
    q, name, im, w, h = scored[0]
    meta = {
        **info,
        "method": name,
        "quality": q,
        "captured_width": w,
        "captured_height": h,
    }
    print(f"[capture] 选用 {name} {w}x{h} quality={q:.3f}")
    return im, meta


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def to_pil(bgr: np.ndarray):
    from PIL import Image

    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)
