# -*- coding: utf-8 -*-
"""兼容入口：转发到 overlay.py

旧用法: python hotkey.py
新入口: python overlay.py
"""

from overlay import main

if __name__ == "__main__":
    raise SystemExit(main())
