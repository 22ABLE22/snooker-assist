# -*- coding: utf-8 -*-
"""检查 dev/ 与 tests/ 里引用的样本图片是否都存在（防止死引用）。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
miss = 0
checked = 0

for f in sorted(list((ROOT / "dev").glob("*.py")) + list((ROOT / "tests").glob("*.py"))):
    if f.name == Path(__file__).name:
        continue
    for i, ln in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
        # 形如 "samples/xxx/yyy.png"
        for m in re.finditer(r'"(samples/[^"]+\.(?:png|json))"', ln):
            checked += 1
            if not (ROOT / m.group(1)).exists():
                print(f"  缺失 {f.name}:{i}  {m.group(1)}")
                miss += 1
        # 形如 ROOT / "samples" / "xxx" / "yyy.png"
        for m in re.finditer(
            r'ROOT\s*/\s*"samples"\s*/\s*"([^"]+)"\s*/\s*"([^"]+)"', ln
        ):
            checked += 1
            p = ROOT / "samples" / m.group(1) / m.group(2)
            if not p.exists() and "*" not in m.group(2):
                print(f"  缺失(拼接) {f.name}:{i}  samples/{m.group(1)}/{m.group(2)}")
                miss += 1

print(f"检查 {checked} 处引用，" + ("全部存在" if miss == 0 else f"{miss} 处缺失"))
