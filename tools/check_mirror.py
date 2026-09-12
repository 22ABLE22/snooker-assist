# -*- coding: utf-8 -*-
"""核查吃库镜面：是否边线内移半径、入射角=反射角。"""
import math
import sys

sys.path.insert(0, r"E:\Users\Admin\SNOOKER")
from geometry import plan_cue_bank, cushion_axis_value, radii_xy, set_table_aspect
from config import TABLE_ASPECT

set_table_aspect(None)
asp = TABLE_ASPECT
r = 0.02
rx, ry = radii_xy(r, asp)
print("=== 镜面几何核查 ===")
print(f"TABLE_ASPECT={asp:.4f}  r={r}  rx={rx:.5f}  ry={ry:.5f}")
print(f"left  镜面 x={cushion_axis_value('left', r):.5f}  (=rx)")
print(f"right 镜面 x={cushion_axis_value('right', r):.5f}  (=1-rx)")
print(f"top   镜面 y={cushion_axis_value('top', r):.5f}  (=ry)")
print(f"bottom镜面 y={cushion_axis_value('bottom', r):.5f}  (=1-ry)")

cue, target = (0.15, 0.70), (0.75, 0.35)
p = plan_cue_bank(cue, target, ["left"], r, asp)
pts = p.points
print("\n一库 left 折点:", [(round(x, 5), round(y, 5)) for x, y in pts])
print("吃库点 x 应为 rx =", round(rx, 5), "实际", round(pts[1][0], 5))
assert abs(pts[1][0] - rx) < 1e-9


def phys(v):
    return (v[0] * asp, v[1])


v1 = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
v2 = pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]
p1, p2 = phys(v1), phys(v2)
n1 = math.hypot(*p1)
n2 = math.hypot(*p2)
cos_i = abs(p1[0]) / n1
cos_r = abs(p2[0]) / n2
print(f"入射 cos={cos_i:.6f}  反射 cos={cos_r:.6f}  差={abs(cos_i - cos_r):.2e}")
assert abs(cos_i - cos_r) < 1e-9

p2c = plan_cue_bank(cue, (0.7, 0.2), ["left", "top"], r, asp)
print("\n二库 left>top 折点:", [(round(x, 5), round(y, 5)) for x, y in p2c.points])
print("第1库 x≈rx", round(rx, 5), "第2库 y≈ry", round(ry, 5))
assert abs(p2c.points[1][0] - rx) < 1e-6
assert abs(p2c.points[2][1] - ry) < 1e-6
print("\n结论: 镜面=边线内移半径（长短边各向异性），入射=反射，几何正确")
