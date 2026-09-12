# -*- coding: utf-8 -*-
"""几何单元测试：python test_geometry.py"""

from __future__ import annotations

import math
import sys

from geometry import (
    fold_path,
    ghost_ball,
    plan_cue_bank,
    plan_direct_cue_to_target,
    plan_object_bank,
    reflect_point,
    reflect_seq,
    find_escape_paths_ranked,
)


def almost(a, b, eps=1e-6):
    return abs(a - b) < eps


def test_reflect_point():
    assert almost(reflect_point((0.3, 0.4), "top")[1], -0.4)
    assert almost(reflect_point((0.3, 0.4), "bottom")[1], 1.6)
    assert almost(reflect_point((0.3, 0.4), "left")[0], -0.3)
    assert almost(reflect_point((0.3, 0.4), "right")[0], 1.7)
    # 球心镜面：left 内移 rx = r/aspect，top 内移 ry = r
    from geometry import radii_xy
    from config import TABLE_ASPECT
    r = 0.02
    rx, ry = radii_xy(r)
    assert almost(rx, r / TABLE_ASPECT)
    assert almost(ry, r)
    assert almost(reflect_point((0.10, 0.5), "left", r)[0], 2 * rx - 0.10)
    assert almost(reflect_point((0.5, 0.10), "top", r)[1], 2 * ry - 0.10)
    print("OK reflect_point anisotropic", "rx", round(rx, 4), "ry", round(ry, 4))


def test_one_cushion_angle():
    cue = (0.15, 0.70)
    target = (0.75, 0.35)
    r = 0.015
    from geometry import radii_xy
    rx, ry = radii_xy(r)
    p = plan_cue_bank(cue, target, ["left"], r)
    assert p is not None
    a = p.points
    assert len(a) == 3
    v1 = (a[1][0] - a[0][0], a[1][1] - a[0][1])
    v2 = (a[2][0] - a[1][0], a[2][1] - a[1][1])
    # 物理方向上的入射角=反射角：用 aspect 加权 x
    from config import TABLE_ASPECT
    def phys(v):
        return (v[0] * TABLE_ASPECT, v[1])
    p1, p2 = phys(v1), phys(v2)
    n1 = math.hypot(*p1)
    n2 = math.hypot(*p2)
    assert almost(abs(p1[0]) / n1, abs(p2[0]) / n2, 1e-4)
    # 触库点在球心镜面 x=rx（不是 r，也不是 2r）
    assert almost(a[1][0], rx, 1e-5)
    print("OK one cushion bounce at rx", a[1][0], "rx", rx)


def test_two_cushion_fold():
    cue = (0.2, 0.8)
    target = (0.8, 0.2)
    seq = ["left", "top"]
    r = 0.02
    from geometry import radii_xy
    rx, ry = radii_xy(r)
    vt = reflect_seq(target, seq, r)
    pts = fold_path(cue, vt, seq, r)
    assert pts and len(pts) == 4
    assert almost(pts[1][0], rx, 1e-4)
    assert almost(pts[2][1], ry, 1e-4)
    assert almost(pts[3][0], target[0], 0.05)
    assert almost(pts[3][1], target[1], 0.05)
    print("OK two cushion fold", pts)


def test_pot_angle_rejects_acute():
    from geometry import pot_angle_ok, find_pot_paths

    # 目标在袋口同侧，白球也在同侧 → 锐角，不应有直接进球
    target = (0.5, 0.15)
    pocket = (0.5, 0.0)
    cue = (0.5, 0.30)  # 白球在目标与库之间？ cue y=0.3, t y=0.15, pk y=0
    # target→cue = +y, target→pocket = -y → 180°，合法
    assert pot_angle_ok(cue, target, pocket)

    # 白球在袋口更远外侧（不可能的锐角）：cue 在 pocket 再往外
    cue_bad = (0.5, -0.05)  # 出界但用于角检测：target→cue 同向 target→pocket
    assert not pot_angle_ok(cue_bad, target, pocket)

    # 侧面薄球：白球几乎在袋口正上方同侧切
    cue_thin = (0.85, 0.05)
    # target (0.5,0.15)→cue (0.85,0.05) vs →pocket (0.5,0)
    # 可能锐角
    ok = pot_angle_ok(cue_thin, target, pocket)
    print("thin cut angle ok?", ok)

    plans = find_pot_paths(
        cue_bad, [target], [pocket], "red", [], 0.015, include_object_bank=False
    )
    # 出界白球不应产生合法直接进球
    assert not plans or all(not (p.kind == "direct") for p in plans)
    print("OK pot angle filter")


def test_cue_bank_pot_exists():
    from geometry import find_pot_paths

    cue = (0.15, 0.75)
    target = (0.55, 0.35)
    pockets = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    r = 0.015
    plans = find_pot_paths(
        cue, [target], pockets, "red", [], r, include_object_bank=True, max_cushions=1
    )
    kinds = {p.kind for p in plans}
    print("pot kinds", kinds, "n", len(plans))
    assert "direct" in kinds or "cue_bank_pot" in kinds
    # 直接进球终点必须是袋心
    for p in plans:
        if p.kind == "direct":
            assert almost(p.points[-1][0], 0.5) or almost(p.points[-1][0], 0.0) or almost(p.points[-1][0], 1.0)
    print("OK cue bank / pot plans")


def test_direct_blocked():
    # 蓝球挡住直线
    cue = (0.15, 0.70)
    target = (0.75, 0.35)
    blocker = (0.45, 0.52)
    plans = find_escape_paths_ranked(cue, [target], "red", [blocker], 0.015, max_cushions=1)
    kinds = {p.kind for p in plans}
    assert "direct" not in kinds
    assert any(p.n_cushions == 1 for p in plans)
    print("OK direct blocked, banks found", len(plans))


def test_near_cue_blocker():
    # 黄球紧贴白球且在直线上，直线解必须被否决
    r = 0.015
    cue = (0.20, 0.50)
    yellow = (0.25, 0.50)  # 距白球 0.05 ≈ 3.3r，在正右方
    red = (0.70, 0.50)
    plans = find_escape_paths_ranked(cue, [red], "red", [yellow], r, max_cushions=2)
    kinds = {p.kind for p in plans}
    assert "direct" not in kinds, f"直线解不应通过，得到 {kinds}"
    assert plans, "至少应有一库解"
    print("OK near-cue yellow blocks direct;", "banks:", len(plans))


def test_shoot_away_from_touching_ball():
    # 白球几乎贴黄球，但目标在另一侧：向左吃库应可行
    r = 0.015
    cue = (0.20, 0.50)
    yellow = (0.24, 0.52)  # 右侧紧贴
    red = (0.15, 0.20)     # 左上方
    plans = find_escape_paths_ranked(cue, [red], "red", [yellow], r, max_cushions=1)
    # 至少不该被「贴身黄球」全部否决
    assert plans, "贴身球不应挡住反方向出杆"
    # 直线可能可通或被否，但应存在方案
    print("OK shoot away from touching ball;", len(plans), "plans, kinds",
          [p.kind for p in plans[:3]])



def test_same_color_other_red_blocks():
    # 打红 A 时，红 B 挡路应否决直线
    r = 0.015
    cue = (0.10, 0.50)
    red_a = (0.80, 0.50)
    red_b = (0.45, 0.50)
    plans = find_escape_paths_ranked(cue, [red_a], "red", [red_a, red_b], r, max_cushions=2)
    kinds = {p.kind for p in plans}
    assert "direct" not in kinds
    print("OK other red on line blocks direct")



def test_pocket_risk_filters_middle():
    # 吃库点压在上中袋 → 应被过滤
    from geometry import plan_cue_bank, path_pocket_risk, find_escape_paths_ranked

    r = 0.015
    cue = (0.30, 0.70)
    # 目标在右上，left-top 折线容易打到上中袋附近
    red = (0.70, 0.12)
    plans = find_escape_paths_ranked(cue, [red], "red", [], r, max_cushions=2)
    for p in plans:
        risky, why = path_pocket_risk(p)
        assert not risky, f"不应保留危险线路 {p.cushions}: {why}"

    # 人工构造危险 plan：第二库点在 (0.5, 0)
    p = plan_cue_bank(cue, red, ["left", "top"], r)
    assert p is not None
    # 手改折点压中袋
    p.points = [(0.30, 0.70), (0.0, 0.40), (0.50, 0.0), (0.70, 0.12)]
    risky, why = path_pocket_risk(p)
    assert risky, f"中袋附近吃库应判危险, got {why}"
    print("OK pocket risk filters middle pocket")


def test_ghost_ball():
    t = (0.5, 0.5)
    pk = (1.0, 0.5)
    r = 0.02
    from geometry import radii_xy
    rx, ry = radii_xy(r)
    g = ghost_ball(t, pk, r)
    # 沿 +x 方向袋：幽灵球在目标左侧 2*rx
    assert almost(g[0], 0.5 - 2 * rx)
    assert almost(g[1], 0.5)
    print("OK ghost ball", g, "rx", rx)


def test_object_bank():
    cue = (0.2, 0.8)
    target = (0.6, 0.4)
    pocket = (0.98, 0.98)  # 角袋中心（可略内收用作几何袋心）
    p = plan_object_bank(cue, target, pocket, ["right"], 0.015)
    assert p is not None
    # 终点必须是袋口中心
    assert almost(p.points[-1][0], pocket[0], 1e-5)
    assert almost(p.points[-1][1], pocket[1], 1e-5)
    print("OK object bank", p.points, p.cushions)


def test_pot_middle_pocket_center():
    from geometry import find_pot_paths

    cue = (0.2, 0.7)
    target = (0.5, 0.25)
    pockets = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    r = 0.015
    plans = find_pot_paths(cue, [target], pockets, "red", [], r, include_object_bank=False)
    assert plans, "应能找到中袋进球线"
    # 找中袋方案
    mid = [p for p in plans if abs(p.points[-1][0] - 0.5) < 1e-6 and abs(p.points[-1][1] - 0.0) < 1e-6]
    assert mid, f"应存在终点为上中袋中心的方案，得到终点 {plans[0].points[-1]}"
    print("OK middle pot aims at pocket center", mid[0].points[-1])


def test_path_corridor_anisotropic():
    """路径走廊：物理距离判定；左右方向半宽为 2r/aspect。"""
    from geometry import path_corridor_blocks, radii_xy
    from config import TABLE_ASPECT

    r = 0.02
    rx, ry = radii_xy(r)
    # 水平路径 y=0.5，从 x=0.1 到 0.9
    a, b = (0.1, 0.5), (0.9, 0.5)
    # 垂直偏移 ry*0.5 < 2r → 应挡住
    assert path_corridor_blocks(a, b, (0.5, 0.5 + ry), r=r)
    # 垂直偏移 3r > 2r → 不挡
    assert not path_corridor_blocks(a, b, (0.5, 0.5 + 3 * r), r=r)
    # 沿路径前方但侧向超出
    assert not path_corridor_blocks(a, b, (0.5, 0.5 + 2.5 * r), r=r)
    print("OK path corridor anisotropic")


def test_snooker_score_prefers_easy_pot():
    """近袋、直球应优于远袋、薄切；直接优于翻袋。"""
    from geometry import PathPlan, score_pot_plan, score_escape_plan, sort_plans

    t_near = (0.55, 0.12)
    t_far = (0.55, 0.75)
    pk = (0.5, 0.0)
    r = 0.02
    # 近袋直球
    near = PathPlan(
        kind="direct", cushions=[], points=[(0.2, 0.5), (0.5, 0.3), t_near, pk],
        aim_dir=(1, 0), impact_point=(0.5, 0.3), length=0.8,
        target_color="red", target_index=0,
        target_pos=t_near, pocket_pos=pk,
    )
    far = PathPlan(
        kind="direct", cushions=[], points=[(0.2, 0.5), (0.3, 0.7), t_far, pk],
        aim_dir=(1, 0), impact_point=(0.3, 0.7), length=1.2,
        target_color="red", target_index=1,
        target_pos=t_far, pocket_pos=pk,
    )
    bank = PathPlan(
        kind="object_bank", cushions=["left"],
        points=[(0.2, 0.5), (0.1, 0.4), t_near, (0.0, 0.3), pk],
        aim_dir=(1, 0), impact_point=(0.1, 0.4), length=1.5,
        target_color="red", target_index=0,
        target_pos=t_near, pocket_pos=pk,
    )
    s_near, s_far, s_bank = score_pot_plan(near), score_pot_plan(far), score_pot_plan(bank)
    print(f"score near={s_near:.1f} far={s_far:.1f} bank={s_bank:.1f}")
    assert s_near < s_far, "近袋直球应更优"
    assert s_near < s_bank, "直接应优于翻袋"

    # 解球：一库优于二库
    e1 = PathPlan(kind="cue_bank", cushions=["left"], points=[(0.2,0.5),(0,0.4),(0.8,0.2)],
                  aim_dir=(1,0), impact_point=(0.78,0.2), length=1.0, target_color="red", target_index=0)
    e2 = PathPlan(kind="cue_bank", cushions=["left","top"], points=[(0.2,0.5),(0,0.4),(0.5,0),(0.8,0.2)],
                  aim_dir=(1,0), impact_point=(0.78,0.2), length=1.3, target_color="red", target_index=0)
    assert score_escape_plan(e1) < score_escape_plan(e2)
    print("OK snooker score ordering")


def test_multicushion_midpath_hits_target():
    """两库之间若白球路径扫过目标球，应判无效（只有最后一段可碰目标）。"""
    from geometry import PathPlan, path_blocked

    r = 0.02
    cue = (0.05, 0.60)
    target = (0.50, 0.50)
    # 中间库段 y=0.5 直接穿过目标球心
    p = PathPlan(
        kind="cue_bank",
        cushions=["left", "right"],
        points=[cue, (0.10, 0.50), (0.90, 0.50), target],
        aim_dir=(1.0, 0.0),
        impact_point=(target[0] - 2 * r, target[1]),
        length=2.0,
        target_color="red",
        target_index=0,
    )
    blocked, reason = path_blocked(p, cue, target, [], r)
    print("midpath through target:", blocked, reason)
    assert blocked, "第一库与第二库之间碰到目标球应无效"

    # 仅一库、末段去目标：允许
    p1 = PathPlan(
        kind="cue_bank",
        cushions=["left"],
        points=[cue, (0.0, 0.55), target],
        aim_dir=(-1, 0),
        impact_point=(target[0] - 2 * r, target[1]),
        length=1.0,
        target_color="red",
        target_index=0,
    )
    blocked1, _ = path_blocked(p1, cue, target, [], r)
    # 末段 (0,0.55)→impact 不穿过目标球体内部（impact 在目标前 2r）
    # 可能被目标挡？ allow_target=True 应不挡
    print("final approach blocked?", blocked1)
    assert not blocked1, "最终触球段不应因目标球被否"
    print("OK multi-cushion midpath target collision")


def test_plant_only_red_to_red():
    """斯诺克：仅打红时允许红传红；打彩球禁止传球。"""
    from geometry import find_pot_paths

    cue = (0.15, 0.75)
    # A=红, B=蓝, 另一颗红
    A = (0.40, 0.50)
    B_blue = (0.60, 0.35)
    B_red = (0.62, 0.38)
    pockets = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    r = 0.015

    # 打红：障碍含蓝与另一红 → 只应出现红传红 plant
    others = [A, B_blue, B_red]
    cols = ["red", "blue", "red"]
    plans = find_pot_paths(
        cue, [A], pockets, "red", others, r,
        include_object_bank=False, max_cushions=1, other_colors=cols,
    )
    plants = [p for p in plans if p.kind == "plant"]
    for p in plants:
        # 入袋球 target_pos 必须是红（B_red）
        assert p.target_pos is not None
        dx = abs(p.target_pos[0] - B_red[0]) + abs(p.target_pos[1] - B_red[1])
        dx_blue = abs(p.target_pos[0] - B_blue[0]) + abs(p.target_pos[1] - B_blue[1])
        assert dx < dx_blue, f"plant 入袋球不应是蓝球: {p.target_pos}"
    print("OK plant red-red only when on reds, n_plants", len(plants))

    # 打绿：禁止任何 plant
    green = (0.25, 0.40)
    plans_g = find_pot_paths(
        cue, [green], pockets, "green", [A, B_blue], r,
        include_object_bank=False, max_cushions=1, other_colors=["red", "blue"],
    )
    kinds = {p.kind for p in plans_g}
    assert "plant" not in kinds, f"打绿不应有 plant: {kinds}"
    print("OK no plant when on colour", kinds)


def test_corner_aim_offset():
    """贴库打角袋：瞄准点沿另一库边外偏一个球半径。"""
    from geometry import corner_aim_point, corner_jaw_angles_deg, radii_xy
    from config import TABLE_ASPECT

    r = 0.02
    rx, ry = radii_xy(r)
    # 目标紧贴左侧，打左上角 (0,0) → 与短库夹角 <15°，应沿上库右移
    t_left = (0.04, 0.50)
    ang_long, ang_short = corner_jaw_angles_deg(t_left, (0.0, 0.0))
    print("left-side to TL: ang_long", round(ang_long, 1), "ang_short", round(ang_short, 1))
    aim = corner_aim_point(t_left, (0.0, 0.0), r)
    print("aim", aim, "expect x≈rx", round(rx, 4))
    assert abs(aim[0] - rx) < 1e-6 and abs(aim[1] - 0.0) < 1e-6

    # 横着打右上角 (1,0)：几乎平行上库 → 沿右库下移
    t_horiz = (0.50, 0.03)
    ang_long, ang_short = corner_jaw_angles_deg(t_horiz, (1.0, 0.0))
    print("horiz to TR: ang_long", round(ang_long, 1), "ang_short", round(ang_short, 1))
    aim2 = corner_aim_point(t_horiz, (1.0, 0.0), r)
    print("aim2", aim2, "expect y≈ry", round(ry, 4))
    assert abs(aim2[0] - 1.0) < 1e-6 and abs(aim2[1] - ry) < 1e-6

    # 正对角袋：两边都大，不偏移
    t_diag = (0.30, 0.30)
    aim3 = corner_aim_point(t_diag, (0.0, 0.0), r)
    assert abs(aim3[0]) < 1e-9 and abs(aim3[1]) < 1e-9
    print("OK corner aim offset")


def test_middle_pocket_rail_angle():
    """中袋：进球线几乎平行长库（夹角<15°）应否决。"""
    from geometry import (
        middle_pocket_rail_angle_deg,
        middle_pocket_rail_ok,
        find_pot_paths,
    )
    from config import MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG

    pk = (0.5, 0.0)  # 上中袋
    # 正对中袋：目标在袋口下方 → 与长库 90°
    t_ok = (0.5, 0.25)
    ang_ok = middle_pocket_rail_angle_deg(t_ok, pk)
    print("angle straight-to-mid", round(ang_ok, 2))
    assert ang_ok >= 80
    assert middle_pocket_rail_ok(t_ok, pk)

    # 几乎沿长库擦向中袋：目标在左侧很近库 → 夹角很小
    t_bad = (0.20, 0.03)
    ang_bad = middle_pocket_rail_angle_deg(t_bad, pk)
    print("angle shallow-to-mid", round(ang_bad, 2))
    assert ang_bad < MIDDLE_POCKET_MIN_RAIL_ANGLE_DEG
    assert not middle_pocket_rail_ok(t_bad, pk)

    # 角袋不受此限
    assert middle_pocket_rail_ok(t_bad, (0.0, 0.0))

    # 端到端：浅角目标不应生成中袋 direct
    cue = (0.15, 0.5)
    plans = find_pot_paths(
        cue, [t_bad], [pk, (0.0, 0.0), (1.0, 0.0)], "red", [],
        0.015, include_object_bank=False,
    )
    mid_direct = [
        p for p in plans
        if p.kind == "direct" and abs(p.pocket_pos[0] - 0.5) < 1e-6 and p.pocket_pos[1] < 0.1
    ]
    assert not mid_direct, f"浅角不应进中袋: {mid_direct}"
    print("OK middle pocket rail angle filter")


def test_measured_aspect_override():
    """set_table_aspect 后 radii_xy / 走廊应使用实测 aspect。"""
    from geometry import radii_xy, get_aspect, set_table_aspect, path_corridor_blocks
    from config import TABLE_ASPECT

    set_table_aspect(None)
    assert abs(get_aspect(None) - TABLE_ASPECT) < 1e-9

    # 模拟更扁的台面 aspect=1.6
    set_table_aspect(1.6)
    assert abs(get_aspect(None) - 1.6) < 1e-9
    rx, ry = radii_xy(0.02)
    assert almost(rx, 0.02 / 1.6)
    assert almost(ry, 0.02)

    # 恢复理论值，避免影响其它单测
    set_table_aspect(None)
    print("OK measured aspect override")


def main():
    test_reflect_point()
    test_one_cushion_angle()
    test_two_cushion_fold()
    test_direct_blocked()
    test_near_cue_blocker()
    test_shoot_away_from_touching_ball()
    test_same_color_other_red_blocks()
    test_pocket_risk_filters_middle()
    test_ghost_ball()
    test_object_bank()
    test_pot_middle_pocket_center()
    test_pot_angle_rejects_acute()
    test_cue_bank_pot_exists()
    test_path_corridor_anisotropic()
    test_snooker_score_prefers_easy_pot()
    test_multicushion_midpath_hits_target()
    test_plant_only_red_to_red()
    test_corner_aim_offset()
    test_middle_pocket_rail_angle()
    test_measured_aspect_override()
    # 最终段被挡
    from geometry import find_pot_paths as _fp

    cue, target, pk, blocker, r = (0.2, 0.7), (0.5, 0.4), (0.5, 0.0), (0.5, 0.2), 0.015
    plans = _fp(cue, [target], [pk], "red", [blocker], r, include_object_bank=False)
    kinds = {p.kind for p in plans}
    assert "direct" not in kinds, f"最终段被挡不应有 direct: {kinds}"
    print("OK final segment blocked", kinds)
    print("\n全部几何测试通过")


if __name__ == "__main__":
    main()
