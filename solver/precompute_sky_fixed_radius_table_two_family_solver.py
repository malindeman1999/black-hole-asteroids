from __future__ import annotations

import argparse
import sys
import time
import warnings
from math import pi
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole, cp
from precompute_earliest_grid import (
    _arrival_direction_at_b_for_pair,
    _build_a_points,
    _build_b_points,
    _build_path_profile,
    _dir_world_to_local,
    _direction_from_angle_at_a_for_pair,
    _local_gamma_at_radius,
    _radial_grid_clustered,
    _regular_phi_grid,
    _save_npz,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Precompute fixed-source-radius sky interpolation table using the new two-family continuation solver. "
            "Family 0 is tracked by continuation in A-phi; family 1 is constructed as X-mirror of family 0."
        )
    )
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="high")
    p.add_argument("--sky-radius-rs", type=float, default=100.0, help="Fixed A radius in Rs.")
    p.add_argument("--a-phi-count", type=int, default=97, help="A azimuth samples at fixed radius.")
    p.add_argument("--b-r-min-rs", type=float, default=1.6, help="Min B radius in Rs.")
    p.add_argument("--b-r-max-rs", type=float, default=10.0, help="Max B radius in Rs.")
    p.add_argument("--b-r-count", type=int, default=28, help="B radial samples.")
    p.add_argument(
        "--b-radial-exponent",
        type=float,
        default=2.8,
        help="Cluster exponent for fallback B-r grid.",
    )
    p.add_argument(
        "--match-b-r-from",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Optional reference precompute file; if present, use its b_r_m exactly.",
    )
    p.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU in candidate solve when available.",
    )
    p.add_argument(
        "--continuity-impact-weight",
        type=float,
        default=0.2,
        help="Impact-parameter weight in continuation cost.",
    )
    p.add_argument(
        "--continuity-source-weight",
        type=float,
        default=0.6,
        help="Source-angle continuity weight.",
    )
    p.add_argument(
        "--continuity-shape-weight",
        type=float,
        default=0.8,
        help="Path-shape continuity weight.",
    )
    p.add_argument(
        "--critical-branch-switch-penalty",
        type=float,
        default=2.0,
        help="Extra penalty for branch changes near critical impact parameter.",
    )
    p.add_argument(
        "--critical-refine-band",
        type=float,
        default=0.08,
        help="Relative |b-b_crit|/b_crit band for high-quality near-critical re-solve.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
    )
    p.add_argument(
        "--true-path-samples",
        type=int,
        default=7000,
        help="Number of samples per saved true path polyline (default: 7000).",
    )
    p.add_argument(
        "--debug-show-rings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live trajectory plot for each B ring as it is solved.",
    )
    p.add_argument(
        "--debug-pause-rings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When debug-show-rings is enabled, pause on each ring until key/mouse input.",
    )
    p.add_argument("--residual-max-phi-rad", type=float, default=1e-3)
    p.add_argument("--residual-max-time-rel", type=float, default=1e-3)
    p.add_argument("--residual-max-impact-rel", type=float, default=1e-3)
    p.add_argument("--null-check-max-ratio", type=float, default=0.2)
    p.add_argument("--null-check-strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--max-spatial-step-rs",
        type=float,
        default=0.01,
        help="Maximum allowed segment length along saved true paths, in Rs (default: 0.01).",
    )
    p.add_argument(
        "--spatial-step-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort save if any saved true path exceeds --max-spatial-step-rs.",
    )
    p.add_argument(
        "--solver-warning-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort save if numerical warnings are emitted during solver calls.",
    )
    return p.parse_args()


def _unit_xy(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / n


def _mirror_world_about_x(world_dir_xy: np.ndarray) -> np.ndarray:
    return _unit_xy(np.asarray([float(world_dir_xy[0]), -float(world_dir_xy[1])], dtype=float))


def _wrap_angle_diff(a: float, b: float) -> float:
    return abs((a - b + pi) % (2.0 * pi) - pi)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    uu = _unit_xy(u)
    vv = _unit_xy(v)
    return float(np.arccos(np.clip(float(np.dot(uu, vv)), -1.0, 1.0)))


def _short_azimuth(a_point: np.ndarray, b_point: np.ndarray) -> float:
    th_a = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_b = float(np.arctan2(float(b_point[1]), float(b_point[0])))
    return abs(((th_b - th_a + pi) % (2.0 * pi)) - pi)


def _radial_null_travel_time_s(rs_m: float, r0_m: float, r1_m: float) -> float:
    lo = max(min(float(r0_m), float(r1_m)), float(rs_m) * (1.0 + 1e-12))
    hi = max(float(r0_m), float(r1_m))
    return ((hi - lo) + float(rs_m) * float(np.log((hi - rs_m) / (lo - rs_m)))) / float(C)


def _angle_between_world(u: np.ndarray, v: np.ndarray) -> float:
    uu = _unit_xy(np.asarray(u[:2], dtype=float))
    vv = _unit_xy(np.asarray(v[:2], dtype=float))
    return float(np.arccos(np.clip(float(np.dot(uu, vv)), -1.0, 1.0)))


def _build_curve_xy_from_chosen_path(
    a_point: np.ndarray,
    b_point: np.ndarray,
    path,
    bh: SchwarzschildBlackHole,
    preferred_start_dir_world: np.ndarray,
    preferred_end_dir_world: np.ndarray,
    path_samples: int,
) -> np.ndarray:
    if abs(float(path.impact_parameter_m)) < 1e-14:
        x = np.linspace(float(a_point[0]), float(b_point[0]), max(2, int(path_samples)), dtype=float)
        y = np.linspace(float(a_point[1]), float(b_point[1]), max(2, int(path_samples)), dtype=float)
        return np.stack([x, y], axis=1)

    r_start = float(np.linalg.norm(a_point))
    r_end = float(np.linalg.norm(b_point))
    th_start = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_end = float(np.arctan2(float(b_point[1]), float(b_point[0])))
    dtheta_short = ((th_end - th_start + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dtheta_short >= 0.0 else -1.0

    r_samples, phi_samples = _build_path_profile(
        bh=bh,
        r_start=r_start,
        r_end=r_end,
        impact_b=float(path.impact_parameter_m),
        target_phi=float(path.target_azimuth_rad),
        branch=str(path.branch),
        n=max(120, int(path_samples)),
    )
    r_m = np.asarray(r_samples, dtype=float)

    def _curve_with_orient(orient_sign: float) -> np.ndarray:
        theta = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
        theta = np.unwrap(theta)
        x = r_m * np.cos(theta)
        y = r_m * np.sin(theta)
        return np.stack([x, y], axis=1)

    c1 = _curve_with_orient(short_sign)
    c2 = _curve_with_orient(-short_sign)

    def _cost(curve_xy: np.ndarray) -> float:
        if curve_xy.shape[0] < 3:
            return 0.0
        t0 = _unit_xy(curve_xy[1, :] - curve_xy[0, :])
        t1 = _unit_xy(curve_xy[-1, :] - curve_xy[-2, :])
        return _angle_between_world(t0, preferred_start_dir_world) + _angle_between_world(t1, preferred_end_dir_world)

    chosen = c1 if _cost(c1) <= _cost(c2) else c2
    return _resample_polyline_equal_arclen(chosen, int(path_samples))


def _resample_polyline_equal_arclen(xy: np.ndarray, n_samples: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    n_out = max(2, int(n_samples))
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        out = np.full((n_out, 2), np.nan, dtype=float)
        if xy.ndim == 2 and xy.shape[1] == 2 and xy.shape[0] > 0:
            out[:] = xy[min(0, xy.shape[0] - 1), :]
        return out

    seg = np.diff(xy, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    s = np.zeros(xy.shape[0], dtype=float)
    if ds.size > 0:
        s[1:] = np.cumsum(ds)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        x = np.linspace(float(xy[0, 0]), float(xy[-1, 0]), n_out, dtype=float)
        y = np.linspace(float(xy[0, 1]), float(xy[-1, 1]), n_out, dtype=float)
        return np.stack([x, y], axis=1)

    s_new = np.linspace(0.0, total, n_out, dtype=float)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.stack([x_new, y_new], axis=1)


def _shape_distance(curve_a: np.ndarray, curve_b: np.ndarray, scale: float) -> float:
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        return 0.0
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    n = min(a.shape[0], b.shape[0], 160)
    if a.shape[0] != n:
        a = _resample_polyline_equal_arclen(a, n)
    if b.shape[0] != n:
        b = _resample_polyline_equal_arclen(b, n)
    d = np.linalg.norm(a - b, axis=1)
    return float(np.mean(d) / max(float(scale), 1e-12))


def _retarget_curve_a_endpoint(
    curve_xy: np.ndarray,
    a_prev_point: np.ndarray,
    a_new_point: np.ndarray,
) -> np.ndarray:
    """
    Reuse previous A->B trajectory as a continuation seed, but update the A-end
    to the new source position while keeping the B-end stable.
    """
    curve = np.asarray(curve_xy, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2:
        return curve
    a_prev = np.asarray(a_prev_point[:2], dtype=float)
    a_new = np.asarray(a_new_point[:2], dtype=float)
    if not (np.all(np.isfinite(a_prev)) and np.all(np.isfinite(a_new))):
        return curve

    delta = a_new - a_prev
    if float(np.linalg.norm(delta)) <= 0.0:
        out = np.asarray(curve, dtype=float).copy()
        out[0, :] = a_new
        return out

    out = np.asarray(curve, dtype=float).copy()
    n = int(out.shape[0])
    n_blend = max(3, int(round(0.35 * n)))
    n_blend = min(n_blend, n)
    if n_blend <= 1:
        out[0, :] = a_new
        return out

    # Smoothly taper translation from A-end to preserve B-end anchoring.
    t = np.linspace(0.0, 1.0, n_blend, dtype=float)
    w = 0.5 * (1.0 + np.cos(np.pi * t))
    out[:n_blend, :] = out[:n_blend, :] + w[:, None] * delta[None, :]
    out[0, :] = a_new
    return out


def _plot_debug_ring(ax, rs: float, a_points: np.ndarray, b_point: np.ndarray, out: Dict[str, np.ndarray], b_i: int, n_b: int) -> None:
    ax.clear()
    t = np.linspace(0.0, 2.0 * pi, 220)
    ax.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.0)
    ax.plot(1.5 * rs * np.cos(t), 1.5 * rs * np.sin(t), "k--", lw=1.0, alpha=0.9)
    ax.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.45", alpha=0.4)
    ax.scatter([float(b_point[0])], [float(b_point[1])], c="green", marker="x", s=70, linewidths=1.8)

    path_plus = np.asarray(out["true_path_plus_xy_m"], dtype=float)
    path_minus = np.asarray(out["true_path_minus_xy_m"], dtype=float)
    for arr, color, lw, alpha in [
        (path_plus, "blue", 0.8, 0.5),
        (path_minus, "red", 0.8, 0.4),
    ]:
        for a_i in range(arr.shape[0]):
            xy = arr[a_i, :, :]
            if xy.ndim != 2 or xy.shape[1] != 2:
                continue
            mask = np.all(np.isfinite(xy), axis=1)
            if int(np.count_nonzero(mask)) < 2:
                continue
            ax.plot(xy[mask, 0], xy[mask, 1], color=color, lw=lw, alpha=alpha)

    r_lim = 1.12 * max(
        float(np.max(np.linalg.norm(a_points[:, :2], axis=1))) if a_points.size > 0 else 1.0,
        float(np.linalg.norm(np.asarray(b_point[:2], dtype=float))),
    )
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Debug trajectories: B ring {b_i + 1}/{n_b} | blue=plus, red=minus")


def _null_ratio_from_xy(rs_m: float, xy: np.ndarray, impact_parameter_m: float) -> float:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return float("inf")
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if pts.shape[0] < 3:
        return float("inf")
    b = abs(float(impact_parameter_m))
    r = np.linalg.norm(pts, axis=1)
    th = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    sum_ds2 = 0.0
    sum_dx2 = 0.0
    for i in range(pts.shape[0] - 1):
        r0 = float(r[i])
        r1 = float(r[i + 1])
        th0 = float(th[i])
        th1 = float(th[i + 1])
        dr = r1 - r0
        dphi = th1 - th0
        rm = max(0.5 * (r0 + r1), rs_m * (1.0 + 1e-12))
        g = max(1e-14, 1.0 - rs_m / rm)
        if b < 1e-14:
            # Radial null branch: dr/dt = c*(1-rs/r), so dt = |dr| / (c*(1-rs/r)).
            dt = abs(dr) / (C * g)
        else:
            dt = abs(dphi) * (rm * rm) / (C * g * b)
        spatial_dx2 = (dr * dr) / g + (rm * rm) * (dphi * dphi)
        ds2 = -g * ((C * dt) ** 2) + spatial_dx2
        sum_ds2 += ds2
        sum_dx2 += spatial_dx2
    return float(abs(sum_ds2) / max(sum_dx2, 1e-30))


def _max_segment_step_rs(rs_m: float, xy: np.ndarray) -> float:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return float("inf")
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if pts.shape[0] < 2:
        return float("inf")
    seg = np.diff(pts, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    if ds.size == 0:
        return 0.0
    return float(np.max(ds) / max(float(rs_m), 1e-30))


def _choose_b_r_axis(args: argparse.Namespace, rs: float) -> np.ndarray:
    if args.match_b_r_from is not None and Path(args.match_b_r_from).exists():
        try:
            d_ref = np.load(Path(args.match_b_r_from), allow_pickle=False)
            b_r_ref = np.asarray(d_ref["b_r_m"], dtype=float).reshape(-1)
            if b_r_ref.size >= 2 and np.all(np.isfinite(b_r_ref)):
                print(f"Using B-r sampling from reference table: {args.match_b_r_from}")
                return b_r_ref.copy()
        except Exception:
            pass

    if args.b_r_count < 2:
        raise ValueError("--b-r-count must be >= 2")
    if args.b_r_max_rs <= args.b_r_min_rs:
        raise ValueError("--b-r-max-rs must be > --b-r-min-rs")
    print(
        "Reference B-r sampling unavailable; using clustered fallback B-r grid "
        f"(rmin={args.b_r_min_rs}, rmax={args.b_r_max_rs}, n={args.b_r_count}, exp={args.b_radial_exponent})."
    )
    return _radial_grid_clustered(
        rs=rs,
        rmin_rs=float(args.b_r_min_rs),
        rmax_rs=float(args.b_r_max_rs),
        n=int(args.b_r_count),
        exponent=float(args.b_radial_exponent),
    )


def _solve_ring_two_families(
    bh: SchwarzschildBlackHole,
    bh_high: SchwarzschildBlackHole,
    rs: float,
    a_points: np.ndarray,
    b_point: np.ndarray,
    use_gpu: bool,
    continuity_impact_weight: float,
    continuity_source_weight: float,
    continuity_shape_weight: float,
    critical_branch_switch_penalty: float,
    critical_refine_band: float,
    true_path_samples: int,
) -> Dict[str, np.ndarray]:
    n_a = int(a_points.shape[0])
    b_r = float(np.linalg.norm(b_point))
    a_r = np.linalg.norm(a_points[:, :2], axis=1)

    ok_plus = np.zeros(n_a, dtype=bool)
    ok_minus = np.zeros(n_a, dtype=bool)
    dt_plus = np.full(n_a, np.nan, dtype=float)
    dt_minus = np.full(n_a, np.nan, dtype=float)
    gamma_b_plus = np.full(n_a, np.nan, dtype=float)
    gamma_b_minus = np.full(n_a, np.nan, dtype=float)
    gamma_a_plus = np.full(n_a, np.nan, dtype=float)
    gamma_a_minus = np.full(n_a, np.nan, dtype=float)
    dir_b_plus = np.full((n_a, 2), np.nan, dtype=float)
    dir_b_minus = np.full((n_a, 2), np.nan, dtype=float)
    dir_a_plus = np.full((n_a, 2), np.nan, dtype=float)
    dir_a_minus = np.full((n_a, 2), np.nan, dtype=float)
    impact_plus = np.full(n_a, np.nan, dtype=float)
    impact_minus = np.full(n_a, np.nan, dtype=float)
    target_plus = np.full(n_a, np.nan, dtype=float)
    target_minus = np.full(n_a, np.nan, dtype=float)
    direction_plus = np.zeros(n_a, dtype=np.int8)
    direction_minus = np.zeros(n_a, dtype=np.int8)
    branch_plus = np.full(n_a, "", dtype="<U16")
    branch_minus = np.full(n_a, "", dtype="<U16")
    true_path_plus_xy = np.full((n_a, int(true_path_samples), 2), np.nan, dtype=float)
    true_path_minus_xy = np.full((n_a, int(true_path_samples), 2), np.nan, dtype=float)
    had_solver_warning = np.zeros(n_a, dtype=bool)
    solver_warning_text = np.full(n_a, "", dtype="<U160")

    b_crit = float(1.5 * np.sqrt(3.0) * rs)
    preview_n = 140
    spatial_scale = max(float(np.linalg.norm(b_point[:2])), float(np.nanmax(np.linalg.norm(a_points[:, :2], axis=1))))
    prev_obs_ang: float | None = None
    prev_src_ang: float | None = None
    prev_imp: float | None = None
    prev_branch: str | None = None
    prev_preview_curve: np.ndarray | None = None
    prev_a_point: np.ndarray | None = None
    for a_i, a_point in enumerate(a_points):
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            rr = bh.find_all_geodesic_candidates(
                a_point,
                b_point,
                a_before_b=True,
                use_gpu=use_gpu,
                warm_start_impact_parameter_m=(float(prev_imp) if prev_imp is not None else None),
                warm_start_branch=(str(prev_branch) if prev_branch is not None else None),
            )
        if wrec:
            had_solver_warning[a_i] = True
            solver_warning_text[a_i] = str(wrec[0].message)[:160]
        # Near critical impact parameter, re-solve with high-quality CPU path for stability.
        refine_near_crit = False
        if rr is not None and rr.paths:
            for p in rr.paths:
                rel = abs(float(p.impact_parameter_m) - b_crit) / max(abs(b_crit), 1e-12)
                if rel < float(critical_refine_band):
                    refine_near_crit = True
                    break
        if prev_imp is not None:
            prev_rel = abs(float(prev_imp) - b_crit) / max(abs(b_crit), 1e-12)
            refine_near_crit = refine_near_crit or (prev_rel < float(critical_refine_band))
        if refine_near_crit:
            with warnings.catch_warnings(record=True) as wrec:
                warnings.simplefilter("always")
                rr_hi = bh_high.find_all_geodesic_candidates(
                    a_point,
                    b_point,
                    a_before_b=True,
                    use_gpu=False,
                    warm_start_impact_parameter_m=(float(prev_imp) if prev_imp is not None else None),
                    warm_start_branch=(str(prev_branch) if prev_branch is not None else None),
                )
            if wrec:
                had_solver_warning[a_i] = True
                solver_warning_text[a_i] = str(wrec[0].message)[:160]
            if rr_hi is not None and rr_hi.paths:
                rr = rr_hi
        cand_rows = []
        gamma_short = _short_azimuth(a_point, b_point)
        to_b_world = _unit_xy(np.asarray(b_point[:2], dtype=float) - np.asarray(a_point[:2], dtype=float))
        from_a_world = _unit_xy(np.asarray(a_point[:2], dtype=float) - np.asarray(b_point[:2], dtype=float))
        for path in rr.paths:
            gb = _local_gamma_at_radius(rs, b_r, float(path.impact_parameter_m))
            ga = _local_gamma_at_radius(rs, float(a_r[a_i]), float(path.impact_parameter_m))
            d_b_prop_world = _arrival_direction_at_b_for_pair(
                a_point=a_point,
                b_point=b_point,
                gamma_at_b=gb,
                branch_side=int(path.direction),
                branch_name=str(path.branch),
            )
            d_a_prop_world = _direction_from_angle_at_a_for_pair(
                a_point=a_point,
                b_point=b_point,
                gamma_at_a=ga,
                branch_side=int(path.direction),
            )
            # Use "coming-from" only for continuation matching; table output remains propagation direction.
            d_b_from_world = -_unit_xy(d_b_prop_world)
            obs_ang = float(np.arctan2(float(d_b_from_world[1]), float(d_b_from_world[0])))
            src_ang = float(np.arctan2(float(d_a_prop_world[1]), float(d_a_prop_world[0])))
            direct_cost = _angle_between(d_a_prop_world, to_b_world) + _angle_between(d_b_from_world, from_a_world)
            preview_curve = None
            try:
                preview_curve = _build_curve_xy_from_chosen_path(
                    a_point=np.asarray(a_point, dtype=float),
                    b_point=np.asarray(b_point, dtype=float),
                    path=path,
                    bh=bh_high,
                    preferred_start_dir_world=d_a_prop_world,
                    preferred_end_dir_world=d_b_prop_world,
                    path_samples=preview_n,
                )
            except Exception:
                preview_curve = None
            cand_rows.append(
                {
                    "path": path,
                    "gamma_b": gb,
                    "gamma_a": ga,
                    "obs_ang": obs_ang,
                    "src_ang": src_ang,
                    "direct_cost": direct_cost,
                    "d_b_prop_world": _unit_xy(d_b_prop_world),
                    "d_a_prop_world": _unit_xy(d_a_prop_world),
                    "preview_curve": preview_curve,
                }
            )
        # Axis-degenerate first sample: inject exact direct radial candidate.
        # This guarantees the first family seed is the straight A->B trajectory.
        if prev_obs_ang is None and gamma_short < 1e-12:
            d_prop_world = to_b_world
            d_from_world = -d_prop_world

            class _DirectPath:
                direction = +1
                branch = "monotonic"
                target_azimuth_rad = 0.0
                impact_parameter_m = 0.0
                travel_time_s = 0.0

            direct_path = _DirectPath()
            direct_path.travel_time_s = _radial_null_travel_time_s(rs, float(a_r[a_i]), b_r)
            cand_rows.append(
                {
                    "path": direct_path,
                    "gamma_b": 0.0,
                    "gamma_a": 0.0,
                    "obs_ang": float(np.arctan2(float(d_from_world[1]), float(d_from_world[0]))),
                    "src_ang": float(np.arctan2(float(d_prop_world[1]), float(d_prop_world[0]))),
                    "direct_cost": 0.0,
                    "d_b_prop_world": d_prop_world,
                    "d_a_prop_world": d_prop_world,
                    "preview_curve": _resample_polyline_equal_arclen(
                        np.stack(
                            [
                                np.linspace(float(a_point[0]), float(b_point[0]), preview_n, dtype=float),
                                np.linspace(float(a_point[1]), float(b_point[1]), preview_n, dtype=float),
                            ],
                            axis=1,
                        ),
                        preview_n,
                    ),
                }
            )

        if not cand_rows:
            raise RuntimeError(f"No candidates found at A index {a_i}")

        if prev_obs_ang is None:
            # Seed family-0 from the straightest branch:
            # prioritize minimal azimuth sweep (near geometric short angle), then monotonic, then endpoint directness.
            chosen = min(
                cand_rows,
                key=lambda r: (
                    abs(float(r["path"].target_azimuth_rad) - float(gamma_short)),
                    0 if str(r["path"].branch) == "monotonic" else 1,
                    float(r["direct_cost"]),
                    float(r["path"].travel_time_s),
                ),
            )
        else:
            def _cost(r: Dict[str, object]) -> float:
                c_ang = _wrap_angle_diff(float(r["obs_ang"]), float(prev_obs_ang))
                c_src = _wrap_angle_diff(float(r["src_ang"]), float(prev_src_ang)) if prev_src_ang is not None else 0.0
                c_imp = (
                    abs(float(r["path"].impact_parameter_m) - float(prev_imp)) / max(abs(float(prev_imp)), 1.0)
                    if prev_imp is not None
                    else 0.0
                )
                c_shape = 0.0
                prev_curve_for_match = None
                if prev_preview_curve is not None and prev_a_point is not None:
                    prev_curve_for_match = _retarget_curve_a_endpoint(
                        curve_xy=prev_preview_curve,
                        a_prev_point=prev_a_point,
                        a_new_point=np.asarray(a_point, dtype=float),
                    )
                if prev_curve_for_match is not None and r.get("preview_curve", None) is not None:
                    c_shape = _shape_distance(np.asarray(r["preview_curve"], dtype=float), prev_curve_for_match, spatial_scale)
                c = (
                    c_ang
                    + float(continuity_source_weight) * c_src
                    + float(continuity_impact_weight) * c_imp
                    + float(continuity_shape_weight) * c_shape
                )
                if prev_branch is not None and str(r["path"].branch) != str(prev_branch):
                    rel = abs(float(r["path"].impact_parameter_m) - b_crit) / max(abs(b_crit), 1e-12)
                    if rel < float(critical_refine_band):
                        c += float(critical_branch_switch_penalty)
                return c

            chosen = min(cand_rows, key=_cost)

        path = chosen["path"]
        d_b0_world = np.asarray(chosen["d_b_prop_world"], dtype=float)
        d_a0_world = np.asarray(chosen["d_a_prop_world"], dtype=float)

        d_b1_world = _mirror_world_about_x(d_b0_world)
        d_a1_world = _mirror_world_about_x(d_a0_world)

        ok_plus[a_i] = True
        ok_minus[a_i] = True
        dt_plus[a_i] = float(path.travel_time_s)
        dt_minus[a_i] = float(path.travel_time_s)
        gamma_b_plus[a_i] = float(chosen["gamma_b"])
        gamma_b_minus[a_i] = float(chosen["gamma_b"])
        gamma_a_plus[a_i] = float(chosen["gamma_a"])
        gamma_a_minus[a_i] = float(chosen["gamma_a"])
        dir_b_plus[a_i, :] = _dir_world_to_local(b_point, d_b0_world)
        dir_b_minus[a_i, :] = _dir_world_to_local(b_point, d_b1_world)
        dir_a_plus[a_i, :] = _dir_world_to_local(a_point, d_a0_world)
        dir_a_minus[a_i, :] = _dir_world_to_local(a_point, d_a1_world)
        impact_plus[a_i] = float(path.impact_parameter_m)
        impact_minus[a_i] = float(path.impact_parameter_m)
        target_plus[a_i] = float(path.target_azimuth_rad)
        target_minus[a_i] = float(path.target_azimuth_rad)
        direction_plus[a_i] = int(path.direction)
        direction_minus[a_i] = int(-int(path.direction))
        branch_plus[a_i] = str(path.branch)
        branch_minus[a_i] = str(path.branch)
        try:
            path_xy_plus = _build_curve_xy_from_chosen_path(
                a_point=np.asarray(a_point, dtype=float),
                b_point=np.asarray(b_point, dtype=float),
                path=path,
                bh=bh,
                preferred_start_dir_world=d_a0_world,
                preferred_end_dir_world=d_b0_world,
                path_samples=int(true_path_samples),
            )
            true_path_plus_xy[a_i, :, :] = path_xy_plus
            # Family 1 is X-mirror by construction.
            true_path_minus_xy[a_i, :, 0] = path_xy_plus[:, 0]
            true_path_minus_xy[a_i, :, 1] = -path_xy_plus[:, 1]
        except Exception:
            pass

        prev_obs_ang = float(chosen["obs_ang"])
        prev_src_ang = float(chosen["src_ang"])
        prev_imp = float(path.impact_parameter_m)
        prev_branch = str(path.branch)
        prev_preview_curve = np.asarray(chosen["preview_curve"], dtype=float) if chosen.get("preview_curve", None) is not None else None
        prev_a_point = np.asarray(a_point, dtype=float)

    return {
        "ok_plus": ok_plus,
        "ok_minus": ok_minus,
        "delta_t_plus_s": dt_plus,
        "delta_t_minus_s": dt_minus,
        "gamma_at_b_plus_rad": gamma_b_plus,
        "gamma_at_b_minus_rad": gamma_b_minus,
        "gamma_at_a_plus_rad": gamma_a_plus,
        "gamma_at_a_minus_rad": gamma_a_minus,
        "dir_at_b_plus_local_xy": dir_b_plus,
        "dir_at_b_minus_local_xy": dir_b_minus,
        "dir_at_a_plus_local_xy": dir_a_plus,
        "dir_at_a_minus_local_xy": dir_a_minus,
        "impact_parameter_plus_m": impact_plus,
        "impact_parameter_minus_m": impact_minus,
        "target_azimuth_plus_rad": target_plus,
        "target_azimuth_minus_rad": target_minus,
        "direction_plus": direction_plus,
        "direction_minus": direction_minus,
        "branch_plus": branch_plus,
        "branch_minus": branch_minus,
        "had_solver_warning": had_solver_warning,
        "solver_warning_text": solver_warning_text,
        "true_path_plus_xy_m": true_path_plus_xy,
        "true_path_minus_xy_m": true_path_minus_xy,
    }


def main() -> None:
    args = _parse_args()
    if args.sky_radius_rs <= 1.0:
        raise ValueError("--sky-radius-rs must be > 1.0")
    if int(args.a_phi_count) < 2:
        raise ValueError("--a-phi-count must be >= 2")

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    bh_high = bh.with_quality("high")
    rs = float(bh.schwarzschild_radius_m)
    use_gpu_active = bool(args.use_gpu and cp is not None)

    a_r = np.asarray([float(args.sky_radius_rs) * rs], dtype=float)
    # Start source ring at +X (0 deg), then sweep counter-clockwise.
    a_phi = np.linspace(0.0, 2.0 * pi, int(args.a_phi_count), endpoint=False, dtype=float)
    b_r = _choose_b_r_axis(args=args, rs=rs)
    b_phi = np.asarray([0.0], dtype=float)

    a_points = _build_a_points(a_r, a_phi)
    b_points = _build_b_points(b_r, b_phi, b_on_x_axis=True)
    n_a = int(a_points.shape[0])
    n_b = int(b_points.shape[0])

    print(
        f"Mode: {'GPU' if use_gpu_active else 'CPU'} | quality={args.quality} | "
        f"A_phi={n_a} | B_r={n_b} | sky_radius={args.sky_radius_rs:.3f} Rs"
    )

    debug_fig = None
    debug_ax = None
    debug_plt = None
    if bool(args.debug_show_rings):
        try:
            import matplotlib.pyplot as plt  # local import so non-debug mode has no mpl dependency
            debug_plt = plt
            if bool(args.debug_pause_rings):
                plt.ioff()
            else:
                plt.ion()
            debug_fig, debug_ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
            debug_fig.canvas.manager.set_window_title("Sky Saver Debug Rings")
            plt.show(block=False)
        except Exception as exc:
            print(f"Debug plotting disabled (matplotlib unavailable): {exc}")
            debug_fig, debug_ax, debug_plt = None, None, None

    dt_plus = np.full((n_b, n_a), np.nan, dtype=float)
    dt_minus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_plus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_minus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_a_plus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_a_minus = np.full((n_b, n_a), np.nan, dtype=float)
    dir_b_plus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    dir_b_minus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    dir_a_plus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    dir_a_minus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    impact_plus = np.full((n_b, n_a), np.nan, dtype=float)
    impact_minus = np.full((n_b, n_a), np.nan, dtype=float)
    target_plus = np.full((n_b, n_a), np.nan, dtype=float)
    direction_plus = np.zeros((n_b, n_a), dtype=np.int8)
    branch_plus = np.full((n_b, n_a), "", dtype="<U16")
    had_solver_warning = np.zeros((n_b, n_a), dtype=bool)
    solver_warning_text = np.full((n_b, n_a), "", dtype="<U160")
    ok_plus = np.zeros((n_b, n_a), dtype=bool)
    ok_minus = np.zeros((n_b, n_a), dtype=bool)
    true_path_plus = np.full((n_b, n_a, int(args.true_path_samples), 2), np.nan, dtype=float)
    true_path_minus = np.full((n_b, n_a, int(args.true_path_samples), 2), np.nan, dtype=float)

    t0 = time.perf_counter()
    for b_i, b_point in enumerate(b_points):
        print(f"Solving B ring {b_i + 1}/{n_b}...", flush=True)
        out = _solve_ring_two_families(
            bh=bh,
            bh_high=bh_high,
            rs=rs,
            a_points=a_points,
            b_point=np.asarray(b_point, dtype=float),
            use_gpu=use_gpu_active,
            continuity_impact_weight=float(args.continuity_impact_weight),
            continuity_source_weight=float(args.continuity_source_weight),
            continuity_shape_weight=float(args.continuity_shape_weight),
            critical_branch_switch_penalty=float(args.critical_branch_switch_penalty),
            critical_refine_band=float(args.critical_refine_band),
            true_path_samples=int(args.true_path_samples),
        )
        ok_plus[b_i, :] = out["ok_plus"]
        ok_minus[b_i, :] = out["ok_minus"]
        dt_plus[b_i, :] = out["delta_t_plus_s"]
        dt_minus[b_i, :] = out["delta_t_minus_s"]
        gamma_plus[b_i, :] = out["gamma_at_b_plus_rad"]
        gamma_minus[b_i, :] = out["gamma_at_b_minus_rad"]
        gamma_a_plus[b_i, :] = out["gamma_at_a_plus_rad"]
        gamma_a_minus[b_i, :] = out["gamma_at_a_minus_rad"]
        dir_b_plus[b_i, :, :] = out["dir_at_b_plus_local_xy"]
        dir_b_minus[b_i, :, :] = out["dir_at_b_minus_local_xy"]
        dir_a_plus[b_i, :, :] = out["dir_at_a_plus_local_xy"]
        dir_a_minus[b_i, :, :] = out["dir_at_a_minus_local_xy"]
        impact_plus[b_i, :] = out["impact_parameter_plus_m"]
        impact_minus[b_i, :] = out["impact_parameter_minus_m"]
        target_plus[b_i, :] = out["target_azimuth_plus_rad"]
        direction_plus[b_i, :] = out["direction_plus"]
        branch_plus[b_i, :] = out["branch_plus"]
        had_solver_warning[b_i, :] = out["had_solver_warning"]
        solver_warning_text[b_i, :] = out["solver_warning_text"]
        true_path_plus[b_i, :, :, :] = out["true_path_plus_xy_m"]
        true_path_minus[b_i, :, :, :] = out["true_path_minus_xy_m"]

        if debug_ax is not None:
            _plot_debug_ring(
                ax=debug_ax,
                rs=rs,
                a_points=a_points,
                b_point=np.asarray(b_point, dtype=float),
                out=out,
                b_i=b_i,
                n_b=n_b,
            )
            debug_fig.canvas.draw_idle()
            try:
                debug_plt.pause(0.001)
                if bool(args.debug_pause_rings):
                    print("Debug pause: close the plot window to continue...", flush=True)
                    debug_plt.show(block=True)
                    if b_i < (n_b - 1):
                        debug_fig, debug_ax = debug_plt.subplots(1, 1, figsize=(7.5, 7.5))
                        debug_fig.canvas.manager.set_window_title("Sky Saver Debug Rings")
            except Exception:
                pass

    # Validate saved +family solutions.
    residual_fail_rows: list[tuple[int, int, float, float, float, str]] = []
    null_fail_rows: list[tuple[int, int, float]] = []
    step_fail_rows: list[tuple[int, int, str, float]] = []
    warning_rows: list[tuple[int, int, str, str]] = []
    for b_i in range(n_b):
        for a_i in range(n_a):
            if bool(had_solver_warning[b_i, a_i]):
                warning_rows.append((b_i, a_i, "find_all_geodesic_candidates", str(solver_warning_text[b_i, a_i])))
    for b_i, b_point in enumerate(b_points):
        r_b = float(np.linalg.norm(b_point))
        for a_i, a_point in enumerate(a_points):
            if not bool(ok_plus[b_i, a_i]):
                continue
            r_a = float(np.linalg.norm(a_point))
            b_imp = float(impact_plus[b_i, a_i])
            t_saved = float(dt_plus[b_i, a_i])
            target_phi = float(target_plus[b_i, a_i])
            branch = str(branch_plus[b_i, a_i])
            try:
                if abs(b_imp) < 1e-14 and abs(target_phi) < 1e-10 and branch == "monotonic":
                    # Synthetic direct radial seed; treat as exact for residual checks.
                    b_res_rel = 0.0
                    t_res_rel = 0.0
                    phi_res = 0.0
                else:
                    with warnings.catch_warnings(record=True) as wrec:
                        warnings.simplefilter("always")
                        sols = bh._solve_for_target_azimuth(r_a, r_b, target_phi, use_gpu=False)
                    if wrec:
                        warning_rows.append((b_i, a_i, "_solve_for_target_azimuth", str(wrec[0].message)[:160]))
                    branch_sols = [s for s in sols if str(s[2]) == branch]
                    if not branch_sols:
                        raise RuntimeError("No branch-matched solution")
                    b_calc, t_calc, _ = min(branch_sols, key=lambda s: abs(float(s[0]) - b_imp))
                    b_calc = float(b_calc)
                    t_calc = float(t_calc)
                    b_res_rel = abs(b_calc - b_imp) / max(abs(b_imp), 1e-12)
                    t_res_rel = abs(t_calc - t_saved) / max(abs(t_saved), 1e-12)
                    if branch == "monotonic":
                        with warnings.catch_warnings(record=True) as wrec:
                            warnings.simplefilter("always")
                            phi_calc = float(bh._delta_phi_mono(r_a, r_b, b_calc, use_gpu=False))
                        if wrec:
                            warning_rows.append((b_i, a_i, "_delta_phi_mono", str(wrec[0].message)[:160]))
                        phi_res = float(phi_calc - target_phi)
                    else:
                        phi_res = 0.0
                if (
                    abs(phi_res) > float(args.residual_max_phi_rad)
                    or t_res_rel > float(args.residual_max_time_rel)
                    or b_res_rel > float(args.residual_max_impact_rel)
                ):
                    residual_fail_rows.append((b_i, a_i, phi_res, t_res_rel, b_res_rel, branch))
            except Exception:
                residual_fail_rows.append((b_i, a_i, float("inf"), float("inf"), float("inf"), branch))

            ratio = _null_ratio_from_xy(rs_m=rs, xy=true_path_plus[b_i, a_i, :, :], impact_parameter_m=b_imp)
            if ratio > float(args.null_check_max_ratio):
                null_fail_rows.append((b_i, a_i, ratio))

            plus_step_rs = _max_segment_step_rs(rs_m=rs, xy=true_path_plus[b_i, a_i, :, :])
            if plus_step_rs > float(args.max_spatial_step_rs):
                step_fail_rows.append((b_i, a_i, "plus", plus_step_rs))
            minus_step_rs = _max_segment_step_rs(rs_m=rs, xy=true_path_minus[b_i, a_i, :, :])
            if minus_step_rs > float(args.max_spatial_step_rs):
                step_fail_rows.append((b_i, a_i, "minus", minus_step_rs))

    total_sol = int(np.count_nonzero(ok_plus))
    if warning_rows:
        print(f"Solver warning check: {len(warning_rows)} warnings captured.", flush=True)
        for b_i, a_i, phase, msg in warning_rows[:12]:
            print(f"  B={b_i:3d} A={a_i:3d} phase={phase:<26s} msg={msg}", flush=True)
        if bool(args.solver_warning_strict):
            raise RuntimeError("Solver warning check failed; aborting save.")
        print("Continuing despite solver warnings (warning-only mode).", flush=True)
    else:
        print("Solver warning check passed: no warnings captured.", flush=True)

    if residual_fail_rows:
        print(
            f"Residual check FAILED: {len(residual_fail_rows)}/{total_sol} candidates exceed tolerances.",
            flush=True,
        )
        for b_i, a_i, phi_res, t_res_rel, b_res_rel, branch in residual_fail_rows[:12]:
            print(
                f"  B={b_i:3d} A={a_i:3d} branch={branch:9s} phi_res={phi_res:.3e} "
                f"rel_t={t_res_rel:.3e} rel_b={b_res_rel:.3e}",
                flush=True,
            )
        raise RuntimeError("Residual-based solver check failed; aborting save.")
    print(
        f"Residual check passed for all {total_sol} candidates "
        f"(phi_tol={float(args.residual_max_phi_rad):.3e}, rel_t_tol={float(args.residual_max_time_rel):.3e}, "
        f"rel_b_tol={float(args.residual_max_impact_rel):.3e}).",
        flush=True,
    )

    if null_fail_rows:
        print(
            f"Null-interval check FAILED: {len(null_fail_rows)}/{total_sol} candidates exceed "
            f"|sum(ds^2)|/sum(dx^2) > {float(args.null_check_max_ratio):.3e}",
            flush=True,
        )
        for b_i, a_i, ratio in null_fail_rows[:12]:
            print(f"  B={b_i:3d} A={a_i:3d} ratio={ratio:.3e}", flush=True)
        if bool(args.null_check_strict):
            raise RuntimeError("Null-interval check failed; aborting save.")
        print("Continuing despite null-interval failures (warning-only mode).", flush=True)
    else:
        print(
            f"Null-interval check passed for all {total_sol} candidates "
            f"(tol={float(args.null_check_max_ratio):.3e}).",
            flush=True,
        )

    if step_fail_rows:
        print(
            f"Spatial-step check FAILED: {len(step_fail_rows)} paths exceed max segment "
            f"{float(args.max_spatial_step_rs):.3e} Rs",
            flush=True,
        )
        for b_i, a_i, fam, step_rs in step_fail_rows[:12]:
            print(f"  B={b_i:3d} A={a_i:3d} fam={fam:5s} max_step={step_rs:.3e} Rs", flush=True)
        if bool(args.spatial_step_strict):
            raise RuntimeError("Spatial-step check failed; aborting save.")
        print("Continuing despite spatial-step failures (warning-only mode).", flush=True)
    else:
        print(
            f"Spatial-step check passed for all saved paths "
            f"(max_step_tol={float(args.max_spatial_step_rs):.3e} Rs).",
            flush=True,
        )

    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {
        "rs_m": np.asarray(rs, dtype=float),
        "a_r_m": np.asarray(a_r, dtype=float),
        "a_phi_rad": np.asarray(a_phi, dtype=float),
        "b_r_m": np.asarray(b_r, dtype=float),
        "b_phi_rad": np.asarray(b_phi, dtype=float),
        "a_points_m": np.asarray(a_points, dtype=float),
        "b_points_m": np.asarray(b_points, dtype=float),
        "delta_t_plus_s": dt_plus,
        "delta_t_minus_s": dt_minus,
        "gamma_at_b_plus_rad": gamma_plus,
        "gamma_at_b_minus_rad": gamma_minus,
        "gamma_at_a_plus_rad": gamma_a_plus,
        "gamma_at_a_minus_rad": gamma_a_minus,
        "dir_at_b_plus_local_xy": dir_b_plus,
        "dir_at_b_minus_local_xy": dir_b_minus,
        "dir_at_a_plus_local_xy": dir_a_plus,
        "dir_at_a_minus_local_xy": dir_a_minus,
        "impact_parameter_plus_m": impact_plus,
        "impact_parameter_minus_m": impact_minus,
        "target_azimuth_plus_rad": target_plus,
        "direction_plus": direction_plus,
        "branch_plus": branch_plus,
        "had_solver_warning": had_solver_warning,
        "solver_warning_text": solver_warning_text,
        "ok_plus": ok_plus,
        "ok_minus": ok_minus,
        "true_path_plus_xy_m": true_path_plus,
        "true_path_minus_xy_m": true_path_minus,
    }
    metadata: Dict[str, object] = {
        "description": "Fixed-source-radius sky interpolation table (two-family continuation solver).",
        "solver": "two_family_continuation_with_x_mirror",
        "quality": args.quality,
        "use_gpu_requested": bool(args.use_gpu),
        "gpu_available": bool(cp is not None),
        "use_gpu_active": bool(use_gpu_active),
        "sky_radius_rs": float(args.sky_radius_rs),
        "a_phi_count": int(n_a),
        "b_r_count": int(n_b),
        "b_on_x_axis": True,
        "continuity_impact_weight": float(args.continuity_impact_weight),
        "continuity_source_weight": float(args.continuity_source_weight),
        "continuity_shape_weight": float(args.continuity_shape_weight),
        "critical_branch_switch_penalty": float(args.critical_branch_switch_penalty),
        "critical_refine_band": float(args.critical_refine_band),
        "observer_dir_convention": "propagation",
        "true_path_samples": int(args.true_path_samples),
        "max_spatial_step_rs": float(args.max_spatial_step_rs),
        "solve_elapsed_s": float(elapsed),
    }
    _save_npz(args.output, arrays=arrays, metadata=metadata)
    print(f"Saved sky interpolation table: {args.output}", flush=True)
    if debug_fig is not None and debug_plt is not None:
        try:
            debug_plt.ioff()
            debug_plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
