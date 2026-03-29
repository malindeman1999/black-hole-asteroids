from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import SchwarzschildBlackHole, cp
from precompute_earliest_grid import (
    _arrival_direction_at_b_for_pair,
    _dir_world_to_local,
    _direction_from_angle_at_a_for_pair,
    _local_gamma_at_radius,
    _regular_phi_grid,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build 2-family sky candidate table by sweeping source azimuth A from 0..360 deg with fixed observer B on +X. "
            "Family 0 is tracked by continuation; Family 1 is the X-mirror of Family 0."
        )
    )
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="high")
    p.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sky-radius-rs", type=float, default=100.0)
    p.add_argument("--b-radius-rs", type=float, default=10.0)
    p.add_argument("--a-phi-count", type=int, default=20, help="Source azimuth samples over [0, 360).")
    p.add_argument("--null-check-segments", type=int, default=400)
    p.add_argument("--null-check-max-ratio", type=float, default=0.3)
    p.add_argument("--null-check-strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--residual-max-phi-rad", type=float, default=1e-3)
    p.add_argument("--residual-max-time-rel", type=float, default=1e-3)
    p.add_argument("--residual-max-impact-rel", type=float, default=1e-3)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "sky_candidates_b_sweep_two_families.npz",
    )
    return p.parse_args()


def _unit_xy(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / n


def _local_to_world(point_xy: np.ndarray, local_dir_xy: np.ndarray) -> np.ndarray:
    er = _unit_xy(np.asarray(point_xy[:2], dtype=float))
    ephi = np.asarray([-er[1], er[0]], dtype=float)
    return _unit_xy(float(local_dir_xy[0]) * er + float(local_dir_xy[1]) * ephi)


def _mirror_world_about_x(world_dir_xy: np.ndarray) -> np.ndarray:
    return _unit_xy(np.asarray([float(world_dir_xy[0]), -float(world_dir_xy[1])], dtype=float))


def _wrap_angle_diff(a: float, b: float) -> float:
    d = (a - b + pi) % (2.0 * pi) - pi
    return abs(d)


def _build_curve_profile(
    bh: SchwarzschildBlackHole,
    a_point: np.ndarray,
    b_point: np.ndarray,
    impact_parameter_m: float,
    target_azimuth_rad: float,
    is_turning: bool,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    from precompute_earliest_grid import _build_path_profile

    r_start = float(np.linalg.norm(a_point))
    r_end = float(np.linalg.norm(b_point))
    th_start = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_end = float(np.arctan2(float(b_point[1]), float(b_point[0])))
    dtheta_short = ((th_end - th_start + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dtheta_short >= 0.0 else -1.0
    gamma_short = abs(dtheta_short)
    gamma_long = 2.0 * pi - gamma_short

    r_samples, phi_samples = _build_path_profile(
        bh=bh,
        r_start=r_start,
        r_end=r_end,
        impact_b=float(impact_parameter_m),
        target_phi=float(target_azimuth_rad),
        branch=("turning" if bool(is_turning) else "monotonic"),
        n=max(1200, int(n)),
    )
    is_short = abs(float(target_azimuth_rad) - gamma_short) <= abs(float(target_azimuth_rad) - gamma_long)
    orient_sign = short_sign if is_short else -short_sign
    theta = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
    theta = np.unwrap(theta)
    r_m = np.asarray(r_samples, dtype=float)
    return r_m, theta


def _null_ratio(
    rs_m: float,
    r_m: np.ndarray,
    theta: np.ndarray,
    impact_parameter_m: float,
    segments: int,
) -> float:
    n = max(2, int(segments) + 1)
    s_old = np.linspace(0.0, 1.0, int(r_m.size), dtype=float)
    s_new = np.linspace(0.0, 1.0, n, dtype=float)
    r_u = np.interp(s_new, s_old, r_m)
    th_u = np.interp(s_new, s_old, theta)
    th_u = np.unwrap(th_u)
    b = max(1e-20, abs(float(impact_parameter_m)))
    sum_ds2 = 0.0
    sum_dx2 = 0.0
    for i in range(n - 1):
        r0 = float(r_u[i])
        r1 = float(r_u[i + 1])
        th0 = float(th_u[i])
        th1 = float(th_u[i + 1])
        dr = r1 - r0
        dphi = th1 - th0
        rm = max(0.5 * (r0 + r1), rs_m * (1.0 + 1e-12))
        g = max(1e-14, 1.0 - rs_m / rm)
        dt = abs(dphi) * (rm * rm) / (C * g * b)
        spatial_dx2 = (dr * dr) / g + (rm * rm) * (dphi * dphi)
        ds2 = -g * ((C * dt) ** 2) + spatial_dx2
        sum_ds2 += ds2
        sum_dx2 += spatial_dx2
    return float(abs(sum_ds2) / max(sum_dx2, 1e-30))


def main() -> None:
    args = _parse_args()
    if int(args.a_phi_count) < 3:
        raise ValueError("--a-phi-count must be >= 3")

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = float(bh.schwarzschild_radius_m)
    use_gpu_active = bool(args.use_gpu and cp is not None)

    a_r = float(args.sky_radius_rs) * rs
    b_r = float(args.b_radius_rs) * rs
    # Start A sweep at +X (0 deg), then advance counter-clockwise.
    a_phi = np.linspace(0.0, 2.0 * pi, int(args.a_phi_count), endpoint=False, dtype=float)
    a_points = np.zeros((a_phi.size, 3), dtype=float)
    a_points[:, 0] = a_r * np.cos(a_phi)
    a_points[:, 1] = a_r * np.sin(a_phi)
    b_point = np.asarray([b_r, 0.0, 0.0], dtype=float)

    n_a = a_phi.size
    n_fam = 2
    cand_ok = np.ones((n_a, n_fam), dtype=bool)
    cand_direction = np.zeros((n_a, n_fam), dtype=np.int8)
    cand_is_turning = np.zeros((n_a, n_fam), dtype=bool)
    cand_target_azimuth_rad = np.full((n_a, n_fam), np.nan, dtype=float)
    cand_travel_time_s = np.full((n_a, n_fam), np.nan, dtype=float)
    cand_impact_parameter_m = np.full((n_a, n_fam), np.nan, dtype=float)
    cand_gamma_at_b_rad = np.full((n_a, n_fam), np.nan, dtype=float)
    cand_gamma_at_a_rad = np.full((n_a, n_fam), np.nan, dtype=float)
    cand_dir_at_b_local_xy = np.full((n_a, n_fam, 2), np.nan, dtype=float)
    cand_dir_at_a_local_xy = np.full((n_a, n_fam, 2), np.nan, dtype=float)

    prev_obs_ang: float | None = None
    prev_imp: float | None = None
    for a_i, a_point in enumerate(a_points, start=0):
        print(f"Solving A sweep {a_i + 1}/{n_a}...", flush=True)
        rr = bh.find_all_geodesic_candidates(a_point, b_point, a_before_b=True, use_gpu=use_gpu_active)
        # Build candidate diagnostics for continuation selection.
        cand_rows: list[dict] = []
        for path in rr.paths:
            gamma_b = _local_gamma_at_radius(rs, b_r, float(path.impact_parameter_m))
            gamma_a = _local_gamma_at_radius(rs, a_r, float(path.impact_parameter_m))
            d_b_prop = _arrival_direction_at_b_for_pair(
                a_point, b_point, gamma_b, int(path.direction), str(path.branch)
            )
            d_b_from = -_unit_xy(d_b_prop)
            d_a = _direction_from_angle_at_a_for_pair(a_point, b_point, gamma_a, int(path.direction))
            obs_ang = float(np.arctan2(float(d_b_from[1]), float(d_b_from[0])))
            cand_rows.append(
                {
                    "path": path,
                    "gamma_b": gamma_b,
                    "gamma_a": gamma_a,
                    "d_b_from_world": d_b_from,
                    "d_a_world": d_a,
                    "obs_ang": obs_ang,
                }
            )
        if not cand_rows:
            raise RuntimeError(f"No candidates found at B index {b_i}.")

        if prev_obs_ang is None:
            chosen = min(cand_rows, key=lambda r: float(r["path"].travel_time_s))
        else:
            def _cost(r: dict) -> float:
                c_ang = _wrap_angle_diff(float(r["obs_ang"]), float(prev_obs_ang))
                c_imp = (
                    abs(float(r["path"].impact_parameter_m) - float(prev_imp)) / max(abs(float(prev_imp)), 1.0)
                    if prev_imp is not None
                    else 0.0
                )
                return c_ang + 0.2 * c_imp

            chosen = min(cand_rows, key=_cost)

        path0 = chosen["path"]
        d_b0_world = _unit_xy(chosen["d_b_from_world"])
        d_a0_world = _unit_xy(chosen["d_a_world"])
        d_b0_local = _dir_world_to_local(b_point, d_b0_world)
        d_a0_local = _dir_world_to_local(a_point, d_a0_world)

        # Family 1 is requested as X-mirror of family 0.
        d_b1_world = _mirror_world_about_x(d_b0_world)
        d_a1_world = _mirror_world_about_x(d_a0_world)
        d_b1_local = _dir_world_to_local(b_point, d_b1_world)
        d_a1_local = _dir_world_to_local(a_point, d_a1_world)

        # Family 0
        cand_direction[a_i, 0] = int(path0.direction)
        cand_is_turning[a_i, 0] = str(path0.branch) == "turning"
        cand_target_azimuth_rad[a_i, 0] = float(path0.target_azimuth_rad)
        cand_travel_time_s[a_i, 0] = float(path0.travel_time_s)
        cand_impact_parameter_m[a_i, 0] = float(path0.impact_parameter_m)
        cand_gamma_at_b_rad[a_i, 0] = float(chosen["gamma_b"])
        cand_gamma_at_a_rad[a_i, 0] = float(chosen["gamma_a"])
        cand_dir_at_b_local_xy[a_i, 0, :] = d_b0_local
        cand_dir_at_a_local_xy[a_i, 0, :] = d_a0_local

        # Family 1 (mirrored)
        cand_direction[a_i, 1] = -int(path0.direction)
        cand_is_turning[a_i, 1] = bool(cand_is_turning[a_i, 0])
        cand_target_azimuth_rad[a_i, 1] = float(path0.target_azimuth_rad)
        cand_travel_time_s[a_i, 1] = float(path0.travel_time_s)
        cand_impact_parameter_m[a_i, 1] = float(path0.impact_parameter_m)
        cand_gamma_at_b_rad[a_i, 1] = float(chosen["gamma_b"])
        cand_gamma_at_a_rad[a_i, 1] = float(chosen["gamma_a"])
        cand_dir_at_b_local_xy[a_i, 1, :] = d_b1_local
        cand_dir_at_a_local_xy[a_i, 1, :] = d_a1_local

        prev_obs_ang = float(chosen["obs_ang"])
        prev_imp = float(path0.impact_parameter_m)

    # Validity checks (same style as raw saver): residual + null-interval.
    residual_fail_rows: list[tuple[int, int, float, float, float, str]] = []
    null_fail_rows: list[tuple[int, int, float]] = []
    for a_i in range(n_a):
        a_point = a_points[a_i]
        r_a = float(np.linalg.norm(a_point))
        r_b = float(np.linalg.norm(b_point))
        for fam in range(n_fam):
            if not bool(cand_ok[a_i, fam]):
                continue
            is_turning = bool(cand_is_turning[a_i, fam])
            target_phi = float(cand_target_azimuth_rad[a_i, fam])
            b_imp = float(cand_impact_parameter_m[a_i, fam])
            t_saved = float(cand_travel_time_s[a_i, fam])
            branch = "turning" if is_turning else "monotonic"
            try:
                sols = bh._solve_for_target_azimuth(r_a, r_b, target_phi, use_gpu=False)
                branch_sols = [s for s in sols if str(s[2]) == branch]
                if not branch_sols:
                    raise RuntimeError("No branch-matched solution.")
                b_calc, t_calc, _ = min(branch_sols, key=lambda s: abs(float(s[0]) - b_imp))
                b_calc = float(b_calc)
                t_calc = float(t_calc)
                b_res_rel = abs(b_calc - b_imp) / max(abs(b_imp), 1e-12)
                t_res_rel = abs(t_calc - t_saved) / max(abs(t_saved), 1e-12)
                if not is_turning:
                    phi_calc = float(bh._delta_phi_mono(r_a, r_b, b_calc, use_gpu=False))
                    phi_res = float(phi_calc - target_phi)
                else:
                    phi_res = 0.0
                if (
                    abs(phi_res) > float(args.residual_max_phi_rad)
                    or t_res_rel > float(args.residual_max_time_rel)
                    or b_res_rel > float(args.residual_max_impact_rel)
                ):
                    residual_fail_rows.append((a_i, fam, phi_res, t_res_rel, b_res_rel, branch))
            except Exception:
                residual_fail_rows.append((a_i, fam, float("inf"), float("inf"), float("inf"), branch))

            try:
                r_prof, th_prof = _build_curve_profile(
                    bh=bh,
                    a_point=a_point,
                    b_point=b_point,
                    impact_parameter_m=b_imp,
                    target_azimuth_rad=target_phi,
                    is_turning=is_turning,
                    n=max(1200, int(args.null_check_segments) * 4),
                )
                nr = _null_ratio(
                    rs_m=rs,
                    r_m=r_prof,
                    theta=th_prof,
                    impact_parameter_m=b_imp,
                    segments=int(args.null_check_segments),
                )
                if nr > float(args.null_check_max_ratio):
                    null_fail_rows.append((a_i, fam, nr))
            except Exception:
                null_fail_rows.append((a_i, fam, float("inf")))

    if residual_fail_rows:
        print(
            f"Residual check FAILED: {len(residual_fail_rows)}/{n_a*n_fam} candidates exceed tolerances.",
            flush=True,
        )
        for a_i, fam, phi_res, t_res_rel, b_res_rel, branch in residual_fail_rows[:12]:
            print(
                f"  A={a_i:3d} fam={fam:d} branch={branch:9s} phi_res={phi_res:.3e} "
                f"rel_t={t_res_rel:.3e} rel_b={b_res_rel:.3e}",
                flush=True,
            )
        raise RuntimeError("Residual-based solver check failed; aborting save.")
    print("Residual check passed for all candidates.", flush=True)

    if null_fail_rows:
        print(
            f"Null-interval check FAILED: {len(null_fail_rows)}/{n_a*n_fam} candidates exceed "
            f"|sum(ds^2)|/sum(dx^2)>{float(args.null_check_max_ratio):.3e}.",
            flush=True,
        )
        for a_i, fam, nr in null_fail_rows[:12]:
            print(f"  A={a_i:3d} fam={fam:d} ratio={nr:.3e}", flush=True)
        if bool(args.null_check_strict):
            raise RuntimeError("Null-interval check failed; aborting save.")
        print("Continuing despite null-interval failures (warning-only mode).", flush=True)
    else:
        print("Null-interval check passed for all candidates.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rs_m=np.asarray(rs, dtype=float),
        a_phi_rad=a_phi.astype(np.float64),
        a_points_m=a_points.astype(np.float64),
        sky_radius_rs=np.asarray(float(args.sky_radius_rs), dtype=float),
        b_radius_rs=np.asarray(float(args.b_radius_rs), dtype=float),
        b_point_m=b_point.astype(np.float64),
        cand_ok=cand_ok,
        cand_direction=cand_direction,
        cand_is_turning=cand_is_turning,
        cand_target_azimuth_rad=cand_target_azimuth_rad,
        cand_travel_time_s=cand_travel_time_s,
        cand_impact_parameter_m=cand_impact_parameter_m,
        cand_gamma_at_b_rad=cand_gamma_at_b_rad,
        cand_gamma_at_a_rad=cand_gamma_at_a_rad,
        cand_dir_at_b_local_xy=cand_dir_at_b_local_xy,
        cand_dir_at_a_local_xy=cand_dir_at_a_local_xy,
        observer_dir_convention=np.asarray("coming_from", dtype="<U32"),
        family_definition=np.asarray("family0_continuation_family1_x_mirror", dtype="<U64"),
    )
    print(f"Saved fixed-B two-family sky candidates: {args.output}", flush=True)


if __name__ == "__main__":
    main()
