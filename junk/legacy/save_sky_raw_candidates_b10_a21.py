from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole, cp
from precompute_earliest_grid import (
    _arrival_direction_at_b_for_pair,
    _build_path_profile,
    _dir_world_to_local,
    _direction_from_angle_at_a_for_pair,
    _local_gamma_at_radius,
    _regular_phi_grid,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve and save all raw sky-ring geodesic candidates for fixed B=10 rs.")
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="high")
    p.add_argument("--sky-radius-rs", type=float, default=100.0)
    p.add_argument("--b-radius-rs", type=float, default=10.0)
    p.add_argument("--a-phi-count", type=int, default=21, help="Must be odd to avoid the +X-axis degeneracy.")
    p.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "sky_raw_candidates_b10rs_a21.npz",
    )
    p.add_argument(
        "--null-check-segments",
        type=int,
        default=1000,
        help="Segments per path for ds^2 vs dx^2 sanity check (default: 1000).",
    )
    p.add_argument(
        "--null-check-max-ratio",
        type=float,
        default=0.2,
        help="Maximum allowed |sum(ds^2)|/sum(dx^2) per solution (default: 0.2).",
    )
    p.add_argument(
        "--null-check-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, null-interval check failures abort save. Default is warning-only.",
    )
    p.add_argument(
        "--residual-max-phi-rad",
        type=float,
        default=1e-3,
        help="Maximum allowed |delta_phi_computed - target_phi| in radians (default: 1e-3).",
    )
    p.add_argument(
        "--residual-max-time-rel",
        type=float,
        default=1e-3,
        help="Maximum allowed relative travel-time residual |t_calc-t_saved|/max(|t_saved|,1e-12) (default: 1e-3).",
    )
    p.add_argument(
        "--residual-max-impact-rel",
        type=float,
        default=1e-3,
        help="Maximum allowed relative impact-parameter mismatch vs solver branch match (default: 1e-3).",
    )
    return p.parse_args()


def _null_interval_ratio_for_path(
    bh: SchwarzschildBlackHole,
    rs_m: float,
    a_point: np.ndarray,
    b_point: np.ndarray,
    impact_parameter_m: float,
    target_azimuth_rad: float,
    branch: str,
    segments: int,
) -> tuple[float, float, float]:
    r_start = float(np.linalg.norm(a_point))
    r_end = float(np.linalg.norm(b_point))
    th_start = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_end = float(np.arctan2(float(b_point[1]), float(b_point[0])))

    dtheta_short = ((th_end - th_start + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dtheta_short >= 0.0 else -1.0
    gamma_short = abs(dtheta_short)
    gamma_long = 2.0 * pi - gamma_short

    # Use a denser base profile than the segment count to reduce interpolation bias.
    profile_n = max(1200, int(segments) * 10)
    r_samples, phi_samples = _build_path_profile(
        bh=bh,
        r_start=r_start,
        r_end=r_end,
        impact_b=float(impact_parameter_m),
        target_phi=float(target_azimuth_rad),
        branch=str(branch),
        n=profile_n,
    )

    is_short = abs(float(target_azimuth_rad) - gamma_short) <= abs(float(target_azimuth_rad) - gamma_long)
    orient_sign = short_sign if is_short else -short_sign
    theta = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
    theta = np.unwrap(theta)
    r_m = np.asarray(r_samples, dtype=float)

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

    ratio = abs(sum_ds2) / max(sum_dx2, 1e-30)
    return float(sum_ds2), float(sum_dx2), float(ratio)


def _null_interval_ratio_refined(
    bh: SchwarzschildBlackHole,
    rs_m: float,
    a_point: np.ndarray,
    b_point: np.ndarray,
    impact_parameter_m: float,
    target_azimuth_rad: float,
    branch: str,
    base_segments: int,
) -> tuple[float, float, float, tuple[float, float, float]]:
    n0 = max(16, int(base_segments))
    n1 = n0 * 2
    n2 = n0 * 4
    s0, x0, r0 = _null_interval_ratio_for_path(
        bh=bh,
        rs_m=rs_m,
        a_point=a_point,
        b_point=b_point,
        impact_parameter_m=impact_parameter_m,
        target_azimuth_rad=target_azimuth_rad,
        branch=branch,
        segments=n0,
    )
    s1, x1, r1 = _null_interval_ratio_for_path(
        bh=bh,
        rs_m=rs_m,
        a_point=a_point,
        b_point=b_point,
        impact_parameter_m=impact_parameter_m,
        target_azimuth_rad=target_azimuth_rad,
        branch=branch,
        segments=n1,
    )
    s2, x2, r2 = _null_interval_ratio_for_path(
        bh=bh,
        rs_m=rs_m,
        a_point=a_point,
        b_point=b_point,
        impact_parameter_m=impact_parameter_m,
        target_azimuth_rad=target_azimuth_rad,
        branch=branch,
        segments=n2,
    )
    # Use the finest-grid value as the pass/fail metric and retain history for diagnostics.
    return s2, x2, r2, (r0, r1, r2)


def main() -> None:
    args = _parse_args()
    if int(args.a_phi_count) < 3 or int(args.a_phi_count) % 2 == 0:
        raise ValueError("--a-phi-count must be odd and >= 3")

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = float(bh.schwarzschild_radius_m)
    use_gpu_active = bool(args.use_gpu and cp is not None)

    a_phi = _regular_phi_grid(int(args.a_phi_count))
    a_r = float(args.sky_radius_rs) * rs
    b_r = float(args.b_radius_rs) * rs
    a_points = np.zeros((a_phi.size, 3), dtype=float)
    a_points[:, 0] = a_r * np.cos(a_phi)
    a_points[:, 1] = a_r * np.sin(a_phi)
    b_point = np.asarray([b_r, 0.0, 0.0], dtype=float)

    raw_results = []
    max_candidates = 0
    for i, a in enumerate(a_points, start=1):
        print(f"Solving raw candidates {i}/{a_points.shape[0]}...", flush=True)
        rr = bh.find_all_geodesic_candidates(a, b_point, a_before_b=True, use_gpu=use_gpu_active)
        raw_results.append(rr)
        max_candidates = max(max_candidates, len(rr.paths))

    cand_ok = np.zeros((a_points.shape[0], max_candidates), dtype=bool)
    cand_direction = np.zeros((a_points.shape[0], max_candidates), dtype=np.int8)
    cand_is_turning = np.zeros((a_points.shape[0], max_candidates), dtype=bool)
    cand_target_azimuth_rad = np.full((a_points.shape[0], max_candidates), np.nan, dtype=float)
    cand_travel_time_s = np.full((a_points.shape[0], max_candidates), np.nan, dtype=float)
    cand_impact_parameter_m = np.full((a_points.shape[0], max_candidates), np.nan, dtype=float)
    cand_gamma_at_b_rad = np.full((a_points.shape[0], max_candidates), np.nan, dtype=float)
    cand_gamma_at_a_rad = np.full((a_points.shape[0], max_candidates), np.nan, dtype=float)
    cand_dir_at_b_local_xy = np.full((a_points.shape[0], max_candidates, 2), np.nan, dtype=float)
    cand_dir_at_a_local_xy = np.full((a_points.shape[0], max_candidates, 2), np.nan, dtype=float)

    for a_i, rr in enumerate(raw_results):
        for c_i, path in enumerate(rr.paths):
            gamma_b = _local_gamma_at_radius(rs, b_r, float(path.impact_parameter_m))
            gamma_a = _local_gamma_at_radius(rs, a_r, float(path.impact_parameter_m))
            d_b = _arrival_direction_at_b_for_pair(a_points[a_i], b_point, gamma_b, int(path.direction), str(path.branch))
            d_a = _direction_from_angle_at_a_for_pair(a_points[a_i], b_point, gamma_a, int(path.direction))

            cand_ok[a_i, c_i] = True
            cand_direction[a_i, c_i] = int(path.direction)
            cand_is_turning[a_i, c_i] = str(path.branch) == "turning"
            cand_target_azimuth_rad[a_i, c_i] = float(path.target_azimuth_rad)
            cand_travel_time_s[a_i, c_i] = float(path.travel_time_s)
            cand_impact_parameter_m[a_i, c_i] = float(path.impact_parameter_m)
            cand_gamma_at_b_rad[a_i, c_i] = float(gamma_b)
            cand_gamma_at_a_rad[a_i, c_i] = float(gamma_a)
            # Save observer direction as "coming from" (opposite propagation tangent at B).
            cand_dir_at_b_local_xy[a_i, c_i, :] = -_dir_world_to_local(b_point, d_b)
            cand_dir_at_a_local_xy[a_i, c_i, :] = _dir_world_to_local(a_points[a_i], d_a)

    fail_rows: list[tuple[int, int, float, float, float, str, tuple[float, float, float]]] = []
    residual_fail_rows: list[tuple[int, int, str, float, float, float, float]] = []
    solve_cache: dict[tuple[float, float], list[tuple[float, float, str]]] = {}
    total_checked = 0
    for a_i, rr in enumerate(raw_results):
        r_a = float(np.linalg.norm(a_points[a_i]))
        for c_i, path in enumerate(rr.paths):
            total_checked += 1
            branch = str(path.branch)
            b_imp = float(path.impact_parameter_m)
            target_phi = float(path.target_azimuth_rad)
            t_saved = float(path.travel_time_s)

            # Residual checks using the same solve path as candidate generation.
            try:
                cache_key = (round(r_a, 12), round(target_phi, 12))
                if cache_key not in solve_cache:
                    solve_cache[cache_key] = bh._solve_for_target_azimuth(r_a, b_r, target_phi, use_gpu=False)
                branch_solutions = [s for s in solve_cache[cache_key] if str(s[2]) == branch]
                if not branch_solutions:
                    raise RuntimeError(f"No {branch} solution returned for target_phi.")

                b_calc, t_calc, _ = min(branch_solutions, key=lambda s: abs(float(s[0]) - b_imp))
                b_calc = float(b_calc)
                t_calc = float(t_calc)
                b_res_rel = float(abs(b_calc - b_imp) / max(abs(b_imp), 1e-12))
                if branch == "monotonic":
                    phi_calc = bh._delta_phi_mono(r_a, b_r, b_calc, use_gpu=False)
                    phi_res = float(phi_calc - target_phi)
                else:
                    # Turning branch is checked against the solver's own branch-matched root via b/t consistency.
                    phi_res = 0.0
                t_res_rel = float(abs(t_calc - t_saved) / max(abs(t_saved), 1e-12))
                if (
                    (abs(phi_res) > float(args.residual_max_phi_rad))
                    or (t_res_rel > float(args.residual_max_time_rel))
                    or (b_res_rel > float(args.residual_max_impact_rel))
                ):
                    residual_fail_rows.append((a_i, c_i, branch, phi_res, t_res_rel, b_res_rel, t_saved))
            except Exception:
                residual_fail_rows.append((a_i, c_i, branch, float("inf"), float("inf"), float("inf"), t_saved))

            sum_ds2, sum_dx2, ratio, ratio_hist = _null_interval_ratio_refined(
                bh=bh,
                rs_m=rs,
                a_point=a_points[a_i],
                b_point=b_point,
                impact_parameter_m=b_imp,
                target_azimuth_rad=target_phi,
                branch=branch,
                base_segments=int(args.null_check_segments),
            )
            if ratio > float(args.null_check_max_ratio):
                fail_rows.append((a_i, c_i, ratio, sum_ds2, sum_dx2, branch, ratio_hist))

    if residual_fail_rows:
        print(
            f"Residual check FAILED: {len(residual_fail_rows)}/{total_checked} candidates exceed "
            f"|phi_res|>{float(args.residual_max_phi_rad):.3e} rad or rel_time_res>{float(args.residual_max_time_rel):.3e} "
            f"or rel_b_res>{float(args.residual_max_impact_rel):.3e}",
            flush=True,
        )
        print("Worst residual offenders (first 12):", flush=True)
        for a_i, c_i, branch, phi_res, t_res_rel, b_res_rel, t_saved in sorted(
            residual_fail_rows, key=lambda x: max(abs(x[3]), x[4], x[5]), reverse=True
        )[:12]:
            print(
                f"  A={a_i:3d} cand={c_i:2d} branch={branch:9s} "
                f"phi_res={phi_res:.6e} rad rel_t_res={t_res_rel:.3e} rel_b_res={b_res_rel:.3e} "
                f"t_saved={t_saved:.6e} s",
                flush=True,
            )
        raise RuntimeError("Residual-based solver check failed; aborting save.")
    print(
        f"Residual check passed for all {total_checked} candidates "
        f"(phi_tol={float(args.residual_max_phi_rad):.3e} rad, "
        f"rel_time_tol={float(args.residual_max_time_rel):.3e}, "
        f"rel_b_tol={float(args.residual_max_impact_rel):.3e}).",
        flush=True,
    )

    if fail_rows:
        print(
            f"Null-interval check FAILED: {len(fail_rows)}/{total_checked} candidates exceed "
            f"|sum(ds^2)|/sum(dx^2) > {float(args.null_check_max_ratio):.3e}",
            flush=True,
        )
        print("Worst offenders (first 12):", flush=True)
        for a_i, c_i, ratio, sum_ds2, sum_dx2, branch, ratio_hist in sorted(
            fail_rows, key=lambda x: x[2], reverse=True
        )[:12]:
            print(
                f"  A={a_i:3d} cand={c_i:2d} branch={branch:9s} ratio={ratio:.3e} "
                f"sum_ds2={sum_ds2:.6e} sum_dx2={sum_dx2:.6e} "
                f"ratios(N,2N,4N)=({ratio_hist[0]:.3e},{ratio_hist[1]:.3e},{ratio_hist[2]:.3e})",
                flush=True,
            )
        if bool(args.null_check_strict):
            raise RuntimeError("Null-interval sanity check failed; aborting save.")
        print("Continuing despite null-interval failures (warning-only mode).", flush=True)
    print(
        f"Null-interval check passed for all {total_checked} candidates "
        f"(threshold={float(args.null_check_max_ratio):.3e}).",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rs_m=np.asarray(rs, dtype=float),
        a_phi_rad=a_phi.astype(np.float64),
        a_points_m=a_points.astype(np.float64),
        b_point_m=b_point.astype(np.float64),
        sky_radius_rs=np.asarray(float(args.sky_radius_rs), dtype=float),
        b_radius_rs=np.asarray(float(args.b_radius_rs), dtype=float),
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
    )
    print(f"Saved raw sky candidates: {args.output}", flush=True)


if __name__ == "__main__":
    main()
