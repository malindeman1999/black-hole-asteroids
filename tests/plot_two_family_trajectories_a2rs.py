from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole, cp
from precompute_earliest_grid import (
    _arrival_direction_at_b_for_pair,
    _build_path_profile,
    _direction_from_angle_at_a_for_pair,
    _local_gamma_at_radius,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot two-family geodesic trajectories using continuation solve with fixed observer B on +X. "
            "Family 0 is tracked over A-phi; family 1 is X-mirror of family 0."
        )
    )
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="high")
    p.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--a-radius-rs", type=float, default=2.0, help="Source A ring radius in Rs.")
    p.add_argument("--b-radius-rs", type=float, default=10.0, help="Observer B radius in Rs (on +X axis).")
    p.add_argument("--a-phi-count", type=int, default=20, help="A azimuth samples over [0, 360).")
    p.add_argument(
        "--continuity-impact-weight",
        type=float,
        default=0.2,
        help="Impact-parameter weight in continuation cost.",
    )
    p.add_argument("--path-samples", type=int, default=1800, help="Samples per reconstructed path.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests") / "two_family_trajectories_a2rs.png",
    )
    p.add_argument(
        "--debug-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot candidate solutions considered by the chooser (default: true).",
    )
    p.add_argument(
        "--debug-a-index",
        type=int,
        default=0,
        help="A index to debug; -1 plots all A indices (default: 0).",
    )
    p.add_argument(
        "--debug-output",
        type=Path,
        default=Path("tests") / "two_family_trajectories_a2rs_candidates.png",
    )
    p.add_argument(
        "--debug-axis-rmin-rs",
        type=float,
        default=1.2,
        help="Debug radial scan min A radius in Rs with A and B fixed on +X.",
    )
    p.add_argument(
        "--debug-axis-rmax-rs",
        type=float,
        default=12.0,
        help="Debug radial scan max A radius in Rs with A and B fixed on +X.",
    )
    p.add_argument(
        "--debug-axis-count",
        type=int,
        default=9,
        help="Number of debug A radii in axis scan (A index maps to radius).",
    )
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _unit_xy(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / n


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


def _build_curve_xy_from_candidate(
    a_point: np.ndarray,
    b_point: np.ndarray,
    impact_parameter_m: float,
    target_azimuth_rad: float,
    is_turning: bool,
    bh: SchwarzschildBlackHole,
    path_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(impact_parameter_m)) < 1e-14:
        x = np.linspace(float(a_point[0]), float(b_point[0]), max(2, int(path_samples)), dtype=float)
        y = np.linspace(float(a_point[1]), float(b_point[1]), max(2, int(path_samples)), dtype=float)
        return x, y

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
        n=max(600, int(path_samples)),
    )

    is_short = abs(float(target_azimuth_rad) - gamma_short) <= abs(float(target_azimuth_rad) - gamma_long)
    orient_sign = short_sign if is_short else -short_sign
    theta = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
    theta = np.unwrap(theta)

    r_m = np.asarray(r_samples, dtype=float)
    x = r_m * np.cos(theta)
    y = r_m * np.sin(theta)
    return x, y


def main() -> None:
    args = _parse_args()
    if int(args.a_phi_count) < 3:
        raise ValueError("--a-phi-count must be >= 3")

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = float(bh.schwarzschild_radius_m)
    use_gpu_active = bool(args.use_gpu and cp is not None)

    a_r = float(args.a_radius_rs) * rs
    b_r = float(args.b_radius_rs) * rs
    a_phi = np.linspace(0.0, 2.0 * pi, int(args.a_phi_count), endpoint=False, dtype=float)
    a_points = np.zeros((a_phi.size, 3), dtype=float)
    a_points[:, 0] = a_r * np.cos(a_phi)
    a_points[:, 1] = a_r * np.sin(a_phi)
    b_point = np.asarray([b_r, 0.0, 0.0], dtype=float)

    family0_curves: list[tuple[np.ndarray, np.ndarray]] = []
    family1_curves: list[tuple[np.ndarray, np.ndarray]] = []

    prev_obs_ang: float | None = None
    prev_imp: float | None = None
    for a_i, a_point in enumerate(a_points):
        print(f"Solving A {a_i + 1}/{a_points.shape[0]}...", flush=True)
        rr = bh.find_all_geodesic_candidates(a_point, b_point, a_before_b=True, use_gpu=use_gpu_active)
        cand_rows: list[dict] = []
        gamma_short = _short_azimuth(a_point, b_point)
        to_b_world = _unit_xy(np.asarray(b_point[:2], dtype=float) - np.asarray(a_point[:2], dtype=float))
        from_a_world = _unit_xy(np.asarray(a_point[:2], dtype=float) - np.asarray(b_point[:2], dtype=float))
        for path in rr.paths:
            gamma_b = _local_gamma_at_radius(rs, b_r, float(path.impact_parameter_m))
            gamma_a = _local_gamma_at_radius(rs, a_r, float(path.impact_parameter_m))
            d_b_prop = _arrival_direction_at_b_for_pair(
                a_point=a_point,
                b_point=b_point,
                gamma_at_b=gamma_b,
                branch_side=int(path.direction),
                branch_name=str(path.branch),
            )
            d_a_prop = _direction_from_angle_at_a_for_pair(
                a_point=a_point,
                b_point=b_point,
                gamma_at_a=gamma_a,
                branch_side=int(path.direction),
            )
            d_b_from = -_unit_xy(d_b_prop)
            obs_ang = float(np.arctan2(float(d_b_from[1]), float(d_b_from[0])))
            direct_cost = _angle_between(d_a_prop, to_b_world) + _angle_between(d_b_from, from_a_world)
            cand_rows.append({"path": path, "obs_ang": obs_ang, "direct_cost": direct_cost})
        if prev_obs_ang is None and gamma_short < 1e-12:
            class _DirectPath:
                direction = +1
                branch = "monotonic"
                target_azimuth_rad = 0.0
                impact_parameter_m = 0.0
                travel_time_s = 0.0

            direct_path = _DirectPath()
            direct_path.travel_time_s = _radial_null_travel_time_s(rs, a_r, b_r)
            obs_ang_direct = float(np.arctan2(float((-to_b_world)[1]), float((-to_b_world)[0])))
            cand_rows.append({"path": direct_path, "obs_ang": obs_ang_direct, "direct_cost": 0.0})

        if not cand_rows:
            raise RuntimeError(f"No candidates found for A index {a_i}.")

        if prev_obs_ang is None:
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
            def _cost(row: dict) -> float:
                c_ang = _wrap_angle_diff(float(row["obs_ang"]), float(prev_obs_ang))
                c_imp = (
                    abs(float(row["path"].impact_parameter_m) - float(prev_imp)) / max(abs(float(prev_imp)), 1.0)
                    if prev_imp is not None
                    else 0.0
                )
                return c_ang + float(args.continuity_impact_weight) * c_imp

            chosen = min(cand_rows, key=_cost)

        chosen_idx = int(cand_rows.index(chosen))
        path0 = chosen["path"]
        x0, y0 = _build_curve_xy_from_candidate(
            a_point=a_point,
            b_point=b_point,
            impact_parameter_m=float(path0.impact_parameter_m),
            target_azimuth_rad=float(path0.target_azimuth_rad),
            is_turning=(str(path0.branch) == "turning"),
            bh=bh,
            path_samples=int(args.path_samples),
        )
        family0_curves.append((x0, y0))
        family1_curves.append((x0, -y0))

        prev_obs_ang = float(chosen["obs_ang"])
        prev_imp = float(path0.impact_parameter_m)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 8.2))
    t = np.linspace(0.0, 2.0 * pi, 320)
    ax.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.2)

    for x0, y0 in family0_curves:
        ax.plot(x0, y0, color="#1f77b4", lw=1.4, alpha=0.85)
    for x1, y1 in family1_curves:
        ax.plot(x1, y1, color="#d62728", lw=1.1, alpha=0.75)

    ax.scatter(a_points[:, 0], a_points[:, 1], c="0.35", s=14, alpha=0.7, label="A samples (2 Rs ring)")
    ax.scatter([b_point[0]], [b_point[1]], c="green", marker="x", s=72, linewidths=2.0, label="Observer B")

    lim = 1.12 * max(float(np.linalg.norm(b_point[:2])), float(np.max(np.linalg.norm(a_points[:, :2], axis=1))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Two-family trajectories with A ring at {args.a_radius_rs:.3f} Rs, B at {args.b_radius_rs:.3f} Rs\n"
        f"family 0=blue (continuation), family 1=red (X-mirror), nA={a_points.shape[0]}",
        fontsize=10,
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}", flush=True)

    if bool(args.debug_candidates):
        n_dbg = max(1, int(args.debug_axis_count))
        rmin_dbg = max(1.0 + 1e-6, float(args.debug_axis_rmin_rs))
        rmax_dbg = max(rmin_dbg, float(args.debug_axis_rmax_rs))
        a_r_dbg_rs = np.linspace(rmin_dbg, rmax_dbg, n_dbg, dtype=float)

        debug_rows: list[dict] = []
        prev_obs_dbg: float | None = None
        prev_imp_dbg: float | None = None
        for dbg_i, rA_rs in enumerate(a_r_dbg_rs):
            a_dbg = np.asarray([float(rA_rs) * rs, 0.0, 0.0], dtype=float)
            rr_dbg = bh.find_all_geodesic_candidates(a_dbg, b_point, a_before_b=True, use_gpu=use_gpu_active)
            cand_rows_dbg: list[dict] = []
            gamma_short_dbg = _short_azimuth(a_dbg, b_point)
            to_b_dbg = _unit_xy(np.asarray(b_point[:2], dtype=float) - np.asarray(a_dbg[:2], dtype=float))
            from_a_dbg = _unit_xy(np.asarray(a_dbg[:2], dtype=float) - np.asarray(b_point[:2], dtype=float))
            for path in rr_dbg.paths:
                gamma_b_dbg = _local_gamma_at_radius(rs, b_r, float(path.impact_parameter_m))
                gamma_a_dbg = _local_gamma_at_radius(rs, float(rA_rs) * rs, float(path.impact_parameter_m))
                d_b_prop_dbg = _arrival_direction_at_b_for_pair(
                    a_point=a_dbg,
                    b_point=b_point,
                    gamma_at_b=gamma_b_dbg,
                    branch_side=int(path.direction),
                    branch_name=str(path.branch),
                )
                d_a_prop_dbg = _direction_from_angle_at_a_for_pair(
                    a_point=a_dbg,
                    b_point=b_point,
                    gamma_at_a=gamma_a_dbg,
                    branch_side=int(path.direction),
                )
                d_b_from_dbg = -_unit_xy(d_b_prop_dbg)
                obs_ang_dbg = float(np.arctan2(float(d_b_from_dbg[1]), float(d_b_from_dbg[0])))
                direct_cost_dbg = _angle_between(d_a_prop_dbg, to_b_dbg) + _angle_between(d_b_from_dbg, from_a_dbg)
                cand_rows_dbg.append({"path": path, "obs_ang": obs_ang_dbg, "direct_cost": direct_cost_dbg})
            if gamma_short_dbg < 1e-12:
                class _DirectPath:
                    direction = +1
                    branch = "monotonic"
                    target_azimuth_rad = 0.0
                    impact_parameter_m = 0.0
                    travel_time_s = 0.0

                direct_path_dbg = _DirectPath()
                direct_path_dbg.travel_time_s = _radial_null_travel_time_s(rs, float(rA_rs) * rs, b_r)
                obs_ang_direct_dbg = float(np.arctan2(float((-to_b_dbg)[1]), float((-to_b_dbg)[0])))
                cand_rows_dbg.append({"path": direct_path_dbg, "obs_ang": obs_ang_direct_dbg, "direct_cost": 0.0})

            if not cand_rows_dbg:
                continue
            if prev_obs_dbg is None:
                chosen_dbg = min(
                    cand_rows_dbg,
                    key=lambda r: (
                        abs(float(r["path"].target_azimuth_rad) - float(gamma_short_dbg)),
                        0 if str(r["path"].branch) == "monotonic" else 1,
                        float(r["direct_cost"]),
                        float(r["path"].travel_time_s),
                    ),
                )
            else:
                def _cost_dbg(row: dict) -> float:
                    c_ang = _wrap_angle_diff(float(row["obs_ang"]), float(prev_obs_dbg))
                    c_imp = (
                        abs(float(row["path"].impact_parameter_m) - float(prev_imp_dbg)) / max(abs(float(prev_imp_dbg)), 1.0)
                        if prev_imp_dbg is not None
                        else 0.0
                    )
                    return c_ang + float(args.continuity_impact_weight) * c_imp

                chosen_dbg = min(cand_rows_dbg, key=_cost_dbg)
            chosen_dbg_idx = int(cand_rows_dbg.index(chosen_dbg))
            cand_curves_dbg = []
            for j_row, row in enumerate(cand_rows_dbg):
                pth = row["path"]
                try:
                    cx, cy = _build_curve_xy_from_candidate(
                        a_point=a_dbg,
                        b_point=b_point,
                        impact_parameter_m=float(pth.impact_parameter_m),
                        target_azimuth_rad=float(pth.target_azimuth_rad),
                        is_turning=(str(pth.branch) == "turning"),
                        bh=bh,
                        path_samples=int(args.path_samples),
                    )
                    cand_curves_dbg.append((j_row, cx, cy, pth))
                except Exception:
                    continue
            debug_rows.append(
                {
                    "a_i": int(dbg_i),
                    "a_r_rs": float(rA_rs),
                    "a_point": a_dbg.copy(),
                    "chosen_idx": chosen_dbg_idx,
                    "cand_curves": cand_curves_dbg,
                }
            )
            prev_obs_dbg = float(chosen_dbg["obs_ang"])
            prev_imp_dbg = float(chosen_dbg["path"].impact_parameter_m)

        if debug_rows:
            if int(args.debug_a_index) == -1:
                n = len(debug_rows)
                ncols = int(np.ceil(np.sqrt(float(n))))
                nrows = int(np.ceil(float(n) / float(ncols)))
                figd, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.9 * nrows))
                axes_arr = np.atleast_1d(axes).ravel()
            else:
                idx = int(np.clip(int(args.debug_a_index), 0, len(debug_rows) - 1))
                debug_rows = [debug_rows[idx]]
                figd, ax_single = plt.subplots(1, 1, figsize=(7.2, 7.2))
                axes_arr = np.asarray([ax_single], dtype=object)

            t_dbg = np.linspace(0.0, 2.0 * pi, 220)
            lim = 1.12 * max(float(np.linalg.norm(b_point[:2])), float(rmax_dbg * rs))
            for k, row in enumerate(debug_rows):
                axd = axes_arr[k]
                a_i = int(row["a_i"])
                a_r_rs = float(row["a_r_rs"])
                a_pt = np.asarray(row["a_point"], dtype=float)
                chosen_idx = int(row["chosen_idx"])
                cand_curves = row["cand_curves"]

                axd.plot(rs * np.cos(t_dbg), rs * np.sin(t_dbg), "k-", lw=1.0)
                for j_row, cx, cy, pth in cand_curves:
                    is_chosen = int(j_row) == chosen_idx
                    axd.plot(
                        cx,
                        cy,
                        color=("#1f77b4" if is_chosen else "#808080"),
                        lw=(2.2 if is_chosen else 0.9),
                        alpha=(0.95 if is_chosen else 0.55),
                    )
                    if not is_chosen:
                        continue
                    txt = (
                        f"chosen j={int(j_row)} dir={int(pth.direction):+d} "
                        f"branch={str(pth.branch)} "
                        f"phi={np.degrees(float(pth.target_azimuth_rad)):.1f} deg "
                        f"t={float(pth.travel_time_s):.2f}s"
                    )
                    axd.text(0.03, 0.97, txt, transform=axd.transAxes, va="top", ha="left", fontsize=8)

                axd.scatter([a_pt[0]], [a_pt[1]], c="black", s=26, zorder=6)
                axd.scatter([b_point[0]], [b_point[1]], c="green", marker="x", s=58, linewidths=1.8, zorder=7)
                axd.set_xlim(-lim, lim)
                axd.set_ylim(-lim, lim)
                axd.set_aspect("equal", adjustable="box")
                axd.grid(alpha=0.2)
                axd.set_xticks([])
                axd.set_yticks([])
                axd.set_title(f"A index {a_i} | A radius {a_r_rs:.3f} Rs (A,B on +X)", fontsize=9)

            for k in range(len(debug_rows), len(axes_arr)):
                axes_arr[k].axis("off")

            figd.suptitle(
                "Axis debug: A and B fixed at 0 deg (+X); gray=candidates, blue=selected",
                fontsize=10,
            )
            figd.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            args.debug_output.parent.mkdir(parents=True, exist_ok=True)
            figd.savefig(args.debug_output, dpi=170)
            print(f"Saved debug candidate figure: {args.debug_output}", flush=True)

    print(f"Mode: {'GPU' if use_gpu_active else 'CPU'}", flush=True)

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
