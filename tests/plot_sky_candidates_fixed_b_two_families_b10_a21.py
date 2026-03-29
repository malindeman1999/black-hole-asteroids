from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole
from precompute_earliest_grid import _build_path_profile


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot raw candidate observer/source directions for the saved sky-ring solve.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "sky_candidates_fixed_b_two_families_b10rs_a20.npz",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests") / "sky_candidates_fixed_b_two_families_b10rs_a20.png",
    )
    p.add_argument(
        "--paths-output",
        type=Path,
        default=None,
        help="Optional output for all-paths figure (default: <output>_paths.png).",
    )
    p.add_argument(
        "--path-samples",
        type=int,
        default=2500,
        help="Samples per reconstructed geodesic curve in the paths figure (default: 2500).",
    )
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _build_curve_xy_from_candidate(
    a_point: np.ndarray,
    b_point: np.ndarray,
    impact_parameter_m: float,
    target_azimuth_rad: float,
    is_turning: bool,
    bh: SchwarzschildBlackHole,
    path_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x, y, r_m, theta


def _interval_diagnostics_from_profile(
    r_m: np.ndarray,
    theta_rad: np.ndarray,
    impact_parameter_m: float,
    rs_m: float,
    segments: int = 100,
) -> tuple[float, float, float]:
    n = max(2, int(segments) + 1)
    s_old = np.linspace(0.0, 1.0, int(r_m.size), dtype=float)
    s_new = np.linspace(0.0, 1.0, n, dtype=float)
    r_u = np.interp(s_new, s_old, np.asarray(r_m, dtype=float))
    th_u = np.interp(s_new, s_old, np.asarray(theta_rad, dtype=float))
    th_u = np.unwrap(th_u)

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
        b = max(1e-20, abs(float(impact_parameter_m)))

        # Coordinate-time increment from null geodesic relation:
        # dt/dphi = r^2 / (c * (1-rs/r) * b)
        dt = abs(dphi) * (rm * rm) / (C * g * b)

        spatial_dx2 = (dr * dr) / g + (rm * rm) * (dphi * dphi)
        ds2 = -g * ((C * dt) ** 2) + spatial_dx2

        sum_ds2 += ds2
        sum_dx2 += spatial_dx2

    ratio = abs(sum_ds2) / max(sum_dx2, 1e-30)
    return sum_ds2, sum_dx2, ratio


def _family_style(fam: int) -> tuple[str, float, float, float]:
    # family 0: smallest observer angle, family 1: largest observer angle
    if fam == 0:
        return "#1f77b4", 1.8, 0.95, 36.0  # blue
    if fam == 1:
        return "#d62728", 1.8, 0.95, 36.0  # red
    greys = ["#666666", "#7f7f7f", "#979797", "#adadad", "#c3c3c3"]
    return greys[(fam - 2) % len(greys)], 0.8, 0.75, 20.0


def _plot_b_sweep_two_family_dataset(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> None:
    rs = float(np.asarray(data["rs_m"], dtype=float))
    b_phi = np.asarray(data["b_phi_rad"], dtype=float)
    b_points = np.asarray(data["b_points_m"], dtype=float)
    a_point = np.asarray(data["a_point_m"], dtype=float)
    cand_ok = np.asarray(data["cand_ok"], dtype=bool)
    cand_dir_b = np.asarray(data["cand_dir_at_b_local_xy"], dtype=float)
    cand_dir_a = np.asarray(data["cand_dir_at_a_local_xy"], dtype=float)
    cand_dt = np.asarray(data["cand_travel_time_s"], dtype=float)
    cand_direction = np.asarray(data["cand_direction"], dtype=int)
    cand_is_turning = np.asarray(data["cand_is_turning"], dtype=bool)
    cand_target_az = np.asarray(data["cand_target_azimuth_rad"], dtype=float)
    cand_impact = np.asarray(data["cand_impact_parameter_m"], dtype=float)
    if "observer_dir_convention" not in data or str(np.asarray(data["observer_dir_convention"]).item()) != "coming_from":
        raise ValueError("Expected observer_dir_convention='coming_from'.")

    b_deg = np.mod(np.degrees(b_phi), 360.0)
    b_order = np.argsort(b_deg)
    n_b, n_fam = cand_ok.shape
    if n_fam != 2:
        raise ValueError("B-sweep dataset expected exactly 2 families.")

    obs_all = np.mod(np.degrees(np.arctan2(cand_dir_b[..., 1], cand_dir_b[..., 0])), 360.0)
    src_all = np.mod(np.degrees(np.arctan2(cand_dir_a[..., 1], cand_dir_a[..., 0])), 360.0)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    ax0, ax1, ax2 = axes
    marker_cycle = ["o", "s"]
    for b_i in b_order:
        for fam in range(n_fam):
            if not bool(cand_ok[b_i, fam]):
                continue
            color, _, alpha, ms = _family_style(fam)
            marker = marker_cycle[fam % len(marker_cycle)]
            ax0.scatter([b_deg[b_i]], [obs_all[b_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)
            ax1.scatter([b_deg[b_i]], [src_all[b_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)
            ax2.scatter([b_deg[b_i]], [cand_dt[b_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)

    for fam in range(n_fam):
        idx = [b_i for b_i in b_order if bool(cand_ok[b_i, fam])]
        if len(idx) < 2:
            continue
        xs = np.asarray([b_deg[b_i] for b_i in idx], dtype=float)
        y_obs = np.asarray([obs_all[b_i, fam] for b_i in idx], dtype=float)
        y_src = np.asarray([src_all[b_i, fam] for b_i in idx], dtype=float)
        y_dt = np.asarray([cand_dt[b_i, fam] for b_i in idx], dtype=float)
        color, lw, alpha, _ = _family_style(fam)
        ax0.plot(xs, y_obs, "-", color=color, lw=lw, alpha=alpha)
        ax1.plot(xs, y_src, "-", color=color, lw=lw, alpha=alpha)
        ax2.plot(xs, y_dt, "-", color=color, lw=lw, alpha=alpha)

    legend_handles = [
        Line2D([0], [0], color=_family_style(0)[0], lw=_family_style(0)[1], label="Family 0 (min obs angle)"),
        Line2D([0], [0], color=_family_style(1)[0], lw=_family_style(1)[1], label="Family 1 (max obs angle / X-mirror)"),
    ]
    ax0.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax0.set_ylabel("Observer angle (deg)")
    ax1.set_ylabel("Source angle (deg)")
    ax2.set_ylabel("Travel time (s)")
    ax2.set_xlabel("Observer azimuth B (deg)")
    ax0.set_ylim(0.0, 360.0)
    ax1.set_ylim(0.0, 360.0)
    ax2.set_xlim(0.0, 360.0)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("B-sweep two-family sky candidates (family 0 continuation, family 1 X-mirror)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")

    if args.paths_output is None:
        paths_output = args.output.with_name(f"{args.output.stem}_paths{args.output.suffix}")
    else:
        paths_output = Path(args.paths_output)
    paths_output.parent.mkdir(parents=True, exist_ok=True)

    bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs) / C).with_quality("fast")
    fig2, axp = plt.subplots(1, 1, figsize=(8.0, 8.0))
    t = np.linspace(0.0, 2.0 * pi, 241)
    axp.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.2)
    axp.scatter([0.0], [0.0], c="k", s=10)
    axp.scatter([a_point[0]], [a_point[1]], c="0.5", s=24, alpha=0.9)
    axp.scatter(b_points[:, 0], b_points[:, 1], s=8, c="0.6", alpha=0.35)

    diagnostics_rows: list[tuple[int, int, int, bool, float, float, float]] = []
    for b_i in range(n_b):
        for fam in range(n_fam):
            if not bool(cand_ok[b_i, fam]):
                continue
            if not np.isfinite(cand_target_az[b_i, fam]) or not np.isfinite(cand_impact[b_i, fam]):
                continue
            color, lw, alpha, _ = _family_style(fam)
            try:
                x_path, y_path, r_path, th_path = _build_curve_xy_from_candidate(
                    a_point=a_point,
                    b_point=b_points[b_i],
                    impact_parameter_m=float(cand_impact[b_i, fam]),
                    target_azimuth_rad=float(cand_target_az[b_i, fam]),
                    is_turning=bool(cand_is_turning[b_i, fam]),
                    bh=bh,
                    path_samples=int(args.path_samples),
                )
            except Exception:
                continue
            if fam == 1:
                x_path = np.asarray(x_path, dtype=float)
                y_path = -np.asarray(y_path, dtype=float)
            axp.plot(x_path, y_path, color=color, lw=lw, alpha=alpha, solid_joinstyle="round")
            sum_ds2, sum_dx2, ratio = _interval_diagnostics_from_profile(
                r_m=r_path,
                theta_rad=th_path,
                impact_parameter_m=float(cand_impact[b_i, fam]),
                rs_m=rs,
                segments=100,
            )
            diagnostics_rows.append(
                (int(b_i), int(fam), int(cand_direction[b_i, fam]), bool(cand_is_turning[b_i, fam]), sum_ds2, sum_dx2, ratio)
            )

    r_lim = 1.10 * max(float(np.max(np.linalg.norm(b_points[:, :2], axis=1))), float(np.linalg.norm(a_point[:2])))
    axp.set_xlim(-r_lim, r_lim)
    axp.set_ylim(-r_lim, r_lim)
    axp.set_aspect("equal", adjustable="box")
    axp.grid(alpha=0.22)
    axp.set_title("B-sweep geodesic paths, colored by family", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(paths_output, dpi=170)
    print(f"Saved figure: {paths_output}")
    print("Null-interval diagnostics (100 segments per solution):")
    print(
        f"{'B_idx':>5} {'fam':>4} {'dir':>4} {'turn':>5} "
        f"{'sum_ds2_m2':>16} {'sum_dx2_m2':>16} {'|ds2|/dx2':>12}"
    )
    for b_i, fam, direction, is_turn, sum_ds2, sum_dx2, ratio in diagnostics_rows:
        print(
            f"{b_i:5d} {fam:4d} {direction:4d} {('yes' if is_turn else 'no'):>5} "
            f"{sum_ds2:16.6e} {sum_dx2:16.6e} {ratio:12.3e}"
        )

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


def _plot_fixed_b_two_family_dataset(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> None:
    rs = float(np.asarray(data["rs_m"], dtype=float))
    a_phi = np.asarray(data["a_phi_rad"], dtype=float)
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_point = np.asarray(data["b_point_m"], dtype=float)
    cand_ok = np.asarray(data["cand_ok"], dtype=bool)
    cand_dir_b = np.asarray(data["cand_dir_at_b_local_xy"], dtype=float)
    cand_dir_a = np.asarray(data["cand_dir_at_a_local_xy"], dtype=float)
    cand_dt = np.asarray(data["cand_travel_time_s"], dtype=float)
    cand_direction = np.asarray(data["cand_direction"], dtype=int)
    cand_is_turning = np.asarray(data["cand_is_turning"], dtype=bool)
    cand_target_az = np.asarray(data["cand_target_azimuth_rad"], dtype=float)
    cand_impact = np.asarray(data["cand_impact_parameter_m"], dtype=float)
    if "observer_dir_convention" not in data or str(np.asarray(data["observer_dir_convention"]).item()) != "coming_from":
        raise ValueError("Expected observer_dir_convention='coming_from'.")

    n_a, n_fam = cand_ok.shape
    if n_fam != 2:
        raise ValueError("Fixed-B two-family dataset expected exactly 2 families.")
    a_deg = np.mod(np.degrees(a_phi), 360.0)
    a_order = np.argsort(a_deg)
    obs_all = np.mod(np.degrees(np.arctan2(cand_dir_b[..., 1], cand_dir_b[..., 0])), 360.0)
    src_all = np.mod(np.degrees(np.arctan2(cand_dir_a[..., 1], cand_dir_a[..., 0])), 360.0)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    ax0, ax1, ax2 = axes
    marker_cycle = ["o", "s"]
    for a_i in a_order:
        for fam in range(n_fam):
            if not bool(cand_ok[a_i, fam]):
                continue
            color, _, alpha, ms = _family_style(fam)
            marker = marker_cycle[fam % len(marker_cycle)]
            ax0.scatter([a_deg[a_i]], [obs_all[a_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)
            ax1.scatter([a_deg[a_i]], [src_all[a_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)
            ax2.scatter([a_deg[a_i]], [cand_dt[a_i, fam]], marker=marker, c=color, alpha=alpha, s=ms)

    for fam in range(n_fam):
        idx = [a_i for a_i in a_order if bool(cand_ok[a_i, fam])]
        if len(idx) < 2:
            continue
        xs = np.asarray([a_deg[a_i] for a_i in idx], dtype=float)
        y_obs = np.asarray([obs_all[a_i, fam] for a_i in idx], dtype=float)
        y_src = np.asarray([src_all[a_i, fam] for a_i in idx], dtype=float)
        y_dt = np.asarray([cand_dt[a_i, fam] for a_i in idx], dtype=float)
        color, lw, alpha, _ = _family_style(fam)
        ax0.plot(xs, y_obs, "-", color=color, lw=lw, alpha=alpha)
        ax1.plot(xs, y_src, "-", color=color, lw=lw, alpha=alpha)
        ax2.plot(xs, y_dt, "-", color=color, lw=lw, alpha=alpha)

    legend_handles = [
        Line2D([0], [0], color=_family_style(0)[0], lw=_family_style(0)[1], label="Family 0 (continuation)"),
        Line2D([0], [0], color=_family_style(1)[0], lw=_family_style(1)[1], label="Family 1 (X-mirror)"),
    ]
    ax0.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax0.set_ylabel("Observer angle (deg)")
    ax1.set_ylabel("Source angle (deg)")
    ax2.set_ylabel("Travel time (s)")
    ax2.set_xlabel("Source azimuth A (deg)")
    ax0.set_ylim(0.0, 360.0)
    ax1.set_ylim(0.0, 360.0)
    ax2.set_xlim(0.0, 360.0)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Fixed-B two-family sky candidates (family 0 continuation, family 1 X-mirror)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")

    if args.paths_output is None:
        paths_output = args.output.with_name(f"{args.output.stem}_paths{args.output.suffix}")
    else:
        paths_output = Path(args.paths_output)
    paths_output.parent.mkdir(parents=True, exist_ok=True)

    bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs) / C).with_quality("fast")
    fig2, axp = plt.subplots(1, 1, figsize=(8.0, 8.0))
    t = np.linspace(0.0, 2.0 * pi, 241)
    axp.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.2)
    axp.scatter([0.0], [0.0], c="k", s=10)
    axp.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.6", alpha=0.35)
    axp.scatter([b_point[0]], [b_point[1]], c="limegreen", s=28)

    diagnostics_rows: list[tuple[int, int, int, bool, float, float, float]] = []
    for a_i in range(n_a):
        for fam in range(n_fam):
            if not bool(cand_ok[a_i, fam]):
                continue
            if not np.isfinite(cand_target_az[a_i, fam]) or not np.isfinite(cand_impact[a_i, fam]):
                continue
            color, lw, alpha, _ = _family_style(fam)
            try:
                x_path, y_path, r_path, th_path = _build_curve_xy_from_candidate(
                    a_point=a_points[a_i],
                    b_point=b_point,
                    impact_parameter_m=float(cand_impact[a_i, fam]),
                    target_azimuth_rad=float(cand_target_az[a_i, fam]),
                    is_turning=bool(cand_is_turning[a_i, fam]),
                    bh=bh,
                    path_samples=int(args.path_samples),
                )
            except Exception:
                continue
            if fam == 1:
                x_path = np.asarray(x_path, dtype=float)
                y_path = -np.asarray(y_path, dtype=float)
            axp.plot(x_path, y_path, color=color, lw=lw, alpha=alpha, solid_joinstyle="round")
            sum_ds2, sum_dx2, ratio = _interval_diagnostics_from_profile(
                r_m=r_path,
                theta_rad=th_path,
                impact_parameter_m=float(cand_impact[a_i, fam]),
                rs_m=rs,
                segments=100,
            )
            diagnostics_rows.append(
                (int(a_i), int(fam), int(cand_direction[a_i, fam]), bool(cand_is_turning[a_i, fam]), sum_ds2, sum_dx2, ratio)
            )

    ring_r = float(np.max(np.linalg.norm(a_points[:, :2], axis=1))) if a_points.size else 1.0
    lim = 1.10 * ring_r
    axp.set_xlim(-lim, lim)
    axp.set_ylim(-lim, lim)
    axp.set_aspect("equal", adjustable="box")
    axp.grid(alpha=0.22)
    axp.set_title("Fixed-B geodesic paths, colored by family", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(paths_output, dpi=170)
    print(f"Saved figure: {paths_output}")
    print("Null-interval diagnostics (100 segments per solution):")
    print(
        f"{'A_idx':>5} {'fam':>4} {'dir':>4} {'turn':>5} "
        f"{'sum_ds2_m2':>16} {'sum_dx2_m2':>16} {'|ds2|/dx2':>12}"
    )
    for a_i, fam, direction, is_turn, sum_ds2, sum_dx2, ratio in diagnostics_rows:
        print(
            f"{a_i:5d} {fam:4d} {direction:4d} {('yes' if is_turn else 'no'):>5} "
            f"{sum_ds2:16.6e} {sum_dx2:16.6e} {ratio:12.3e}"
        )

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    args = _parse_args()
    data = np.load(args.input, allow_pickle=False)
    if (
        "family_definition" in data
        and str(np.asarray(data["family_definition"]).item()) == "family0_continuation_family1_x_mirror"
        and "a_phi_rad" in data
        and "a_points_m" in data
        and "b_point_m" in data
    ):
        _plot_fixed_b_two_family_dataset(args=args, data=data)
        return
    if "b_phi_rad" in data and "a_point_m" in data and "b_points_m" in data:
        _plot_b_sweep_two_family_dataset(args=args, data=data)
        return
    a_phi = np.asarray(data["a_phi_rad"], dtype=float)
    cand_ok = np.asarray(data["cand_ok"], dtype=bool)
    cand_dir_b = np.asarray(data["cand_dir_at_b_local_xy"], dtype=float)
    cand_dir_a = np.asarray(data["cand_dir_at_a_local_xy"], dtype=float)
    cand_dt = np.asarray(data["cand_travel_time_s"], dtype=float)
    cand_direction = np.asarray(data["cand_direction"], dtype=int)
    cand_is_turning = np.asarray(data["cand_is_turning"], dtype=bool)
    cand_target_az = np.asarray(data["cand_target_azimuth_rad"], dtype=float)
    cand_impact = np.asarray(data["cand_impact_parameter_m"], dtype=float)
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_point = np.asarray(data["b_point_m"], dtype=float)
    rs = float(np.asarray(data["rs_m"], dtype=float))
    if "observer_dir_convention" not in data:
        raise ValueError("Missing observer_dir_convention. Recompute saver table with current script.")
    obs_conv = str(np.asarray(data["observer_dir_convention"]).item())
    if obs_conv != "coming_from":
        raise ValueError("Expected observer_dir_convention='coming_from'. Recompute saver table with current script.")
    cand_dir_b_from = np.asarray(cand_dir_b, dtype=float)

    n_a, n_cand = cand_ok.shape
    phi_deg_360 = np.mod(np.degrees(a_phi), 360.0)
    a_order = np.argsort(phi_deg_360)
    ring_r = float(np.max(np.linalg.norm(a_points[:, :2], axis=1))) if a_points.size else 1.0
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    # Precompute candidate curves/signatures once.
    bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs) / C).with_quality("fast")
    curve_xy: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for a_i in range(n_a):
        for c_i in range(n_cand):
            if not bool(cand_ok[a_i, c_i]):
                continue
            if not np.isfinite(cand_target_az[a_i, c_i]) or not np.isfinite(cand_impact[a_i, c_i]):
                continue
            try:
                x_path, y_path, _, _ = _build_curve_xy_from_candidate(
                    a_point=a_points[a_i],
                    b_point=b_point,
                    impact_parameter_m=float(cand_impact[a_i, c_i]),
                    target_azimuth_rad=float(cand_target_az[a_i, c_i]),
                    is_turning=bool(cand_is_turning[a_i, c_i]),
                    bh=bh,
                    path_samples=int(args.path_samples),
                )
            except Exception:
                continue
            curve_xy[(a_i, c_i)] = (x_path, y_path)

    # Family assignment by observer-angle rank in [0, 360) at each A step.
    n_fam = int(max(np.sum(cand_ok, axis=1).max(initial=0), 1))
    cand_family = np.full((n_a, n_cand), -1, dtype=int)
    for a_i in a_order:
        cur = [c for c in range(n_cand) if (a_i, c) in curve_xy and bool(cand_ok[a_i, c])]
        if not cur:
            continue
        cur_sorted = sorted(
            cur,
            key=lambda c: float(np.mod(np.degrees(np.arctan2(cand_dir_b_from[a_i, c, 1], cand_dir_b_from[a_i, c, 0])), 360.0)),
        )
        m = len(cur_sorted)
        if m == 1:
            cand_family[a_i, cur_sorted[0]] = 0
            continue
        # Pin family 0 to minimum observer angle and family 1 to maximum observer angle.
        cand_family[a_i, cur_sorted[0]] = 0
        cand_family[a_i, cur_sorted[-1]] = 1
        # Fill middle families in ascending observer-angle order.
        middle_family_ids = list(range(2, n_fam))
        middle_candidates = cur_sorted[1:-1]
        for fam, c_i in zip(middle_family_ids, middle_candidates):
            cand_family[a_i, c_i] = fam

    phi_deg = phi_deg_360
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    ax0, ax1, ax2 = axes
    obs_all = np.mod(np.degrees(np.arctan2(cand_dir_b_from[..., 1], cand_dir_b_from[..., 0])), 360.0)
    src_all = np.mod(np.degrees(np.arctan2(cand_dir_a[..., 1], cand_dir_a[..., 0])), 360.0)

    # Scatter by candidate index marker, color by matched family.
    for a_i in a_order:
        for c_i in range(n_cand):
            fam = int(cand_family[a_i, c_i])
            if fam < 0 or not bool(cand_ok[a_i, c_i]):
                continue
            color, _, alpha, ms = _family_style(fam)
            marker = marker_cycle[c_i % len(marker_cycle)]
            ax0.scatter([phi_deg[a_i]], [obs_all[a_i, c_i]], marker=marker, c=color, alpha=alpha, s=ms)
            ax1.scatter([phi_deg[a_i]], [src_all[a_i, c_i]], marker=marker, c=color, alpha=alpha, s=ms)
            ax2.scatter([phi_deg[a_i]], [cand_dt[a_i, c_i]], marker=marker, c=color, alpha=alpha, s=ms)

    # Connect points by matched family across A.
    for fam in range(n_fam):
        idx = [
            (a_i, int(np.where(cand_family[a_i] == fam)[0][0]))
            for a_i in a_order
            if np.any(cand_family[a_i] == fam)
        ]
        if len(idx) < 2:
            continue
        xs = np.asarray([phi_deg[a_i] for a_i, _ in idx], dtype=float)
        y_obs = np.asarray([obs_all[a_i, c_i] for a_i, c_i in idx], dtype=float)
        y_src = np.asarray([src_all[a_i, c_i] for a_i, c_i in idx], dtype=float)
        y_dt = np.asarray([cand_dt[a_i, c_i] for a_i, c_i in idx], dtype=float)
        color, lw, alpha, _ = _family_style(fam)
        ax0.plot(xs, y_obs, "-", color=color, lw=lw, alpha=alpha)
        ax1.plot(xs, y_src, "-", color=color, lw=lw, alpha=alpha)
        ax2.plot(xs, y_dt, "-", color=color, lw=lw, alpha=alpha)

    legend_handles: list[Line2D] = []
    for fam in range(n_fam):
        if not np.any(cand_family == fam):
            continue
        color, lw, _, _ = _family_style(fam)
        legend_handles.append(
            Line2D([0], [0], color=color, lw=lw, label=f"Family {fam}")
        )
    if legend_handles:
        ax0.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax0.set_ylabel("Observer angle (deg)")
    ax1.set_ylabel("Source angle (deg)")
    ax2.set_ylabel("Travel time (s)")
    ax2.set_xlabel("Source azimuth (deg)")
    ax0.set_ylim(0.0, 360.0)
    ax1.set_ylim(0.0, 360.0)
    ax2.set_xlim(0.0, 360.0)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle(
        "Raw candidates grouped by observer-angle rank (family 0=min, family 1=max) at B=10 rs",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")

    if args.paths_output is None:
        paths_output = args.output.with_name(f"{args.output.stem}_paths{args.output.suffix}")
    else:
        paths_output = Path(args.paths_output)
    paths_output.parent.mkdir(parents=True, exist_ok=True)

    fig2, axp = plt.subplots(1, 1, figsize=(8.0, 8.0))
    t = np.linspace(0.0, 2.0 * pi, 241)
    axp.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.2)
    axp.scatter([0.0], [0.0], c="k", s=10)
    axp.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.6", alpha=0.4)
    axp.scatter([b_point[0]], [b_point[1]], c="limegreen", s=28)

    diagnostics_rows: list[tuple[int, int, int, bool, float, float, float]] = []
    for a_i in range(cand_ok.shape[0]):
        for c_i in range(n_cand):
            if not bool(cand_ok[a_i, c_i]):
                continue
            if not np.isfinite(cand_target_az[a_i, c_i]) or not np.isfinite(cand_impact[a_i, c_i]):
                continue
            fam = int(cand_family[a_i, c_i])
            if fam < 0:
                continue
            color, lw, alpha, _ = _family_style(fam)
            if (a_i, c_i) not in curve_xy:
                continue
            x_path, y_path = curve_xy[(a_i, c_i)]
            axp.plot(x_path, y_path, color=color, lw=lw, alpha=alpha, solid_joinstyle="round")
            _, _, r_path, th_path = _build_curve_xy_from_candidate(
                a_point=a_points[a_i],
                b_point=b_point,
                impact_parameter_m=float(cand_impact[a_i, c_i]),
                target_azimuth_rad=float(cand_target_az[a_i, c_i]),
                is_turning=bool(cand_is_turning[a_i, c_i]),
                bh=bh,
                path_samples=int(args.path_samples),
            )
            sum_ds2, sum_dx2, ratio = _interval_diagnostics_from_profile(
                r_m=r_path,
                theta_rad=th_path,
                impact_parameter_m=float(cand_impact[a_i, c_i]),
                rs_m=rs,
                segments=100,
            )
            diagnostics_rows.append(
                (
                    int(a_i),
                    int(c_i),
                    int(cand_direction[a_i, c_i]),
                    bool(cand_is_turning[a_i, c_i]),
                    float(sum_ds2),
                    float(sum_dx2),
                    float(ratio),
                )
            )

    lim = 1.10 * ring_r
    axp.set_xlim(-lim, lim)
    axp.set_ylim(-lim, lim)
    axp.set_aspect("equal", adjustable="box")
    axp.grid(alpha=0.22)
    axp.set_title("All solved sky->observer paths, colored by family", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(paths_output, dpi=170)
    print(f"Saved figure: {paths_output}")
    print("Null-interval diagnostics (100 segments per solution):")
    print(
        f"{'A_idx':>5} {'cand':>5} {'dir':>4} {'turn':>5} "
        f"{'sum_ds2_m2':>16} {'sum_dx2_m2':>16} {'|ds2|/dx2':>12}"
    )
    for a_i, c_i, direction, is_turn, sum_ds2, sum_dx2, ratio in diagnostics_rows:
        print(
            f"{a_i:5d} {c_i:5d} {direction:4d} "
            f"{('yes' if is_turn else 'no'):>5} "
            f"{sum_ds2:16.6e} {sum_dx2:16.6e} {ratio:12.3e}"
        )

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
