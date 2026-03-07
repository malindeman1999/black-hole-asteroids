from __future__ import annotations

import argparse
from math import asin, pi, sqrt
from pathlib import Path
import random
import sys
import time
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import SchwarzschildBlackHole, cp
from precompute_earliest_grid import PrecomputedEarliestInterpolator


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _gamma_at_radius(rs: float, r: float, impact_parameter_m: float) -> float:
    s = impact_parameter_m * sqrt(1.0 - rs / r) / r
    return asin(_clamp(s, -1.0, 1.0))


def _path_by_direction(paths, direction: int):
    for p in paths:
        if p.direction == direction:
            return p
    return None


def _make_point(r: float, theta: float, z_scale: float) -> Tuple[float, float, float]:
    from math import cos, sin

    return (r * cos(theta), r * sin(theta), z_scale * r)


def _build_point_pairs(
    n: int, rs: float, rng: random.Random, rmin_rs: float, rmax_rs: float
) -> Tuple[np.ndarray, np.ndarray]:
    a_list: List[Tuple[float, float, float]] = []
    b_list: List[Tuple[float, float, float]] = []

    # Generate mirrored pairs to enforce an explicitly symmetric sample set.
    n_base = n // 2
    for _ in range(n_base):
        ra = rs * rng.uniform(rmin_rs, rmax_rs)
        rb = rs * rng.uniform(rmin_rs, rmax_rs)
        ta = rng.uniform(0.0, 2.0 * pi)
        tb = rng.uniform(0.0, 2.0 * pi)
        za = rng.uniform(-0.18, 0.18)
        zb = rng.uniform(-0.18, 0.18)

        a0 = _make_point(ra, ta, za)
        b0 = _make_point(rb, tb, zb)
        a1 = _make_point(ra, -ta, -za)
        b1 = _make_point(rb, -tb, -zb)

        a_list.append(a0)
        b_list.append(b0)
        a_list.append(a1)
        b_list.append(b1)

    if len(a_list) < n:
        # Odd-N fallback: add one extra unbiased random sample.
        ra = rs * rng.uniform(rmin_rs, rmax_rs)
        rb = rs * rng.uniform(rmin_rs, rmax_rs)
        ta = rng.uniform(0.0, 2.0 * pi)
        tb = rng.uniform(0.0, 2.0 * pi)
        za = rng.uniform(-0.18, 0.18)
        zb = rng.uniform(-0.18, 0.18)
        a_list.append(_make_point(ra, ta, za))
        b_list.append(_make_point(rb, tb, zb))

    a_list = a_list[:n]
    b_list = b_list[:n]
    return np.asarray(a_list, dtype=float), np.asarray(b_list, dtype=float)


def _resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    fallbacks = [
        Path("earliest_angles_precompute_10rs.npz"),
        Path("tests") / "earliest_angles_precompute_10rs.npz",
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Input file not found: {path}")
    print(f"Input not found at {path}; using fallback: {found}")
    return found


def _scatter_with_diag(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    unit: str,
    color: str = "tab:blue",
    lim: Tuple[float, float] | None = None,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    ax.scatter(xv, yv, s=8, alpha=0.45, color=color, edgecolors="none")
    if lim is not None:
        lo, hi = lim
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    elif xv.size > 0:
        lo = float(min(np.min(xv), np.min(yv)))
        hi = float(max(np.max(xv), np.max(yv)))
        pad = 0.05 * max(1e-12, hi - lo)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.grid(alpha=0.22)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Direct ({unit})")
    ax.set_ylabel(f"Interpolated ({unit})")


def _scatter_two_branches(
    ax,
    x_plus: np.ndarray,
    y_plus: np.ndarray,
    x_minus: np.ndarray,
    y_minus: np.ndarray,
    title: str,
    unit: str,
) -> None:
    mp = np.isfinite(x_plus) & np.isfinite(y_plus)
    mm = np.isfinite(x_minus) & np.isfinite(y_minus)
    xp = x_plus[mp]
    yp = y_plus[mp]
    xm = x_minus[mm]
    ym = y_minus[mm]

    if xp.size > 0:
        ax.scatter(xp, yp, s=8, alpha=0.5, color="red", edgecolors="none", label="+ branch")
    if xm.size > 0:
        ax.scatter(xm, ym, s=8, alpha=0.5, color="blue", edgecolors="none", label="- branch")

    all_x = np.concatenate([xp, xm]) if (xp.size + xm.size) > 0 else np.asarray([], dtype=float)
    all_y = np.concatenate([yp, ym]) if (yp.size + ym.size) > 0 else np.asarray([], dtype=float)
    if all_x.size > 0:
        lo = float(min(np.min(all_x), np.min(all_y)))
        hi = float(max(np.max(all_x), np.max(all_y)))
        pad = 0.05 * max(1e-12, hi - lo)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.grid(alpha=0.22)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Direct ({unit})")
    ax.set_ylabel(f"Interpolated ({unit})")
    ax.legend(loc="best", fontsize=8)


def _branch_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    unit: str,
    color: str,
    lim: Tuple[float, float] | None = None,
) -> None:
    _scatter_with_diag(ax, x=x, y=y, title=title, unit=unit, color=color, lim=lim)


def _joint_limits(
    x_minus: np.ndarray, y_minus: np.ndarray, x_plus: np.ndarray, y_plus: np.ndarray
) -> Tuple[float, float] | None:
    vals = []
    for arr in (x_minus, y_minus, x_plus, y_plus):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a)
    if not vals:
        return None
    allv = np.concatenate(vals)
    lo = float(np.min(allv))
    hi = float(np.max(allv))
    pad = 0.05 * max(1e-12, hi - lo)
    return lo - pad, hi + pad


def _percent_diff(interp_vals: np.ndarray, direct_vals: np.ndarray) -> np.ndarray:
    dv = np.asarray(direct_vals, dtype=float)
    iv = np.asarray(interp_vals, dtype=float)
    out = np.full(dv.shape, np.nan, dtype=float)
    mask = np.isfinite(dv) & np.isfinite(iv)
    if np.any(mask):
        denom = np.maximum(np.abs(dv[mask]), 1e-12)
        out[mask] = 100.0 * (iv[mask] - dv[mask]) / denom
    return out


def _scatter_pct_vs_delay(
    ax,
    delay_s: np.ndarray,
    pct_diff: np.ndarray,
    title: str,
    color: str,
) -> None:
    x = np.asarray(delay_s, dtype=float)
    y = np.asarray(pct_diff, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=8, alpha=0.45, color=color, edgecolors="none")
    ax.axhline(0.0, color="k", linestyle="--", lw=1.0, alpha=0.8)
    ax.grid(alpha=0.22)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Percent diff (%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct geodesic solve vs precomputed interpolation on random (A,B) pairs "
            "and plot stacked direct-vs-interpolated scatters for lag and +/- angles."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz table.",
    )
    parser.add_argument("--n-pairs", type=int, default=1000, help="Number of random (A,B) pairs (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--rmin-rs",
        type=float,
        default=5.0,
        help="Minimum random radius in rs units for both A and B (default: 5.0).",
    )
    parser.add_argument(
        "--rmax-rs",
        type=float,
        default=9.0,
        help="Maximum random radius in rs units for both A and B (default: 9.0).",
    )
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "fast"],
        default="fast",
        help="Direct solver precision preset (default: fast).",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU where available (default: true).",
    )
    parser.add_argument(
        "--interp-batch-size",
        type=int,
        default=5000,
        help="Interpolation batch size (default: 5000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures") / "direct_vs_interpolated_scatter.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output-pctdiff",
        type=Path,
        default=None,
        help=(
            "Optional output path for percent-difference-vs-delay figure. "
            "If omitted, uses <output_stem>_pctdiff_vs_delay.png in the same folder."
        ),
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show figure interactively when backend supports it (default: false).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    interp = PrecomputedEarliestInterpolator.from_npz(input_path)

    rs = interp.rs_m
    rng = random.Random(args.seed)
    n = int(args.n_pairs)
    a_points, b_points = _build_point_pairs(n, rs, rng, float(args.rmin_rs), float(args.rmax_rs))

    use_gpu = bool(args.use_gpu)
    mode = "GPU" if (use_gpu and cp is not None) else "CPU"
    print(f"Mode: {mode} | n_pairs={n} | direct_quality={args.quality}")

    # Direct batch solve.
    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    pairs = [(a_points[i], b_points[i]) for i in range(n)]
    t0 = time.perf_counter()
    results = bh.find_two_shortest_geodesics_batch(pairs, a_before_b=True, use_gpu=use_gpu)
    t_direct = time.perf_counter() - t0
    print(f"Direct batch solve time: {t_direct:.3f} s")

    direct_dt_plus = np.full(n, np.nan, dtype=float)
    direct_dt_minus = np.full(n, np.nan, dtype=float)
    direct_gamma_plus = np.full(n, np.nan, dtype=float)
    direct_gamma_minus = np.full(n, np.nan, dtype=float)
    direct_gamma_a_plus = np.full(n, np.nan, dtype=float)
    direct_gamma_a_minus = np.full(n, np.nan, dtype=float)
    direct_ok_plus = np.zeros(n, dtype=bool)
    direct_ok_minus = np.zeros(n, dtype=bool)

    for i, rr in enumerate(results):
        r_b = float(np.linalg.norm(b_points[i]))
        r_a = float(np.linalg.norm(a_points[i]))
        p_plus = _path_by_direction(rr.paths, +1)
        p_minus = _path_by_direction(rr.paths, -1)
        if p_plus is not None:
            direct_ok_plus[i] = True
            direct_dt_plus[i] = float(p_plus.travel_time_s)
            direct_gamma_plus[i] = _gamma_at_radius(rs, r_b, float(p_plus.impact_parameter_m))
            direct_gamma_a_plus[i] = _gamma_at_radius(rs, r_a, float(p_plus.impact_parameter_m))
        if p_minus is not None:
            direct_ok_minus[i] = True
            direct_dt_minus[i] = float(p_minus.travel_time_s)
            direct_gamma_minus[i] = _gamma_at_radius(rs, r_b, float(p_minus.impact_parameter_m))
            direct_gamma_a_minus[i] = _gamma_at_radius(rs, r_a, float(p_minus.impact_parameter_m))

    direct_ok_both = direct_ok_plus & direct_ok_minus
    direct_lag_abs = np.abs(direct_dt_minus - direct_dt_plus)
    direct_lag_abs[~direct_ok_both] = np.nan

    # Interpolated (3D reduction + back-rotation).
    t1 = time.perf_counter()
    out = interp.interpolate_pairs_3d(
        a_points_m=a_points,
        b_points_m=b_points,
        use_gpu=use_gpu,
        batch_size=int(args.interp_batch_size),
    )
    t_interp = time.perf_counter() - t1
    print(f"Interpolated batch time: {t_interp:.3f} s")

    interp_lag_abs = np.asarray(out["time_lag_abs_s"], dtype=float)
    interp_gamma_plus = np.asarray(out["gamma_at_b_plus_rad"], dtype=float)
    interp_gamma_minus = np.asarray(out["gamma_at_b_minus_rad"], dtype=float)
    interp_gamma_a_plus = np.asarray(out["gamma_at_a_plus_rad"], dtype=float)
    interp_gamma_a_minus = np.asarray(out["gamma_at_a_minus_rad"], dtype=float)
    interp_ok_plus = np.asarray(out["ok_plus"], dtype=bool)
    interp_ok_minus = np.asarray(out["ok_minus"], dtype=bool)
    interp_ok_both = np.asarray(out["ok_both"], dtype=bool)

    plus_valid = direct_ok_plus & interp_ok_plus
    minus_valid = direct_ok_minus & interp_ok_minus
    print(f"Pairs with + branch valid in direct+interp: {int(np.sum(plus_valid))}/{n}")
    print(f"Pairs with - branch valid in direct+interp: {int(np.sum(minus_valid))}/{n}")

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 13.0))
    lim_delay = _joint_limits(
        direct_dt_minus[minus_valid],
        np.asarray(out["delta_t_minus_s"], dtype=float)[minus_valid],
        direct_dt_plus[plus_valid],
        np.asarray(out["delta_t_plus_s"], dtype=float)[plus_valid],
    )
    lim_gb = _joint_limits(
        direct_gamma_minus[minus_valid],
        interp_gamma_minus[minus_valid],
        direct_gamma_plus[plus_valid],
        interp_gamma_plus[plus_valid],
    )
    lim_ga = _joint_limits(
        direct_gamma_a_minus[minus_valid],
        interp_gamma_a_minus[minus_valid],
        direct_gamma_a_plus[plus_valid],
        interp_gamma_a_plus[plus_valid],
    )

    # Left column: - branch
    _branch_scatter(
        axes[0, 0],
        direct_dt_minus[minus_valid],
        np.asarray(out["delta_t_minus_s"], dtype=float)[minus_valid],
        title="Delay (- branch): direct vs interpolated",
        unit="s",
        color="blue",
        lim=lim_delay,
    )
    _branch_scatter(
        axes[1, 0],
        direct_gamma_minus[minus_valid],
        interp_gamma_minus[minus_valid],
        title="Gamma at B (- branch): direct vs interpolated",
        unit="rad",
        color="blue",
        lim=lim_gb,
    )
    _branch_scatter(
        axes[2, 0],
        direct_gamma_a_minus[minus_valid],
        interp_gamma_a_minus[minus_valid],
        title="Gamma at A (- branch): direct vs interpolated",
        unit="rad",
        color="blue",
        lim=lim_ga,
    )

    # Right column: + branch
    _branch_scatter(
        axes[0, 1],
        direct_dt_plus[plus_valid],
        np.asarray(out["delta_t_plus_s"], dtype=float)[plus_valid],
        title="Delay (+ branch): direct vs interpolated",
        unit="s",
        color="red",
        lim=lim_delay,
    )
    _branch_scatter(
        axes[1, 1],
        direct_gamma_plus[plus_valid],
        interp_gamma_plus[plus_valid],
        title="Gamma at B (+ branch): direct vs interpolated",
        unit="rad",
        color="red",
        lim=lim_gb,
    )
    _branch_scatter(
        axes[2, 1],
        direct_gamma_a_plus[plus_valid],
        interp_gamma_a_plus[plus_valid],
        title="Gamma at A (+ branch): direct vs interpolated",
        unit="rad",
        color="red",
        lim=lim_ga,
    )

    fig.suptitle(
        "Direct geodesic vs precomputed interpolation comparison\n"
        f"N={n}, mode={mode}, direct_t={t_direct:.3f}s, interp_t={t_interp:.3f}s",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved figure: {args.output}")

    # Additional figure: percent difference vs delay (6 panels, branch-split).
    pct_dt_minus = _percent_diff(np.asarray(out["delta_t_minus_s"], dtype=float), direct_dt_minus)
    pct_dt_plus = _percent_diff(np.asarray(out["delta_t_plus_s"], dtype=float), direct_dt_plus)
    pct_gb_minus = _percent_diff(interp_gamma_minus, direct_gamma_minus)
    pct_gb_plus = _percent_diff(interp_gamma_plus, direct_gamma_plus)
    pct_ga_minus = _percent_diff(interp_gamma_a_minus, direct_gamma_a_minus)
    pct_ga_plus = _percent_diff(interp_gamma_a_plus, direct_gamma_a_plus)

    fig2, axes2 = plt.subplots(3, 2, figsize=(13.0, 13.0))
    _scatter_pct_vs_delay(
        axes2[0, 0],
        direct_dt_minus[minus_valid],
        pct_dt_minus[minus_valid],
        title="Delay error (- branch)",
        color="blue",
    )
    _scatter_pct_vs_delay(
        axes2[0, 1],
        direct_dt_plus[plus_valid],
        pct_dt_plus[plus_valid],
        title="Delay error (+ branch)",
        color="red",
    )
    _scatter_pct_vs_delay(
        axes2[1, 0],
        direct_dt_minus[minus_valid],
        pct_gb_minus[minus_valid],
        title="Gamma at B error (- branch)",
        color="blue",
    )
    _scatter_pct_vs_delay(
        axes2[1, 1],
        direct_dt_plus[plus_valid],
        pct_gb_plus[plus_valid],
        title="Gamma at B error (+ branch)",
        color="red",
    )
    _scatter_pct_vs_delay(
        axes2[2, 0],
        direct_dt_minus[minus_valid],
        pct_ga_minus[minus_valid],
        title="Gamma at A error (- branch)",
        color="blue",
    )
    _scatter_pct_vs_delay(
        axes2[2, 1],
        direct_dt_plus[plus_valid],
        pct_ga_plus[plus_valid],
        title="Gamma at A error (+ branch)",
        color="red",
    )
    fig2.suptitle(
        "Percent difference (interp vs direct) vs delay\n"
        f"N={n}, mode={mode}",
        fontsize=12,
    )
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_pct = args.output_pctdiff
    if output_pct is None:
        output_pct = args.output.with_name(f"{args.output.stem}_pctdiff_vs_delay.png")
    output_pct.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(output_pct, dpi=160)
    print(f"Saved figure: {output_pct}")

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
