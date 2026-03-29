from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


def _load_debug_pyplot():
    """
    Import matplotlib.pyplot with a conservative backend strategy on Windows.
    This avoids native Qt backend init crashes (e.g., STATUS_DLL_INIT_FAILED).
    """
    try:
        import matplotlib

        # Respect explicit backend choice from environment/CLI first.
        if os.environ.get("MPLBACKEND", "").strip() == "" and sys.platform.startswith("win"):
            matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        return plt, None
    except Exception as exc:
        return None, str(exc)

def _unit_xy(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(vv))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return vv / n


def _polar_to_xy(r: np.ndarray, th: np.ndarray) -> np.ndarray:
    rr = np.asarray(r, dtype=float)
    tt = np.asarray(th, dtype=float)
    return np.stack([rr * np.cos(tt), rr * np.sin(tt)], axis=1)


def _segment_dt_s(rs_m: float, r0: float, r1: float, th0: float, th1: float) -> float:
    rm = 0.5 * (float(r0) + float(r1))
    if rm <= rs_m:
        return float("inf")
    g = 1.0 - rs_m / rm
    if g <= 0.0:
        return float("inf")
    dr = float(r1 - r0)
    dth = float(th1 - th0)
    spatial_over_c2 = (dr * dr) / (g * g) + (rm * rm * dth * dth) / g
    if spatial_over_c2 < 0.0:
        return float("inf")
    return sqrt(spatial_over_c2) / C


def _segment_proper_time_s(rs_m: float, r0: float, r1: float, th0: float, th1: float, dt_s: float) -> float:
    rm = 0.5 * (float(r0) + float(r1))
    if rm <= rs_m:
        return float("inf")
    g = 1.0 - rs_m / rm
    if g <= 0.0:
        return float("inf")
    dr = float(r1 - r0)
    dth = float(th1 - th0)
    ds2 = -(g * C * C * float(dt_s) * float(dt_s)) + (dr * dr) / g + (rm * rm * dth * dth)
    # For null/spacelike interval we report zero proper-time accumulation.
    if ds2 >= 0.0:
        return 0.0
    return sqrt(-ds2) / C


def _travel_time_and_proper_time_s(rs_m: float, r_nodes: np.ndarray, theta_nodes: np.ndarray) -> Tuple[float, float]:
    total_t = 0.0
    total_tau = 0.0
    for i in range(len(r_nodes) - 1):
        dt = _segment_dt_s(rs_m, r_nodes[i], r_nodes[i + 1], theta_nodes[i], theta_nodes[i + 1])
        if not np.isfinite(dt):
            return float("inf"), float("inf")
        dt = float(dt)
        d_tau = _segment_proper_time_s(rs_m, r_nodes[i], r_nodes[i + 1], theta_nodes[i], theta_nodes[i + 1], dt)
        if not np.isfinite(d_tau):
            return float("inf"), float("inf")
        total_t += dt
        total_tau += float(d_tau)
    return total_t, total_tau


def _build_theta(theta_a: float, theta_b: float, theta_interior: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.asarray([theta_a], dtype=float),
        np.asarray(theta_interior, dtype=float),
        np.asarray([theta_b], dtype=float),
    ])


def _retarget_theta_warm_start(prev_theta_nodes: np.ndarray, theta_a_new: float, theta_b_new: float) -> np.ndarray:
    """
    Update previous theta profile to new endpoint angles.
    Taper endpoint corrections smoothly across interior nodes.
    """
    th_prev = np.asarray(prev_theta_nodes, dtype=float).reshape(-1)
    n = int(th_prev.size)
    if n < 2:
        return np.zeros(0, dtype=float)
    if n == 2:
        return np.zeros(0, dtype=float)
    delta_a = float(theta_a_new - th_prev[0])
    delta_b = float(theta_b_new - th_prev[-1])
    s = np.linspace(0.0, 1.0, n, dtype=float)
    th_new = th_prev + (1.0 - s) * delta_a + s * delta_b
    # Return interior only.
    return np.asarray(th_new[1:-1], dtype=float)


def _build_path_r_nodes(a_radius_m: float, b_radius_m: float, n: int, spacing: str) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("node_count must be >= 2")
    a = float(a_radius_m)
    b = float(b_radius_m)
    if spacing == "linear":
        return np.linspace(a, b, int(n), dtype=float)
    if spacing == "r3":
        # Match the sweep's near-origin emphasis: approximately constant d(1/r^2),
        # which yields local spacing law dr ~ r^3 (denser near small r).
        inv2_a = 1.0 / max(a * a, 1e-30)
        inv2_b = 1.0 / max(b * b, 1e-30)
        inv2 = np.linspace(inv2_a, inv2_b, int(n), dtype=float)
        inv2 = np.maximum(inv2, 1e-30)
        return 1.0 / np.sqrt(inv2)
    if spacing == "log":
        # Geometric spacing between endpoints (works for either radial direction).
        if a <= 0.0 or b <= 0.0:
            raise ValueError("log node spacing requires positive radii")
        return np.geomspace(a, b, int(n), dtype=float)
    raise ValueError("node spacing must be one of: linear, r3, log")


def _optimize_thetas(
    rs_m: float,
    r_nodes: np.ndarray,
    theta_a: float,
    theta_b: float,
    theta0_interior: np.ndarray,
    max_iter: int,
    optimizer: str,
) -> Tuple[np.ndarray, float, float, bool, int, str]:
    n_int = int(len(theta0_interior))
    if n_int <= 0:
        theta = _build_theta(theta_a, theta_b, np.zeros(0, dtype=float))
        t, tau = _travel_time_and_proper_time_s(rs_m, r_nodes, theta)
        return np.zeros(0, dtype=float), float(t), float(tau), np.isfinite(t), 0, "no interior nodes"

    theta0 = np.asarray(theta0_interior, dtype=float).copy()
    # Make objective numerically scale-invariant to BH size.
    # t scales ~ linearly with rs_m when geometry is fixed in Rs units.
    t_scale_s = max(float(rs_m) / C, 1e-30)
    smooth_weight_s = 1e-12

    def objective(x: np.ndarray) -> float:
        th = _build_theta(theta_a, theta_b, x)
        t, _ = _travel_time_and_proper_time_s(rs_m, r_nodes, th)
        if not np.isfinite(t):
            return 1e40
        # Small smoothness regularization keeps continuation stable.
        d2 = np.diff(th, n=2)
        return float((t + smooth_weight_s * np.dot(d2, d2)) / t_scale_s)

    if optimizer == "scipy" and minimize is not None:
        lo = min(theta_a, theta_b) - 4.0 * pi
        hi = max(theta_a, theta_b) + 4.0 * pi
        bounds = [(lo, hi)] * n_int
        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter), "ftol": 1e-16},
        )
        x = np.asarray(res.x, dtype=float)
        th = _build_theta(theta_a, theta_b, x)
        t, tau = _travel_time_and_proper_time_s(rs_m, r_nodes, th)
        return x, float(t), float(tau), bool(res.success) and np.isfinite(t), int(getattr(res, "nit", 0)), str(res.message)

    # Fallback: coordinate descent.
    x = theta0.copy()
    f = float(objective(x))
    step = 0.25
    n_it = 0
    for k in range(max(10, int(max_iter))):
        improved = False
        for j in range(n_int):
            base = x[j]
            best_val = f
            best_xj = base
            for cand in (base - step, base + step):
                x[j] = cand
                fv = float(objective(x))
                if fv < best_val:
                    best_val = fv
                    best_xj = cand
            x[j] = best_xj
            if best_val < f:
                f = best_val
                improved = True
        n_it = k + 1
        if not improved:
            step *= 0.55
            if step < 1e-6:
                break
    th = _build_theta(theta_a, theta_b, x)
    t, tau = _travel_time_and_proper_time_s(rs_m, r_nodes, th)
    return x, float(t), float(tau), np.isfinite(t), int(n_it), "coordinate-descent"


@dataclass
class RingSolveResult:
    b_radius_m: float
    a_phi_rad: np.ndarray
    path_plus_xy_m: np.ndarray
    path_minus_xy_m: np.ndarray
    time_plus_s: np.ndarray
    time_minus_s: np.ndarray
    proper_time_plus_s: np.ndarray
    proper_time_minus_s: np.ndarray
    emit_dir_plus_xy: np.ndarray
    emit_dir_minus_xy: np.ndarray
    arrive_from_plus_xy: np.ndarray
    arrive_from_minus_xy: np.ndarray
    success_plus: np.ndarray
    iters_plus: np.ndarray


def solve_ring_for_b_radius(
    rs_m: float,
    a_radius_m: float,
    b_radius_m: float,
    a_phi_count: int = 97,
    a_phi_second_rad: float | None = None,
    a_phi_step_rad: float | None = None,
    node_count: int = 20,
    node_spacing: str = "r3",
    optimizer: str = "scipy",
    max_iter: int = 250,
    debug_show_paths: bool = False,
    debug_pause: bool = False,
) -> RingSolveResult:
    if node_count < 2:
        raise ValueError("node_count must be >= 2")

    if a_phi_second_rad is not None:
        second = float(a_phi_second_rad)
        if not np.isfinite(second):
            raise ValueError("a_phi_second_rad must be finite when provided")
        a_phi = np.asarray([0.0, second], dtype=float)
    elif a_phi_step_rad is not None:
        count = int(a_phi_count)
        if count < 1:
            raise ValueError("a_phi_count must be >= 1")
        step = float(a_phi_step_rad)
        if not np.isfinite(step):
            raise ValueError("a_phi_step_rad must be finite when provided")
        a_phi = step * np.arange(count, dtype=float)
    else:
        a_phi = np.linspace(0.0, 2.0 * pi, int(a_phi_count), endpoint=False, dtype=float)
    r_nodes = _build_path_r_nodes(
        a_radius_m=float(a_radius_m),
        b_radius_m=float(b_radius_m),
        n=int(node_count),
        spacing=str(node_spacing),
    )

    path_plus = np.full((a_phi.size, node_count, 2), np.nan, dtype=float)
    path_minus = np.full_like(path_plus, np.nan)
    t_plus = np.full(a_phi.size, np.nan, dtype=float)
    t_minus = np.full(a_phi.size, np.nan, dtype=float)
    tau_plus = np.full(a_phi.size, np.nan, dtype=float)
    tau_minus = np.full(a_phi.size, np.nan, dtype=float)
    emit_plus = np.full((a_phi.size, 2), np.nan, dtype=float)
    emit_minus = np.full((a_phi.size, 2), np.nan, dtype=float)
    from_plus = np.full((a_phi.size, 2), np.nan, dtype=float)
    from_minus = np.full((a_phi.size, 2), np.nan, dtype=float)
    ok_plus = np.zeros(a_phi.size, dtype=bool)
    iters = np.zeros(a_phi.size, dtype=np.int32)

    n_int = node_count - 2
    prev_theta_nodes = _build_theta(0.0, 0.0, np.zeros(n_int, dtype=float))

    for i, phi_a in enumerate(a_phi):
        theta_a = float(phi_a)
        theta_b = 0.0

        if i == 0:
            # Requested seed: A and B on +X axis, start all node angles at 0.
            theta0_interior = np.zeros(n_int, dtype=float)
        else:
            theta0_interior = _retarget_theta_warm_start(prev_theta_nodes, theta_a_new=theta_a, theta_b_new=theta_b)

        x_opt, t_s, tau_s, ok, n_it, _ = _optimize_thetas(
            rs_m=rs_m,
            r_nodes=r_nodes,
            theta_a=theta_a,
            theta_b=theta_b,
            theta0_interior=theta0_interior,
            max_iter=max_iter,
            optimizer=optimizer,
        )
        theta_nodes = _build_theta(theta_a, theta_b, x_opt)
        xy = _polar_to_xy(r_nodes, theta_nodes)

        path_plus[i, :, :] = xy
        t_plus[i] = t_s
        tau_plus[i] = tau_s
        ok_plus[i] = bool(ok)
        iters[i] = int(n_it)

        emit = _unit_xy(xy[1, :] - xy[0, :])
        # "coming from" at B points from B back toward the incoming segment.
        come_from = _unit_xy(xy[-2, :] - xy[-1, :])
        emit_plus[i, :] = emit
        from_plus[i, :] = come_from

        # Requested second family: mirror about x-axis.
        xy_m = xy.copy()
        xy_m[:, 1] *= -1.0
        path_minus[i, :, :] = xy_m
        t_minus[i] = t_s
        tau_minus[i] = tau_s
        emit_minus[i, :] = _unit_xy(np.asarray([emit[0], -emit[1]], dtype=float))
        from_minus[i, :] = _unit_xy(np.asarray([come_from[0], -come_from[1]], dtype=float))

        prev_theta_nodes = theta_nodes.copy()

    if debug_show_paths:
        plt, plot_err = _load_debug_pyplot()
        if plt is None:
            print(f"[debug] Skipping ring plot: matplotlib backend init failed: {plot_err}")
            return RingSolveResult(
                b_radius_m=float(b_radius_m),
                a_phi_rad=a_phi,
                path_plus_xy_m=path_plus,
                path_minus_xy_m=path_minus,
                time_plus_s=t_plus,
                time_minus_s=t_minus,
                proper_time_plus_s=tau_plus,
                proper_time_minus_s=tau_minus,
                emit_dir_plus_xy=emit_plus,
                emit_dir_minus_xy=emit_minus,
                arrive_from_plus_xy=from_plus,
                arrive_from_minus_xy=from_minus,
                success_plus=ok_plus,
                iters_plus=iters,
            )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        tt = np.linspace(0.0, 2.0 * pi, 300)
        ax.plot(rs_m * np.cos(tt), rs_m * np.sin(tt), "k-", lw=1.1, label="horizon")
        r_ph = 1.5 * rs_m
        ax.plot(r_ph * np.cos(tt), r_ph * np.sin(tt), "k-.", lw=1.0, alpha=0.9, label="photon sphere")
        ax.plot(a_radius_m * np.cos(tt), a_radius_m * np.sin(tt), "k--", lw=0.9, alpha=0.5, label="A ring")
        ax.scatter([b_radius_m], [0.0], marker="x", c="green", s=70, label="B")

        for i in range(a_phi.size):
            ax.plot(path_plus[i, :, 0], path_plus[i, :, 1], color="tab:blue", alpha=0.45, lw=0.9)
            ax.plot(path_minus[i, :, 0], path_minus[i, :, 1], color="tab:red", alpha=0.35, lw=0.9)
            ax.scatter(path_plus[i, :, 0], path_plus[i, :, 1], color="tab:blue", s=10, alpha=0.7)
            ax.scatter(path_minus[i, :, 0], path_minus[i, :, 1], color="tab:red", s=10, alpha=0.6)

        r_lim = 1.15 * max(float(a_radius_m), float(b_radius_m))
        ax.set_xlim(-r_lim, r_lim)
        ax.set_ylim(-r_lim, r_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_title(f"Segmented null-path ring solve | B={b_radius_m/rs_m:.3f} Rs")
        ax.legend(loc="upper right")
        plt.show(block=bool(debug_pause))
        plt.close(fig)

    return RingSolveResult(
        b_radius_m=float(b_radius_m),
        a_phi_rad=a_phi,
        path_plus_xy_m=path_plus,
        path_minus_xy_m=path_minus,
        time_plus_s=t_plus,
        time_minus_s=t_minus,
        proper_time_plus_s=tau_plus,
        proper_time_minus_s=tau_minus,
        emit_dir_plus_xy=emit_plus,
        emit_dir_minus_xy=emit_minus,
        arrive_from_plus_xy=from_plus,
        arrive_from_minus_xy=from_minus,
        success_plus=ok_plus,
        iters_plus=iters,
    )


def solve_b_radial_series(
    rs_m: float,
    a_radius_m: float,
    b_radii_m: np.ndarray,
    a_phi_count: int,
    a_phi_second_rad: float | None,
    a_phi_step_rad: float | None,
    node_count: int,
    node_spacing: str,
    optimizer: str,
    max_iter: int,
    debug_show_rings: bool,
    debug_pause_rings: bool,
) -> Dict[str, np.ndarray]:
    b_vals = np.asarray(b_radii_m, dtype=float).reshape(-1)
    if b_vals.size < 1:
        raise ValueError("Need at least one B radius")

    ring0 = solve_ring_for_b_radius(
        rs_m=rs_m,
        a_radius_m=a_radius_m,
        b_radius_m=float(b_vals[0]),
        a_phi_count=a_phi_count,
        a_phi_second_rad=a_phi_second_rad,
        a_phi_step_rad=a_phi_step_rad,
        node_count=node_count,
        node_spacing=node_spacing,
        optimizer=optimizer,
        max_iter=max_iter,
        debug_show_paths=debug_show_rings,
        debug_pause=debug_pause_rings,
    )

    n_b = b_vals.size
    n_a = ring0.a_phi_rad.size
    n_nodes = node_count

    path_plus = np.full((n_b, n_a, n_nodes, 2), np.nan, dtype=float)
    path_minus = np.full_like(path_plus, np.nan)
    t_plus = np.full((n_b, n_a), np.nan, dtype=float)
    t_minus = np.full((n_b, n_a), np.nan, dtype=float)
    tau_plus = np.full((n_b, n_a), np.nan, dtype=float)
    tau_minus = np.full((n_b, n_a), np.nan, dtype=float)
    emit_plus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    emit_minus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    from_plus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    from_minus = np.full((n_b, n_a, 2), np.nan, dtype=float)
    ok_plus = np.zeros((n_b, n_a), dtype=bool)
    iters_plus = np.zeros((n_b, n_a), dtype=np.int32)

    def _assign(bi: int, rr: RingSolveResult) -> None:
        path_plus[bi] = rr.path_plus_xy_m
        path_minus[bi] = rr.path_minus_xy_m
        t_plus[bi] = rr.time_plus_s
        t_minus[bi] = rr.time_minus_s
        tau_plus[bi] = rr.proper_time_plus_s
        tau_minus[bi] = rr.proper_time_minus_s
        emit_plus[bi] = rr.emit_dir_plus_xy
        emit_minus[bi] = rr.emit_dir_minus_xy
        from_plus[bi] = rr.arrive_from_plus_xy
        from_minus[bi] = rr.arrive_from_minus_xy
        ok_plus[bi] = rr.success_plus
        iters_plus[bi] = rr.iters_plus

    _assign(0, ring0)

    for bi in range(1, n_b):
        rr = solve_ring_for_b_radius(
            rs_m=rs_m,
            a_radius_m=a_radius_m,
            b_radius_m=float(b_vals[bi]),
            a_phi_count=a_phi_count,
            a_phi_second_rad=a_phi_second_rad,
            a_phi_step_rad=a_phi_step_rad,
            node_count=node_count,
            node_spacing=node_spacing,
            optimizer=optimizer,
            max_iter=max_iter,
            debug_show_paths=debug_show_rings,
            debug_pause=debug_pause_rings,
        )
        _assign(bi, rr)

    return {
        "rs_m": np.asarray(rs_m, dtype=float),
        "a_radius_m": np.asarray(a_radius_m, dtype=float),
        "b_radii_m": b_vals,
        "a_phi_rad": ring0.a_phi_rad,
        "path_plus_xy_m": path_plus,
        "path_minus_xy_m": path_minus,
        "time_plus_s": t_plus,
        "time_minus_s": t_minus,
        "proper_time_plus_s": tau_plus,
        "proper_time_minus_s": tau_minus,
        "emit_dir_plus_xy": emit_plus,
        "emit_dir_minus_xy": emit_minus,
        "arrive_from_plus_xy": from_plus,
        "arrive_from_minus_xy": from_minus,
        "success_plus": ok_plus,
        "iters_plus": iters_plus,
    }


def _build_b_radii_rs(r_min_rs: float, r_max_rs: float, n: int, spacing: str) -> np.ndarray:
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if float(r_max_rs) <= float(r_min_rs):
        raise ValueError("r_max_rs must be > r_min_rs")
    if int(n) == 1:
        return np.asarray([float(r_min_rs)], dtype=float)

    if spacing == "linear":
        return np.linspace(float(r_min_rs), float(r_max_rs), int(n), dtype=float)

    if spacing == "r3":
        # Heuristic: enforce approximately constant |d(1/r^2)| per sample.
        # This corresponds to local spacing law Δr ~ r^3.
        inv2_min = 1.0 / (float(r_min_rs) * float(r_min_rs))
        inv2_max = 1.0 / (float(r_max_rs) * float(r_max_rs))
        inv2 = np.linspace(inv2_min, inv2_max, int(n), dtype=float)
        inv2 = np.maximum(inv2, 1e-30)
        return 1.0 / np.sqrt(inv2)

    raise ValueError("spacing must be one of: linear, r3")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segmented null-path geodesic solver with A-phi continuation and mirrored second family."
    )
    p.add_argument(
        "--rs-m",
        type=float,
        default=1.0,
        help=(
            "Schwarzschild radius in meters (legacy input). "
            "A SchwarzschildBlackHole object is still built from this value."
        ),
    )
    p.add_argument(
        "--diameter-light-seconds",
        type=float,
        default=None,
        help=(
            "Black-hole diameter in light-seconds (preferred project-wide input). "
            "If provided, overrides --rs-m."
        ),
    )
    p.add_argument("--a-radius-rs", type=float, default=100.0, help="A ring radius in Rs.")
    p.add_argument("--b-r-min-rs", type=float, default=1.6, help="Min B radius in Rs.")
    p.add_argument("--b-r-max-rs", type=float, default=10.0, help="Max B radius in Rs.")
    p.add_argument("--b-r-count", type=int, default=12, help="Number of B radii.")
    p.add_argument(
        "--b-spacing",
        choices=["r3", "linear"],
        default="r3",
        help="B radial spacing: r3 uses Δr~r^3 heuristic; linear uses uniform radius spacing.",
    )
    p.add_argument("--a-phi-count", type=int, default=97, help="A angles around ring.")
    p.add_argument(
        "--a-phi-second-rad",
        type=float,
        default=None,
        help=(
            "Optional explicit second A angle (radians) used for debug continuation. "
            "When set, solver uses exactly two A paths: 0 and this value."
        ),
    )
    p.add_argument(
        "--a-phi-step-rad",
        type=float,
        default=None,
        help=(
            "Optional A-angle step in radians. When set (and --a-phi-second-rad is not set), "
            "A angles are generated as i*step for i=0..a-phi-count-1."
        ),
    )
    p.add_argument("--node-count", type=int, default=20, help="Nodes along A->B path.")
    p.add_argument(
        "--node-spacing",
        choices=["r3", "linear", "log"],
        default="r3",
        help=(
            "Path node radial spacing: r3 uses near-origin-dense spacing (dr~r^3); "
            "linear uses uniform radius spacing; log uses geometric spacing."
        ),
    )
    p.add_argument("--optimizer", choices=["scipy", "coord"], default="scipy")
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--debug-show-rings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug-pause-rings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "segmented_null_geodesic_two_family_sweep.npz",
    )
    return p.parse_args()


def _build_black_hole(args: argparse.Namespace) -> SchwarzschildBlackHole:
    if args.diameter_light_seconds is not None:
        d_ls = float(args.diameter_light_seconds)
        if d_ls <= 0.0:
            raise ValueError("--diameter-light-seconds must be > 0")
        return SchwarzschildBlackHole.from_diameter_light_seconds(d_ls)

    rs_m = float(args.rs_m)
    if rs_m <= 0.0:
        raise ValueError("--rs-m must be > 0")
    return SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs_m) / C)


def main() -> None:
    args = _parse_args()

    bh = _build_black_hole(args)
    rs_m = float(bh.schwarzschild_radius_m)
    a_radius_m = float(args.a_radius_rs) * rs_m
    if int(args.b_r_count) < 1:
        raise ValueError("--b-r-count must be >= 1")
    if float(args.b_r_max_rs) <= float(args.b_r_min_rs):
        raise ValueError("--b-r-max-rs must be > --b-r-min-rs")

    b_radii_rs = _build_b_radii_rs(
        r_min_rs=float(args.b_r_min_rs),
        r_max_rs=float(args.b_r_max_rs),
        n=int(args.b_r_count),
        spacing=str(args.b_spacing),
    )
    b_radii_m = b_radii_rs * rs_m

    out = solve_b_radial_series(
        rs_m=rs_m,
        a_radius_m=a_radius_m,
        b_radii_m=b_radii_m,
        a_phi_count=int(args.a_phi_count),
        a_phi_second_rad=(None if args.a_phi_second_rad is None else float(args.a_phi_second_rad)),
        a_phi_step_rad=(None if args.a_phi_step_rad is None else float(args.a_phi_step_rad)),
        node_count=int(args.node_count),
        node_spacing=str(args.node_spacing),
        optimizer=str(args.optimizer),
        max_iter=int(args.max_iter),
        debug_show_rings=bool(args.debug_show_rings),
        debug_pause_rings=bool(args.debug_pause_rings),
    )
    out["bh_mass_kg"] = np.asarray(bh.mass_kg, dtype=float)
    out["bh_diameter_light_seconds"] = np.asarray(bh.diameter_light_seconds, dtype=float)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"Saved: {args.output}")
    print(
        "Solve summary: "
        f"B_rings={out['b_radii_m'].size}, A_phi={out['a_phi_rad'].size}, "
        f"nodes={int(args.node_count)}, success={int(np.count_nonzero(out['success_plus']))}/{out['success_plus'].size}"
    )


if __name__ == "__main__":
    main()
