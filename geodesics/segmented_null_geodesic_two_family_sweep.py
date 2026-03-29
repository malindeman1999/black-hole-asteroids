from __future__ import annotations

import argparse
import concurrent.futures
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


def _segment_metric_terms_rs(
    r_nodes_rs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Segment metric terms in normalized units where Rs=1.
    """
    r0 = np.asarray(r_nodes_rs[:-1], dtype=float)
    r1 = np.asarray(r_nodes_rs[1:], dtype=float)
    rm = 0.5 * (r0 + r1)
    g = 1.0 - 1.0 / rm
    if np.any(~np.isfinite(rm)) or np.any(rm <= 1.0) or np.any(g <= 0.0):
        return None
    dr = r1 - r0
    a_term = (dr * dr) / (g * g)
    b_term = (rm * rm) / g
    return rm, g, dr, a_term, b_term


def _travel_time_only_rs_from_dtheta(dtheta: np.ndarray, a_term: np.ndarray, b_term: np.ndarray) -> float:
    arg = a_term + b_term * (dtheta * dtheta)
    if np.any(~np.isfinite(arg)) or np.any(arg < 0.0):
        return float("inf")
    return float(np.sum(np.sqrt(arg), dtype=float))


def _travel_time_and_proper_time_rs(r_nodes_rs: np.ndarray, theta_nodes: np.ndarray) -> Tuple[float, float]:
    seg = _segment_metric_terms_rs(r_nodes_rs=r_nodes_rs)
    if seg is None:
        return float("inf"), float("inf")
    rm, g, dr, a_term, b_term = seg
    dtheta = np.asarray(theta_nodes[1:] - theta_nodes[:-1], dtype=float)
    arg = a_term + b_term * (dtheta * dtheta)
    if np.any(~np.isfinite(arg)) or np.any(arg < 0.0):
        return float("inf"), float("inf")
    sqrt_arg = np.sqrt(arg)
    dt_seg = sqrt_arg
    t_total = float(np.sum(dt_seg, dtype=float))
    ds2 = -(g * dt_seg * dt_seg) + (dr * dr) / g + (rm * rm * dtheta * dtheta)
    if np.any(~np.isfinite(ds2)):
        return float("inf"), float("inf")
    tau_seg = np.where(ds2 < 0.0, np.sqrt(np.maximum(-ds2, 0.0)), 0.0)
    tau_total = float(np.sum(tau_seg, dtype=float))
    return t_total, tau_total


def _path_min_radius(xy: np.ndarray) -> float:
    """
    Minimum radius from origin attained along a piecewise-linear XY path.
    Checks both nodes and segment interiors.
    """
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
        return float("inf")
    if pts.shape[0] == 1:
        return float(np.linalg.norm(pts[0]))
    p0 = pts[:-1]
    p1 = pts[1:]
    d = p1 - p0
    dd = np.sum(d * d, axis=1)
    dot = np.sum(p0 * d, axis=1)
    t = np.zeros_like(dd)
    nz = dd > 0.0
    t[nz] = -dot[nz] / dd[nz]
    t = np.clip(t, 0.0, 1.0)
    closest = p0 + d * t[:, None]
    r2_nodes = np.sum(pts * pts, axis=1)
    r2_seg = np.sum(closest * closest, axis=1)
    r2_min = float(min(float(np.min(r2_nodes)), float(np.min(r2_seg))))
    if r2_min < 0.0:
        return float("inf")
    return float(sqrt(r2_min))


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


def _is_iteration_limit_message(msg: str) -> bool:
    m = str(msg).upper()
    return ("ITERATION" in m and "REACHED LIMIT" in m) or ("TOTAL NO. OF ITERATIONS" in m)


def _build_path_r_nodes(a_radius_m: float, b_radius_m: float, n: int, spacing: str) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("node_count must be >= 2")
    a = float(a_radius_m)
    b = float(b_radius_m)
    if spacing == "linear":
        return np.linspace(a, b, int(n), dtype=float)
    if spacing == "r3":
        # Near-origin emphasis with softened high-r collapse.
        # Pure linear interpolation in 1/r^2 is too aggressive at large radii for
        # wide spans (e.g., 100Rs -> 1.6Rs), causing the second node to jump to
        # single-digit Rs and leaving no samples near mid-range (~50Rs).
        # Warping the interpolation parameter keeps r3-style concentration near
        # small radii but restores useful coverage across the full radial span.
        inv2_a = 1.0 / max(a * a, 1e-30)
        inv2_b = 1.0 / max(b * b, 1e-30)
        u = np.linspace(0.0, 1.0, int(n), dtype=float)
        u_warp = np.power(u, 2.4)
        inv2 = inv2_a + (inv2_b - inv2_a) * u_warp
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
    r_nodes_rs: np.ndarray,
    theta_a: float,
    theta_b: float,
    theta0_interior: np.ndarray,
    max_iter: int,
    optimizer: str,
    opt_ftol: float,
    opt_gtol: float,
) -> Tuple[np.ndarray, float, float, bool, int, str]:
    n_int = int(len(theta0_interior))
    seg = _segment_metric_terms_rs(r_nodes_rs=r_nodes_rs)
    if seg is None:
        return np.asarray(theta0_interior, dtype=float), float("inf"), float("inf"), False, 0, "invalid segment geometry"
    _rm, _g, _dr, a_term, b_term = seg
    sec_per_rs = max(float(rs_m) / C, 1e-30)

    if n_int <= 0:
        theta = _build_theta(theta_a, theta_b, np.zeros(0, dtype=float))
        t_rs, tau_rs = _travel_time_and_proper_time_rs(r_nodes_rs, theta)
        return (
            np.zeros(0, dtype=float),
            float(t_rs * sec_per_rs),
            float(tau_rs * sec_per_rs),
            np.isfinite(t_rs),
            0,
            "no interior nodes",
        )

    theta0 = np.asarray(theta0_interior, dtype=float).copy()
    # Make objective numerically scale-invariant to BH size.
    # t scales ~ linearly with rs_m when geometry is fixed in Rs units.
    t_scale_s = sec_per_rs
    smooth_weight_s = 1e-12
    smooth_weight_rs = smooth_weight_s / t_scale_s

    theta_full = np.empty(n_int + 2, dtype=float)
    grad_theta = np.empty_like(theta_full)

    def _fill_theta(x: np.ndarray) -> np.ndarray:
        theta_full[0] = float(theta_a)
        theta_full[-1] = float(theta_b)
        theta_full[1:-1] = np.asarray(x, dtype=float)
        return theta_full

    def objective(x: np.ndarray) -> float:
        th = _fill_theta(x)
        dtheta = th[1:] - th[:-1]
        t = _travel_time_only_rs_from_dtheta(dtheta=dtheta, a_term=a_term, b_term=b_term)
        if not np.isfinite(t):
            return 1e40
        d2 = th[2:] - 2.0 * th[1:-1] + th[:-2]
        return float(t + smooth_weight_rs * np.dot(d2, d2))

    def objective_grad(x: np.ndarray) -> np.ndarray:
        th = _fill_theta(x)
        dtheta = th[1:] - th[:-1]
        arg = a_term + b_term * (dtheta * dtheta)
        if np.any(~np.isfinite(arg)) or np.any(arg <= 0.0):
            return np.zeros(n_int, dtype=float)
        sqrt_arg = np.sqrt(arg)
        v = (b_term * dtheta) / sqrt_arg
        grad_theta.fill(0.0)
        grad_theta[:-1] -= v
        grad_theta[1:] += v
        d2 = th[2:] - 2.0 * th[1:-1] + th[:-2]
        grad_theta[:-2] += 2.0 * smooth_weight_rs * d2
        grad_theta[1:-1] -= 4.0 * smooth_weight_rs * d2
        grad_theta[2:] += 2.0 * smooth_weight_rs * d2
        return np.asarray(grad_theta[1:-1], dtype=float)

    if optimizer == "scipy" and minimize is not None:
        lo = min(theta_a, theta_b) - 4.0 * pi
        hi = max(theta_a, theta_b) + 4.0 * pi
        bounds = [(lo, hi)] * n_int
        g0 = objective_grad(theta0)
        if np.all(np.isfinite(g0)) and float(np.max(np.abs(g0))) < 5e-7:
            th0 = _build_theta(theta_a, theta_b, theta0)
            t0_rs, tau0_rs = _travel_time_and_proper_time_rs(r_nodes_rs, th0)
            return (
                theta0,
                float(t0_rs * sec_per_rs),
                float(tau0_rs * sec_per_rs),
                np.isfinite(t0_rs),
                0,
                "warm-start accepted",
            )
        res = minimize(
            objective,
            theta0,
            jac=objective_grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter), "ftol": float(opt_ftol), "gtol": float(opt_gtol)},
        )
        x = np.asarray(res.x, dtype=float)
        th = _build_theta(theta_a, theta_b, x)
        t_rs, tau_rs = _travel_time_and_proper_time_rs(r_nodes_rs, th)
        t_s = float(t_rs * sec_per_rs)
        tau_s = float(tau_rs * sec_per_rs)
        return x, t_s, tau_s, bool(res.success) and np.isfinite(t_rs), int(getattr(res, "nit", 0)), str(res.message)

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
    t_rs, tau_rs = _travel_time_and_proper_time_rs(r_nodes_rs, th)
    return x, float(t_rs * sec_per_rs), float(tau_rs * sec_per_rs), np.isfinite(t_rs), int(n_it), "coordinate-descent"


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
    photon_safe_plus: np.ndarray
    min_radius_plus_m: np.ndarray
    iters_plus: np.ndarray


def _solve_ring_worker(task: Tuple[int, Dict[str, object]]) -> Tuple[int, RingSolveResult]:
    bi, kwargs = task
    rr = solve_ring_for_b_radius(**kwargs)
    return int(bi), rr


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
    opt_ftol: float = 1e-9,
    opt_gtol: float = 1e-6,
    adaptive_iter: bool = True,
    fail_fast_series: bool = True,
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
    a_radius_rs = float(a_radius_m) / float(rs_m)
    b_radius_rs = float(b_radius_m) / float(rs_m)
    r_nodes_rs = _build_path_r_nodes(
        a_radius_m=float(a_radius_rs),
        b_radius_m=float(b_radius_rs),
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
    photon_safe_plus = np.zeros(a_phi.size, dtype=bool)
    min_radius_plus_m = np.full(a_phi.size, np.nan, dtype=float)
    iters = np.zeros(a_phi.size, dtype=np.int32)
    r_ph_rs = 1.5

    n_int = node_count - 2
    prev_theta_nodes = _build_theta(0.0, 0.0, np.zeros(n_int, dtype=float))
    prev_n_it = max(8, int(max_iter) // 4)
    solve_indices = list(range(a_phi.size))

    for i in solve_indices:
        phi_a = float(a_phi[i])
        theta_a = float(phi_a)
        theta_b = 0.0

        if i == 0:
            # Requested seed: A and B on +X axis, start all node angles at 0.
            theta0_interior = np.zeros(n_int, dtype=float)
        else:
            theta0_interior = _retarget_theta_warm_start(prev_theta_nodes, theta_a_new=theta_a, theta_b_new=theta_b)

        max_iter_local = int(max_iter)
        if adaptive_iter and i > 0:
            max_iter_local = min(int(max_iter), max(18, int(1.5 * prev_n_it) + 8))

        x_opt, t_s, tau_s, ok, n_it, opt_msg = _optimize_thetas(
            rs_m=rs_m,
            r_nodes_rs=r_nodes_rs,
            theta_a=theta_a,
            theta_b=theta_b,
            theta0_interior=theta0_interior,
            max_iter=max_iter_local,
            optimizer=optimizer,
            opt_ftol=opt_ftol,
            opt_gtol=opt_gtol,
        )
        if (not bool(ok)) and (optimizer == "scipy") and _is_iteration_limit_message(str(opt_msg)):
            retry_interior = np.asarray(x_opt, dtype=float)
            retry_budget = int(max_iter_local)
            for _retry in range(2):
                retry_budget = min(max(50, 2 * retry_budget), max(50, 8 * int(max_iter)))
                x_opt2, t_s2, tau_s2, ok2, n_it2, opt_msg2 = _optimize_thetas(
                    rs_m=rs_m,
                    r_nodes_rs=r_nodes_rs,
                    theta_a=theta_a,
                    theta_b=theta_b,
                    theta0_interior=retry_interior,
                    max_iter=retry_budget,
                    optimizer=optimizer,
                    opt_ftol=opt_ftol,
                    opt_gtol=opt_gtol,
                )
                x_opt, t_s, tau_s, ok, n_it, opt_msg = x_opt2, t_s2, tau_s2, ok2, n_it2, opt_msg2
                retry_interior = np.asarray(x_opt, dtype=float)
                if bool(ok):
                    print(
                        "[retry] "
                        f"B={b_radius_rs:.6f} Rs | idx={i} | phi_a={theta_a:.6f} rad | "
                        f"resolved iteration cap with max_iter={retry_budget}.",
                        flush=True,
                    )
                    break
        theta_nodes = _build_theta(theta_a, theta_b, x_opt)
        xy_rs = _polar_to_xy(r_nodes_rs, theta_nodes)
        xy = xy_rs * float(rs_m)

        path_plus[i, :, :] = xy
        t_plus[i] = t_s
        tau_plus[i] = tau_s
        min_r_rs = _path_min_radius(xy_rs)
        min_radius_plus_m[i] = float(min_r_rs * float(rs_m))
        photon_safe = bool(np.isfinite(min_r_rs) and (min_r_rs >= r_ph_rs))
        photon_safe_plus[i] = photon_safe
        ok_plus[i] = bool(ok) and photon_safe
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

        current_ok = bool(ok) and bool(photon_safe)
        prev_theta_nodes = theta_nodes.copy()
        prev_n_it = max(1, int(n_it))

        if fail_fast_series and (not current_ok):
            if not bool(ok):
                reason = f"optimizer failed ({str(opt_msg)})"
            elif not bool(photon_safe):
                reason = (
                    "photon-sphere violation "
                    f"(min_r={min_r_rs:.6f} Rs < 1.500000 Rs)"
                )
            else:
                reason = "unknown failure"
            print(
                "[fail-fast] "
                f"B={b_radius_rs:.6f} Rs | idx={i} | phi_a={theta_a:.6f} rad | {reason}. "
                "Stopping remaining paths in this ring.",
                flush=True,
            )
            break

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
                photon_safe_plus=photon_safe_plus,
                min_radius_plus_m=min_radius_plus_m,
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
        photon_safe_plus=photon_safe_plus,
        min_radius_plus_m=min_radius_plus_m,
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
    opt_ftol: float,
    opt_gtol: float,
    adaptive_iter: bool,
    fail_fast_series: bool,
    workers: int,
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
        opt_ftol=opt_ftol,
        opt_gtol=opt_gtol,
        adaptive_iter=adaptive_iter,
        fail_fast_series=fail_fast_series,
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
    photon_safe_plus = np.zeros((n_b, n_a), dtype=bool)
    min_radius_plus_m = np.full((n_b, n_a), np.nan, dtype=float)
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
        photon_safe_plus[bi] = rr.photon_safe_plus
        min_radius_plus_m[bi] = rr.min_radius_plus_m
        iters_plus[bi] = rr.iters_plus

    _assign(0, ring0)

    n_workers = int(workers)
    if n_workers <= 0:
        n_workers = min(max(1, (os.cpu_count() or 1) - 1), int(max(1, n_b - 1)))

    if (n_b <= 1) or debug_show_rings or (n_workers <= 1):
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
                opt_ftol=opt_ftol,
                opt_gtol=opt_gtol,
                adaptive_iter=adaptive_iter,
                fail_fast_series=fail_fast_series,
                debug_show_paths=debug_show_rings,
                debug_pause=debug_pause_rings,
            )
            _assign(bi, rr)
    else:
        tasks: List[Tuple[int, Dict[str, object]]] = []
        for bi in range(1, n_b):
            tasks.append(
                (
                    int(bi),
                    {
                        "rs_m": float(rs_m),
                        "a_radius_m": float(a_radius_m),
                        "b_radius_m": float(b_vals[bi]),
                        "a_phi_count": int(a_phi_count),
                        "a_phi_second_rad": a_phi_second_rad,
                        "a_phi_step_rad": a_phi_step_rad,
                        "node_count": int(node_count),
                        "node_spacing": str(node_spacing),
                        "optimizer": str(optimizer),
                        "max_iter": int(max_iter),
                        "opt_ftol": float(opt_ftol),
                        "opt_gtol": float(opt_gtol),
                        "adaptive_iter": bool(adaptive_iter),
                        "fail_fast_series": bool(fail_fast_series),
                        "debug_show_paths": False,
                        "debug_pause": False,
                    },
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            for bi, rr in ex.map(_solve_ring_worker, tasks):
                _assign(int(bi), rr)

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
        "photon_safe_plus": photon_safe_plus,
        "min_radius_plus_m": min_radius_plus_m,
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
    p.add_argument(
        "--opt-ftol",
        type=float,
        default=1e-9,
        help="L-BFGS-B objective tolerance (default: 1e-9).",
    )
    p.add_argument(
        "--opt-gtol",
        type=float,
        default=1e-6,
        help="L-BFGS-B projected-gradient tolerance (default: 1e-6).",
    )
    p.add_argument(
        "--adaptive-iter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt per-angle max iterations from previous continuation solve (default: on).",
    )
    p.add_argument(
        "--fail-fast-series",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop A-phi continuation immediately after first failed path in a ring (default: on).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="B-ring worker processes. 0=auto, 1=serial, >1=parallel (default: 0).",
    )
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
        opt_ftol=float(args.opt_ftol),
        opt_gtol=float(args.opt_gtol),
        adaptive_iter=bool(args.adaptive_iter),
        fail_fast_series=bool(args.fail_fast_series),
        workers=int(args.workers),
        debug_show_rings=bool(args.debug_show_rings),
        debug_pause_rings=bool(args.debug_pause_rings),
    )
    out["bh_mass_kg"] = np.asarray(bh.mass_kg, dtype=float)
    out["bh_diameter_light_seconds"] = np.asarray(bh.diameter_light_seconds, dtype=float)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"Saved: {args.output}")
    n_photo_viol = int(np.count_nonzero(~np.asarray(out["photon_safe_plus"], dtype=bool)))
    print(
        "Solve summary: "
        f"B_rings={out['b_radii_m'].size}, A_phi={out['a_phi_rad'].size}, "
        f"nodes={int(args.node_count)}, success={int(np.count_nonzero(out['success_plus']))}/{out['success_plus'].size}, "
        f"photon_violations={n_photo_viol}"
    )


if __name__ == "__main__":
    main()
