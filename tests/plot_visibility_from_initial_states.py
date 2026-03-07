from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole
from earliest_visible_interpolated_session import SampledTrajectory3D
from initial_state_visibility import TimelikeVisibilitySession
from precompute_earliest_grid import PrecomputedEarliestInterpolator


def _build_initial_velocity(speed_m_s: float, direction_deg: float) -> Tuple[float, float, float]:
    th = direction_deg * pi / 180.0
    return (speed_m_s * cos(th), speed_m_s * sin(th), 0.0)


def _local_circular_speed_schwarzschild(r_m: float, rs_m: float) -> float:
    if r_m <= rs_m:
        raise ValueError("Circular orbit requires r > rs.")
    return C * (rs_m / (2.0 * (r_m - rs_m))) ** 0.5


def _resolve_precompute_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("earliest_angles_precompute_10rs.npz"),
        Path("tests") / "earliest_angles_precompute_10rs.npz",
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is not None:
        print(f"Input not found at {path}; using fallback: {found}")
    return found


def _find_first_bracket(ts: np.ndarray, fvals: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    for i in range(ts.size):
        fi = float(fvals[i])
        if np.isfinite(fi) and abs(fi) <= 1e-12:
            return None, float(ts[i])
    for i in range(1, ts.size):
        f0 = float(fvals[i - 1])
        f1 = float(fvals[i])
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 * f1 <= 0.0:
            return (float(ts[i - 1]), float(ts[i])), None
    return None, None


def _interp_eval_many(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    ts_by_case: List[np.ndarray],
    point_b: Tuple[float, float, float],
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> List[Dict[str, np.ndarray]]:
    if not sampled_trajectories:
        return []
    counts: List[int] = []
    a_chunks: List[np.ndarray] = []
    for sampled, ts in zip(sampled_trajectories, ts_by_case):
        t = np.asarray(ts, dtype=float).reshape(-1)
        counts.append(int(t.size))
        a_chunks.append(sampled.eval_points(t))
    total = int(sum(counts))
    if total <= 0:
        return []
    a_all = np.concatenate(a_chunks, axis=0)
    b = np.asarray(point_b, dtype=float).reshape(1, 3)
    b_all = np.repeat(b, total, axis=0)
    use_gpu_eval = bool(use_gpu and total >= int(max(1, gpu_min_batch)))
    out = interp.interpolate_pairs_3d(
        a_points_m=a_all,
        b_points_m=b_all,
        use_gpu=use_gpu_eval,
        batch_size=int(batch_size),
    )
    ok_plus_all = np.asarray(out["ok_plus"], dtype=bool)
    ok_minus_all = np.asarray(out["ok_minus"], dtype=bool)
    dt_plus_all = np.asarray(out["delta_t_plus_s"], dtype=float)
    dt_minus_all = np.asarray(out["delta_t_minus_s"], dtype=float)
    gb_plus_all = np.asarray(out["gamma_at_b_plus_rad"], dtype=float)
    gb_minus_all = np.asarray(out["gamma_at_b_minus_rad"], dtype=float)

    csum = np.cumsum(np.asarray([0] + counts, dtype=int))
    per_case: List[Dict[str, np.ndarray]] = []
    for i in range(len(counts)):
        lo = int(csum[i])
        hi = int(csum[i + 1])
        per_case.append(
            {
                "ok_plus": ok_plus_all[lo:hi],
                "ok_minus": ok_minus_all[lo:hi],
                "delta_t_plus_s": dt_plus_all[lo:hi],
                "delta_t_minus_s": dt_minus_all[lo:hi],
                "gamma_at_b_plus_rad": gb_plus_all[lo:hi],
                "gamma_at_b_minus_rad": gb_minus_all[lo:hi],
            }
        )
    return per_case


def _interp_eval_points(
    interp: PrecomputedEarliestInterpolator,
    a_points: np.ndarray,
    point_b: Tuple[float, float, float],
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> Dict[str, np.ndarray]:
    pts = np.asarray(a_points, dtype=float).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return {
            "ok_plus": np.asarray([], dtype=bool),
            "ok_minus": np.asarray([], dtype=bool),
            "delta_t_plus_s": np.asarray([], dtype=float),
            "delta_t_minus_s": np.asarray([], dtype=float),
            "gamma_at_b_plus_rad": np.asarray([], dtype=float),
            "gamma_at_b_minus_rad": np.asarray([], dtype=float),
        }
    b = np.asarray(point_b, dtype=float).reshape(1, 3)
    b_all = np.repeat(b, n, axis=0)
    use_gpu_eval = bool(use_gpu and n >= int(max(1, gpu_min_batch)))
    out = interp.interpolate_pairs_3d(
        a_points_m=pts,
        b_points_m=b_all,
        use_gpu=use_gpu_eval,
        batch_size=int(batch_size),
    )
    return {
        "ok_plus": np.asarray(out["ok_plus"], dtype=bool),
        "ok_minus": np.asarray(out["ok_minus"], dtype=bool),
        "delta_t_plus_s": np.asarray(out["delta_t_plus_s"], dtype=float),
        "delta_t_minus_s": np.asarray(out["delta_t_minus_s"], dtype=float),
        "gamma_at_b_plus_rad": np.asarray(out["gamma_at_b_plus_rad"], dtype=float),
        "gamma_at_b_minus_rad": np.asarray(out["gamma_at_b_minus_rad"], dtype=float),
        "arrival_dir_plus_xyz": np.asarray(out["arrival_dir_plus_xyz"], dtype=float),
        "arrival_dir_minus_xyz": np.asarray(out["arrival_dir_minus_xyz"], dtype=float),
    }


def _trajectory_velocity(sampled: SampledTrajectory3D, t: float) -> np.ndarray:
    ts = np.asarray(sampled.ts, dtype=float)
    if ts.size < 2:
        return np.zeros(3, dtype=float)

    # Cache per-trajectory velocity tables to avoid repeated finite-difference
    # point evaluations during linearized updates.
    if not hasattr(sampled, "_vel_cache"):
        vx = np.gradient(np.asarray(sampled.xs, dtype=float), ts, edge_order=1)
        vy = np.gradient(np.asarray(sampled.ys, dtype=float), ts, edge_order=1)
        vz = np.gradient(np.asarray(sampled.zs, dtype=float), ts, edge_order=1)
        sampled._vel_cache = (ts.copy(), vx, vy, vz)

    vts, vx, vy, vz = sampled._vel_cache
    t0 = float(np.clip(t, float(vts[0]), float(vts[-1])))
    return np.asarray(
        [
            float(np.interp(t0, vts, vx)),
            float(np.interp(t0, vts, vy)),
            float(np.interp(t0, vts, vz)),
        ],
        dtype=float,
    )


def _solve_interpolated_subset_for_t0(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    point_b: Tuple[float, float, float],
    t0: float,
    tmin: float,
    tmax: float,
    scan_samples: int,
    root_max_iter: int,
    root_tol_time: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
    case_indices: List[int],
    previous_batch: Optional[List[Dict[str, object]]] = None,
    previous_t0: Optional[float] = None,
) -> Dict[int, Dict[str, object]]:
    if not case_indices:
        return {}
    sub_traj = [sampled_trajectories[i] for i in case_indices]
    sub_prev = None
    if previous_batch is not None:
        sub_prev = [previous_batch[i] for i in case_indices if i < len(previous_batch)]
        if len(sub_prev) != len(case_indices):
            sub_prev = None
    sub_out = _solve_interpolated_batch_for_t0(
        interp=interp,
        sampled_trajectories=sub_traj,
        point_b=point_b,
        t0=float(t0),
        tmin=float(tmin),
        tmax=float(tmax),
        scan_samples=int(scan_samples),
        root_max_iter=int(root_max_iter),
        root_tol_time=float(root_tol_time),
        use_gpu=use_gpu,
        batch_size=int(batch_size),
        gpu_min_batch=int(gpu_min_batch),
        previous_batch=sub_prev,
        previous_t0=previous_t0,
    )
    return {idx: sub_out[k] for k, idx in enumerate(case_indices)}


def _solve_root_bisection_interpolated(
    interp: PrecomputedEarliestInterpolator,
    sampled: SampledTrajectory3D,
    side: str,
    bracket: Tuple[float, float],
    point_b: Tuple[float, float, float],
    t0: float,
    root_max_iter: int,
    root_tol_time: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> float:
    a, c = float(bracket[0]), float(bracket[1])
    for _ in range(max(1, int(root_max_iter))):
        m = 0.5 * (a + c)
        ts = np.asarray([a, m, c], dtype=float)
        ev = _interp_eval_many(
            interp=interp,
            sampled_trajectories=[sampled],
            ts_by_case=[ts],
            point_b=point_b,
            use_gpu=use_gpu,
            batch_size=batch_size,
            gpu_min_batch=gpu_min_batch,
        )[0]
        dt = ev["delta_t_plus_s"] if side == "plus" else ev["delta_t_minus_s"]
        f = ts + dt - float(t0)
        fa, fm, fc = float(f[0]), float(f[1]), float(f[2])
        if abs(fm) <= 1e-12 or abs(c - a) <= float(root_tol_time):
            return float(m)
        if np.isfinite(fa) and np.isfinite(fm) and (fa * fm <= 0.0):
            c = m
        elif np.isfinite(fm) and np.isfinite(fc):
            a = m
        else:
            return float(m)
    return float(0.5 * (a + c))


def _solve_root_bisection_interpolated_batch(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    side: str,
    bracket_by_case: List[Optional[Tuple[float, float]]],
    point_b: Tuple[float, float, float],
    t0: float,
    root_max_iter: int,
    root_tol_time: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> List[Optional[float]]:
    n = len(sampled_trajectories)
    te_out: List[Optional[float]] = [None] * n
    active = [i for i, b in enumerate(bracket_by_case) if b is not None]
    if not active:
        return te_out

    a_arr = np.asarray([float(bracket_by_case[i][0]) for i in active], dtype=float)
    c_arr = np.asarray([float(bracket_by_case[i][1]) for i in active], dtype=float)

    for _ in range(max(1, int(root_max_iter))):
        if not active:
            break
        m_arr = 0.5 * (a_arr + c_arr)
        ts_by_case = [np.asarray([a_arr[k], m_arr[k], c_arr[k]], dtype=float) for k in range(len(active))]
        sampled_active = [sampled_trajectories[i] for i in active]
        evals = _interp_eval_many(
            interp=interp,
            sampled_trajectories=sampled_active,
            ts_by_case=ts_by_case,
            point_b=point_b,
            use_gpu=use_gpu,
            batch_size=batch_size,
            gpu_min_batch=gpu_min_batch,
        )

        keep_active_idx: List[int] = []
        next_a: List[float] = []
        next_c: List[float] = []
        for k, case_idx in enumerate(active):
            dt = evals[k]["delta_t_plus_s"] if side == "plus" else evals[k]["delta_t_minus_s"]
            ts = ts_by_case[k]
            f = ts + dt - float(t0)
            fa, fm, fc = float(f[0]), float(f[1]), float(f[2])
            a = float(a_arr[k])
            c = float(c_arr[k])
            m = float(m_arr[k])

            if (not np.isfinite(fm)) or abs(fm) <= 1e-12 or abs(c - a) <= float(root_tol_time):
                te_out[case_idx] = m
                continue
            if np.isfinite(fa) and np.isfinite(fm) and (fa * fm <= 0.0):
                keep_active_idx.append(case_idx)
                next_a.append(a)
                next_c.append(m)
            elif np.isfinite(fm) and np.isfinite(fc):
                keep_active_idx.append(case_idx)
                next_a.append(m)
                next_c.append(c)
            else:
                te_out[case_idx] = m

        active = keep_active_idx
        if active:
            a_arr = np.asarray(next_a, dtype=float)
            c_arr = np.asarray(next_c, dtype=float)

    for k, case_idx in enumerate(active):
        te_out[case_idx] = float(0.5 * (a_arr[k] + c_arr[k]))
    return te_out


def _build_observed_from_interp_batch(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    requests: List[Tuple[int, str, float]],
    point_b: Tuple[float, float, float],
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> Dict[Tuple[int, str], Optional[Dict[str, float]]]:
    out: Dict[Tuple[int, str], Optional[Dict[str, float]]] = {}
    if not requests:
        return out

    points: List[np.ndarray] = []
    side_arr: List[str] = []
    case_arr: List[int] = []
    te_arr: List[float] = []
    for case_idx, _side, te in requests:
        pa = sampled_trajectories[int(case_idx)].eval_points(np.asarray([float(te)], dtype=float))[0]
        points.append(np.asarray(pa, dtype=float))
        case_arr.append(int(case_idx))
        side_arr.append(str(_side))
        te_arr.append(float(te))
    ev = _interp_eval_points(
        interp=interp,
        a_points=np.asarray(points, dtype=float),
        point_b=point_b,
        use_gpu=use_gpu,
        batch_size=batch_size,
        gpu_min_batch=gpu_min_batch,
    )
    for j, (case_idx, side, te) in enumerate(zip(case_arr, side_arr, te_arr)):
        if side == "plus":
            ok = bool(ev["ok_plus"][j])
            dt = float(ev["delta_t_plus_s"][j])
            gamma_b = float(ev["gamma_at_b_plus_rad"][j])
            arrival_dir = np.asarray(ev["arrival_dir_plus_xyz"][j], dtype=float)
            direction = +1
        else:
            ok = bool(ev["ok_minus"][j])
            dt = float(ev["delta_t_minus_s"][j])
            gamma_b = float(ev["gamma_at_b_minus_rad"][j])
            arrival_dir = np.asarray(ev["arrival_dir_minus_xyz"][j], dtype=float)
            direction = -1
        if (not ok) or (not np.isfinite(dt)):
            out[(case_idx, side)] = None
            continue
        pa = points[j]
        out[(case_idx, side)] = {
            "emission_time_s": float(te),
            "travel_time_s": float(dt),
            "arrival_time_s": float(te + dt),
            "gamma_at_b_rad": float(gamma_b),
            "point_a_m": (float(pa[0]), float(pa[1]), float(pa[2])),
            "arrival_dir_xyz": (
                float(arrival_dir[0]),
                float(arrival_dir[1]),
                float(arrival_dir[2]),
            ),
            "direction": int(direction),
        }
    return out


def _build_observed_from_interp_single(
    interp: PrecomputedEarliestInterpolator,
    sampled: SampledTrajectory3D,
    side: str,
    te: float,
    point_b: Tuple[float, float, float],
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
) -> Optional[Dict[str, float]]:
    observed = _build_observed_from_interp_batch(
        interp=interp,
        sampled_trajectories=[sampled],
        requests=[(0, side, float(te))],
        point_b=point_b,
        use_gpu=use_gpu,
        batch_size=batch_size,
        gpu_min_batch=gpu_min_batch,
    )
    return observed.get((0, side))


def _solve_interpolated_batch_for_t0(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    point_b: Tuple[float, float, float],
    t0: float,
    tmin: float,
    tmax: float,
    scan_samples: int,
    root_max_iter: int,
    root_tol_time: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
    previous_batch: Optional[List[Dict[str, object]]] = None,
    previous_t0: Optional[float] = None,
    seed_base_step_s: float = 0.1,
    seed_expand_steps: int = 8,
) -> List[Dict[str, object]]:
    """
    Batched precomputed-interpolation visibility solver for one observer time `t0`.

    This is the reference bracket/root solver used by batch timing and plotting.
    It solves all trajectories in `sampled_trajectories` together and returns a
    list of per-trajectory dicts with `plus`, `minus`, and `earliest` entries.

    Solver behavior:
    - Warm-start per branch from `previous_batch` and `previous_t0` when provided.
    - Expand a local bracket around each seed (batched across trajectories).
    - For unresolved branches, fallback to full scan over `[tmin, min(tmax, t0)]`.
    - Refine bracketed roots with bisection and build observed records.

    Options/controls:
    - `scan_samples`: fallback scan density.
    - `root_max_iter`, `root_tol_time`: root refinement controls.
    - `use_gpu`: enable GPU interpolation backend when available.
    - `batch_size`: interpolation chunk size.
    - `gpu_min_batch`: minimum query count to dispatch interpolation on GPU.
    - `seed_base_step_s`, `seed_expand_steps`: warm-start bracket expansion controls.
    """
    t_lo = float(tmin)
    t_hi = min(float(tmax), float(t0))
    out: List[Dict[str, object]] = []
    if t_hi < t_lo:
        for _ in sampled_trajectories:
            out.append({"plus": None, "minus": None, "earliest": None})
        return out

    n_cases = len(sampled_trajectories)
    point_b_arr = np.asarray(point_b, dtype=float)
    dt0 = float(t0 - previous_t0) if previous_t0 is not None else 0.0
    base_step = max(float(seed_base_step_s), abs(dt0), float(root_tol_time) * 8.0, 1e-6)
    scan_n = max(3, int(scan_samples))
    te_plus: List[Optional[float]] = [None] * n_cases
    te_minus: List[Optional[float]] = [None] * n_cases

    for side in ("plus", "minus"):
        seeds: List[float] = []
        for i, sampled in enumerate(sampled_trajectories):
            prev_te: Optional[float] = None
            if previous_batch is not None and i < len(previous_batch):
                prev_side = previous_batch[i].get(side) if isinstance(previous_batch[i], dict) else None
                if isinstance(prev_side, dict):
                    cand = prev_side.get("emission_time_s")
                    if cand is not None and np.isfinite(float(cand)):
                        prev_te = float(cand)
            if prev_te is not None:
                seed = prev_te + dt0
            else:
                t_seed = float(np.clip(t0, t_lo, t_hi))
                pa = sampled.eval_points(np.asarray([t_seed], dtype=float))[0]
                d = float(np.linalg.norm(point_b_arr - np.asarray(pa, dtype=float)))
                seed = float(t0) - d / C
            seeds.append(float(np.clip(seed, t_lo, t_hi)))

        brackets: List[Optional[Tuple[float, float]]] = [None] * n_cases
        exacts: List[Optional[float]] = [None] * n_cases
        step = float(base_step)
        for _ in range(max(1, int(seed_expand_steps))):
            unresolved = [i for i in range(n_cases) if brackets[i] is None and exacts[i] is None]
            if not unresolved:
                break
            ts_local: List[np.ndarray] = []
            sampled_local: List[SampledTrajectory3D] = []
            for i in unresolved:
                lo = max(t_lo, seeds[i] - step)
                hi = min(t_hi, seeds[i] + step)
                ts_local.append(np.asarray(sorted(set([lo, seeds[i], hi])), dtype=float))
                sampled_local.append(sampled_trajectories[i])
            local = _interp_eval_many(
                interp=interp,
                sampled_trajectories=sampled_local,
                ts_by_case=ts_local,
                point_b=point_b,
                use_gpu=use_gpu,
                batch_size=batch_size,
                gpu_min_batch=gpu_min_batch,
            )
            for k, i in enumerate(unresolved):
                ts = ts_local[k]
                dt = local[k]["delta_t_plus_s"] if side == "plus" else local[k]["delta_t_minus_s"]
                fvals = ts + dt - float(t0)
                bracket, exact = _find_first_bracket(ts, fvals)
                if exact is not None:
                    exacts[i] = float(exact)
                elif bracket is not None:
                    brackets[i] = (float(bracket[0]), float(bracket[1]))
            step *= 2.0

        unresolved = [i for i in range(n_cases) if brackets[i] is None and exacts[i] is None]
        if unresolved:
            ts_scan = np.linspace(t_lo, t_hi, scan_n, dtype=float)
            sampled_local = [sampled_trajectories[i] for i in unresolved]
            scan = _interp_eval_many(
                interp=interp,
                sampled_trajectories=sampled_local,
                ts_by_case=[ts_scan for _ in sampled_local],
                point_b=point_b,
                use_gpu=use_gpu,
                batch_size=batch_size,
                gpu_min_batch=gpu_min_batch,
            )
            for k, i in enumerate(unresolved):
                dt = scan[k]["delta_t_plus_s"] if side == "plus" else scan[k]["delta_t_minus_s"]
                fvals = ts_scan + dt - float(t0)
                bracket, exact = _find_first_bracket(ts_scan, fvals)
                if exact is not None:
                    exacts[i] = float(exact)
                elif bracket is not None:
                    brackets[i] = (float(bracket[0]), float(bracket[1]))

        te_from_brackets = _solve_root_bisection_interpolated_batch(
            interp=interp,
            sampled_trajectories=sampled_trajectories,
            side=side,
            bracket_by_case=brackets,
            point_b=point_b,
            t0=float(t0),
            root_max_iter=int(root_max_iter),
            root_tol_time=float(root_tol_time),
            use_gpu=use_gpu,
            batch_size=batch_size,
            gpu_min_batch=gpu_min_batch,
        )
        for i in range(n_cases):
            te = float(exacts[i]) if exacts[i] is not None else te_from_brackets[i]
            if side == "plus":
                te_plus[i] = te
            else:
                te_minus[i] = te

    requests: List[Tuple[int, str, float]] = []
    for i in range(n_cases):
        if te_plus[i] is not None:
            requests.append((i, "plus", float(te_plus[i])))
        if te_minus[i] is not None:
            requests.append((i, "minus", float(te_minus[i])))
    observed_by_key = _build_observed_from_interp_batch(
        interp=interp,
        sampled_trajectories=sampled_trajectories,
        requests=requests,
        point_b=point_b,
        use_gpu=use_gpu,
        batch_size=batch_size,
        gpu_min_batch=gpu_min_batch,
    )
    plus_rows: List[Optional[Dict[str, float]]] = [None] * n_cases
    minus_rows: List[Optional[Dict[str, float]]] = [None] * n_cases
    for i in range(n_cases):
        plus_rows[i] = observed_by_key.get((i, "plus")) if isinstance(observed_by_key.get((i, "plus")), dict) else None
        minus_rows[i] = observed_by_key.get((i, "minus")) if isinstance(observed_by_key.get((i, "minus")), dict) else None

    for i in range(n_cases):
        plus = plus_rows[i]
        minus = minus_rows[i]
        earliest = None
        if plus is not None and minus is not None:
            earliest = plus if float(plus["emission_time_s"]) <= float(minus["emission_time_s"]) else minus
        elif plus is not None:
            earliest = plus
        elif minus is not None:
            earliest = minus
        out.append({"plus": plus, "minus": minus, "earliest": earliest})
    return out


def _solve_interpolated_linearized_batch_for_t0(
    interp: PrecomputedEarliestInterpolator,
    sampled_trajectories: List[SampledTrajectory3D],
    point_b: Tuple[float, float, float],
    t0: float,
    tmin: float,
    tmax: float,
    scan_samples: int,
    root_max_iter: int,
    root_tol_time: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
    previous_batch: Optional[List[Dict[str, object]]] = None,
    previous_t0: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    Batched one-step linearized visibility update for one observer time `t0`.

    This mode assumes small time steps between successive observer queries and
    advances each branch emission time by linearization of:
        f(te, t0) = te + delta_t(A(te), B) - t0 = 0
    using:
        dte ~= dt0 / (1 + grad_A(delta_t) . v_A)
    where `v_A` is the source trajectory velocity at the previous emission time.

    Workflow:
    - If previous solution state is unavailable, fallback to
      `_solve_interpolated_batch_for_t0`.
    - Estimate spatial gradient `grad_A(delta_t)` by finite differences on the
      precomputed interpolator.
    - Compute one-step `te` update for plus/minus branches.
    - Validate updated branch via interpolated observation build.
    - Any unresolved branches fallback to `_solve_interpolated_batch_for_t0`.

    Options/controls:
    - `use_gpu`, `batch_size`, `gpu_min_batch`: interpolation backend controls.
    - `scan_samples`, `root_max_iter`, `root_tol_time`: fallback solver controls.
    - `previous_batch`, `previous_t0`: required for linearized warm-start updates.

    Example benchmark output (per-trajectory per-call timing):

    Timing batched circular-orbit visibility solves (precomputed interpolation)
    queries_per_batch=11 (initial + 10 subsequent), scan_samples=41, root_max_iter=12
    batch_n | interp_cpu_init | interp_cpu_sub | linear_cpu_init | linear_cpu_sub | interp_gpu_init | interp_gpu_sub | linear_gpu_init | linear_gpu_sub
    --------+-----------------+----------------+-----------------+----------------+-----------------+----------------+-----------------+----------------
          1 |         83.8947 |         34.098 |         50.1923 |          2.912 |            51.9139 |         32.357 |            44.8063 |          2.367
          5 |         11.5066 |          6.825 |          9.6930 |          0.672 |             9.7778 |          6.861 |             9.8165 |          0.642
         25 |          3.3229 |          2.035 |          2.9218 |          0.329 |             3.2049 |          2.679 |             3.9636 |          5.178
    """
    if previous_batch is None or previous_t0 is None:
        return _solve_interpolated_batch_for_t0(
            interp=interp,
            sampled_trajectories=sampled_trajectories,
            point_b=point_b,
            t0=float(t0),
            tmin=float(tmin),
            tmax=float(tmax),
            scan_samples=int(scan_samples),
            root_max_iter=int(root_max_iter),
            root_tol_time=float(root_tol_time),
            use_gpu=use_gpu,
            batch_size=int(batch_size),
            gpu_min_batch=int(gpu_min_batch),
            previous_batch=None,
            previous_t0=None,
        )

    t_lo = float(tmin)
    t_hi = min(float(tmax), float(t0))
    n_cases = len(sampled_trajectories)
    if t_hi < t_lo:
        return [{"plus": None, "minus": None, "earliest": None} for _ in sampled_trajectories]

    dt0 = float(t0 - previous_t0)
    rs = float(interp.rs_m)
    eps = max(1.0, 1e-5 * rs)

    out: List[Dict[str, object]] = [{"plus": None, "minus": None, "earliest": None} for _ in sampled_trajectories]
    unresolved: List[Tuple[int, str]] = []

    work_points: List[np.ndarray] = []
    work_meta: List[Tuple[int, str, float, np.ndarray]] = []
    for i, sampled in enumerate(sampled_trajectories):
        prev_row = previous_batch[i] if i < len(previous_batch) else None
        for side in ("plus", "minus"):
            prev_side = prev_row.get(side) if isinstance(prev_row, dict) else None
            if not isinstance(prev_side, dict):
                unresolved.append((i, side))
                continue
            te_prev_val = prev_side.get("emission_time_s")
            if te_prev_val is None or (not np.isfinite(float(te_prev_val))):
                unresolved.append((i, side))
                continue
            te_prev = float(te_prev_val)
            te_prev = float(np.clip(te_prev, t_lo, t_hi))
            a_prev = sampled.eval_points(np.asarray([te_prev], dtype=float))[0]
            v_prev = _trajectory_velocity(sampled, te_prev)
            work_meta.append((i, side, te_prev, v_prev))
            for ax in range(3):
                delta = np.zeros(3, dtype=float)
                delta[ax] = eps
                work_points.append(np.asarray(a_prev + delta, dtype=float))
                work_points.append(np.asarray(a_prev - delta, dtype=float))

    if work_points:
        grad_eval = _interp_eval_points(
            interp=interp,
            a_points=np.asarray(work_points, dtype=float),
            point_b=point_b,
            use_gpu=use_gpu,
            batch_size=batch_size,
            gpu_min_batch=gpu_min_batch,
        )
    else:
        grad_eval = _interp_eval_points(
            interp=interp,
            a_points=np.zeros((0, 3), dtype=float),
            point_b=point_b,
            use_gpu=use_gpu,
            batch_size=batch_size,
            gpu_min_batch=gpu_min_batch,
        )

    ptr = 0
    predicted_requests: List[Tuple[int, str, float]] = []
    for i, side, te_prev, v_prev in work_meta:
        ok_key = "ok_plus" if side == "plus" else "ok_minus"
        dt_key = "delta_t_plus_s" if side == "plus" else "delta_t_minus_s"
        grad = np.zeros(3, dtype=float)
        good = True
        for ax in range(3):
            i_p = ptr
            i_m = ptr + 1
            ptr += 2
            ok_p = bool(grad_eval[ok_key][i_p]) if i_p < grad_eval[ok_key].size else False
            ok_m = bool(grad_eval[ok_key][i_m]) if i_m < grad_eval[ok_key].size else False
            dt_p = float(grad_eval[dt_key][i_p]) if i_p < grad_eval[dt_key].size else float("nan")
            dt_m = float(grad_eval[dt_key][i_m]) if i_m < grad_eval[dt_key].size else float("nan")
            if (not ok_p) or (not ok_m) or (not np.isfinite(dt_p)) or (not np.isfinite(dt_m)):
                good = False
                break
            grad[ax] = (dt_p - dt_m) / (2.0 * eps)
        if not good:
            unresolved.append((i, side))
            continue

        denom = 1.0 + float(np.dot(grad, v_prev))
        if (not np.isfinite(denom)) or abs(denom) <= 1e-9:
            unresolved.append((i, side))
            continue
        te = float(np.clip(te_prev + dt0 / denom, t_lo, t_hi))
        predicted_requests.append((i, side, te))

    predicted_obs = _build_observed_from_interp_batch(
        interp=interp,
        sampled_trajectories=sampled_trajectories,
        requests=predicted_requests,
        point_b=point_b,
        use_gpu=use_gpu,
        batch_size=batch_size,
        gpu_min_batch=gpu_min_batch,
    )
    for i, side, _te in predicted_requests:
        obs = predicted_obs.get((i, side))
        if obs is None:
            unresolved.append((i, side))
            continue
        out[i][side] = obs

    if unresolved:
        unresolved_cases = sorted(set(int(i) for i, _ in unresolved))
        fallback_sub = _solve_interpolated_subset_for_t0(
            interp=interp,
            sampled_trajectories=sampled_trajectories,
            point_b=point_b,
            t0=float(t0),
            tmin=float(tmin),
            tmax=float(tmax),
            scan_samples=int(scan_samples),
            root_max_iter=int(root_max_iter),
            root_tol_time=float(root_tol_time),
            use_gpu=use_gpu,
            batch_size=int(batch_size),
            gpu_min_batch=int(gpu_min_batch),
            case_indices=unresolved_cases,
            previous_batch=previous_batch,
            previous_t0=previous_t0,
        )
        for i, side in unresolved:
            row = fallback_sub.get(int(i))
            if row is not None:
                out[i][side] = row.get(side)

    for i in range(n_cases):
        plus = out[i].get("plus")
        minus = out[i].get("minus")
        if isinstance(plus, dict) and isinstance(minus, dict):
            out[i]["earliest"] = plus if float(plus["emission_time_s"]) <= float(minus["emission_time_s"]) else minus
        elif isinstance(plus, dict):
            out[i]["earliest"] = plus
        elif isinstance(minus, dict):
            out[i]["earliest"] = minus
    return out


def _solve_direct_batch_for_t0(
    direct_sessions: List[TimelikeVisibilitySession],
    point_b: Tuple[float, float, float],
    t0: float,
    root_max_iter: int,
    root_tol_time: float,
    fallback_scan_samples: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for session in direct_sessions:
        out.append(
            session.solve(
                observer_point_b=point_b,
                observer_time_s=float(t0),
                root_max_iter=int(root_max_iter),
                root_tol_time=float(root_tol_time),
                fallback_scan_samples=int(fallback_scan_samples),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Advance observer time and compute earliest visible coordinates for multiple "
            "timelike trajectories from initial position/velocity."
        )
    )
    parser.add_argument("--quality", choices=["high", "medium", "fast"], default="fast")
    parser.add_argument("--integrator", choices=["symplectic", "rk4", "euler"], default="symplectic")
    parser.add_argument("--start-radius-light-seconds", type=float, default=5.0)
    parser.add_argument("--speed-frac-c", type=float, default=0.20)
    parser.add_argument("--tmin", type=float, default=-8.0)
    parser.add_argument("--tmax", type=float, default=320.0)
    parser.add_argument("--observer-times", type=int, default=100, help="Number of observer-time samples.")
    parser.add_argument("--dtau", type=float, default=1e-3)
    parser.add_argument("--max-tau", type=float, default=25.0, help="Max proper time in seconds.")
    parser.add_argument("--max-steps", type=int, default=300000, help="Max integration steps.")
    parser.add_argument(
        "--escape-radius-rs",
        type=float,
        default=80.0,
        help="Escape threshold radius in units of rs (default: 80).",
    )
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--solver",
        choices=["interpolated", "interpolated-linear", "direct"],
        default="interpolated-linear",
        help="Light-cone solver mode (default: interpolated-linear).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz table for interpolated mode.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--gpu-min-batch", type=int, default=256)
    args = parser.parse_args()

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = bh.schwarzschild_radius_m
    r0 = float(args.start_radius_light_seconds) * C
    speed = max(0.0, min(float(args.speed_frac_c), 0.999)) * C
    escape_radius_m = float(args.escape_radius_rs) * rs

    # Observer fixed at 2 rs on +x axis.
    point_b = (2.0 * rs, 0.0, 0.0)
    t_obs = np.linspace(float(args.tmin), float(args.tmax), max(4, int(args.observer_times)), dtype=float)

    cases: List[Tuple[str, Tuple[float, float, float], float, float, int]] = []
    v_circ = _local_circular_speed_schwarzschild(r0, rs)
    gamma_circ = 1.0 / (1.0 - (v_circ / C) ** 2) ** 0.5
    dphi_dtau_circ = gamma_circ * v_circ / r0
    tau_for_two_orbits = (4.0 * pi) / dphi_dtau_circ
    circular_max_tau = float(tau_for_two_orbits)
    circular_dtau = min(float(args.dtau), float(tau_for_two_orbits) / 20_000.0)
    circular_max_steps = max(int(args.max_steps), int(circular_max_tau / circular_dtau) + 10)
    cases.append(("Circular (90 deg @ v_circ)", (0.0, v_circ, 0.0), circular_dtau, circular_max_tau, circular_max_steps))
    for label, deg in [
        ("Outward (0 deg)", 0.0),
        ("Mostly Tangential (70 deg)", 70.0),
        ("Tangential (90 deg)", 90.0),
        ("Inward-angled (140 deg)", 140.0),
        ("Inward (180 deg)", 180.0),
    ]:
        cases.append((label, _build_initial_velocity(speed, deg), float(args.dtau), float(args.max_tau), int(args.max_steps)))

    solver_mode = str(args.solver)
    precompute_path: Optional[Path] = None
    if solver_mode in ("interpolated", "interpolated-linear"):
        precompute_path = _resolve_precompute_path(Path(args.input))
        if precompute_path is None:
            print(f"Precompute table not found at {args.input}; falling back to direct solve mode.")
            solver_mode = "direct"

    labels: List[str] = []
    trajectories: List[object] = []
    sampled_trajectories: List[SampledTrajectory3D] = []
    direct_sessions: List[TimelikeVisibilitySession] = []
    for label, v0, dtau_case, max_tau_case, max_steps_case in cases:
        base_session = TimelikeVisibilitySession(
            bh=bh,
            initial_position_m=(r0, 0.0, 0.0),
            initial_velocity_m_s=v0,
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            proper_time_step_s=float(dtau_case),
            integrator=str(args.integrator),
            max_steps=int(max_steps_case),
            max_proper_time_s=float(max_tau_case),
            escape_radius_m=float(escape_radius_m),
            use_gpu=bool(args.use_gpu),
        )
        traj = base_session.trajectory
        sampled = SampledTrajectory3D.from_arrays(
            ts=traj.ts,
            xs=traj.xs,
            ys=traj.ys,
            zs=traj.zs,
        )
        labels.append(label)
        trajectories.append(traj)
        sampled_trajectories.append(sampled)
        direct_sessions.append(base_session)

    interp_shared: Optional[PrecomputedEarliestInterpolator] = None
    if solver_mode in ("interpolated", "interpolated-linear"):
        interp_shared = PrecomputedEarliestInterpolator.from_npz(precompute_path)
        interp_shared.prepare_backend(use_gpu=bool(args.use_gpu))

    fig, (ax_xy, ax_time) = plt.subplots(1, 2, figsize=(16, 7.2))

    # Plot horizon/photon sphere and observer on x-y panel.
    tt = [2.0 * pi * i / 500 for i in range(501)]
    rph = bh.photon_sphere_radius_m
    ax_xy.fill([rs * cos(v) for v in tt], [rs * sin(v) for v in tt], color="black", alpha=0.2)
    ax_xy.plot([rs * cos(v) for v in tt], [rs * sin(v) for v in tt], color="black", lw=1.1, label="Event horizon")
    ax_xy.plot([rph * cos(v) for v in tt], [rph * sin(v) for v in tt], "k--", lw=1.0, label="Photon sphere")
    ax_xy.scatter([point_b[0]], [point_b[1]], color="magenta", s=55, marker="*", label="Observer B @ 2rs")

    colors = ["tab:cyan", "tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]
    r_plot_max = max(r0, rs * 4.0)
    vis_x_all: List[List[float]] = [[] for _ in labels]
    vis_y_all: List[List[float]] = [[] for _ in labels]
    plus_tobs_all: List[List[float]] = [[] for _ in labels]
    plus_te_all: List[List[float]] = [[] for _ in labels]
    minus_tobs_all: List[List[float]] = [[] for _ in labels]
    minus_te_all: List[List[float]] = [[] for _ in labels]
    prev_batch_interp: Optional[List[Dict[str, object]]] = None
    prev_t0_interp: Optional[float] = None

    for t0 in t_obs:
        if solver_mode in ("interpolated", "interpolated-linear"):
            if solver_mode == "interpolated-linear":
                batch_out = _solve_interpolated_linearized_batch_for_t0(
                    interp=interp_shared,
                    sampled_trajectories=sampled_trajectories,
                    point_b=point_b,
                    t0=float(t0),
                    tmin=float(args.tmin),
                    tmax=float(args.tmax),
                    scan_samples=41,
                    root_max_iter=12,
                    root_tol_time=1e-6,
                    use_gpu=bool(args.use_gpu),
                    batch_size=int(args.batch_size),
                    gpu_min_batch=int(args.gpu_min_batch),
                    previous_batch=prev_batch_interp,
                    previous_t0=prev_t0_interp,
                )
            else:
                batch_out = _solve_interpolated_batch_for_t0(
                    interp=interp_shared,
                    sampled_trajectories=sampled_trajectories,
                    point_b=point_b,
                    t0=float(t0),
                    tmin=float(args.tmin),
                    tmax=float(args.tmax),
                    scan_samples=41,
                    root_max_iter=12,
                    root_tol_time=1e-6,
                    use_gpu=bool(args.use_gpu),
                    batch_size=int(args.batch_size),
                    gpu_min_batch=int(args.gpu_min_batch),
                    previous_batch=prev_batch_interp,
                    previous_t0=prev_t0_interp,
                    seed_base_step_s=0.1,
                    seed_expand_steps=8,
                )
            prev_batch_interp = batch_out
            prev_t0_interp = float(t0)
        else:
            batch_out = _solve_direct_batch_for_t0(
                direct_sessions=direct_sessions,
                point_b=point_b,
                t0=float(t0),
                root_max_iter=12,
                root_tol_time=1e-6,
                fallback_scan_samples=41,
            )

        for i, out in enumerate(batch_out):
            plus = out.get("plus")
            if isinstance(plus, dict):
                plus_tobs_all[i].append(float(t0))
                plus_te_all[i].append(float(plus["emission_time_s"]))
            minus = out.get("minus")
            if isinstance(minus, dict):
                minus_tobs_all[i].append(float(t0))
                minus_te_all[i].append(float(minus["emission_time_s"]))
            earliest = out.get("earliest")
            if not isinstance(earliest, dict):
                continue
            te = float(earliest["emission_time_s"])
            pa = earliest.get("point_a_m")
            if not (isinstance(pa, (tuple, list)) and len(pa) == 3):
                pa = trajectories[i].eval_point(te)
            vis_x_all[i].append(float(pa[0]))
            vis_y_all[i].append(float(pa[1]))

    for i, (label, traj) in enumerate(zip(labels, trajectories)):
        col = colors[i % len(colors)]
        forward = np.asarray(traj.ts, dtype=float) >= -1e-12
        xs_plot = np.asarray(traj.xs, dtype=float)[forward]
        ys_plot = np.asarray(traj.ys, dtype=float)[forward]
        if xs_plot.size > 0 and ys_plot.size > 0:
            ax_xy.plot(xs_plot, ys_plot, color=col, lw=1.0, alpha=0.35, label=f"{label} trajectory")
            r_plot_max = max(r_plot_max, float(np.max(np.hypot(xs_plot, ys_plot))))
        if vis_x_all[i]:
            ax_xy.scatter(
                vis_x_all[i],
                vis_y_all[i],
                s=18,
                color=col,
                alpha=0.9,
                label=f"{label} earliest-visible points",
            )
        if plus_tobs_all[i]:
            ax_time.plot(plus_tobs_all[i], plus_te_all[i], color=col, lw=1.5, ls="-", label=f"{label} (+)")
        if minus_tobs_all[i]:
            ax_time.plot(minus_tobs_all[i], minus_te_all[i], color=col, lw=1.5, ls="--", label=f"{label} (-)")

    lim = 1.1 * max(r_plot_max, rs * 2.5)
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)
    ax_xy.set_aspect("equal", "box")
    ax_xy.grid(alpha=0.25)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Timelike trajectories + earliest-visible coordinates")
    ax_xy.legend(loc="lower left", fontsize=8)

    ax_time.grid(alpha=0.25)
    ax_time.set_xlabel("observer time t0 (s)")
    ax_time.set_ylabel("earliest emission time te (s)")
    ax_time.set_title(
        f"Earliest-visible te vs t0 (B fixed at 2rs, integrator={args.integrator}, solver={solver_mode})"
    )
    handles, labels = ax_time.get_legend_handles_labels()
    if handles:
        ax_time.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
