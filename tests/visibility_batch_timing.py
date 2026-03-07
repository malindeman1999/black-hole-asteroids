"""Batch timing benchmark for visibility solvers using precomputed interpolation.

This script compares per-trajectory per-call timing for batched circular-orbit
visibility queries across these solver variants:

- `interpolated`: batched bracket/root solve using precomputed interpolation.
- `interpolated-linear`: batched one-step linearized update using
  `dte ~= dt0 / (1 + grad(delta_t) . v)` with fallback behavior inside the
  implementation.

Backends:
- CPU interpolation (`--disable-gpu` or no CuPy/GPU available)
- GPU interpolation (when CuPy is available and `--disable-gpu` is not set)

Timing protocol:
- Batch sizes: `5^0 ... 5^k` where `k=--max-power` (default 5).
- For each batch size, run one initial query plus `--subsequent-samples`
  additional sequential queries (default: 10).
- Report times in milliseconds per trajectory per call:
  - `*_init`: first call only
  - `*_sub`: mean over subsequent calls only

Key CLI options:
- `--input`: precomputed `.npz` interpolation table.
- `--max-power`: largest exponent for batch sizes `5^k`.
- `--scan-samples`, `--root-max-iter`, `--root-tol-time`: root-solve controls.
- `--batch-size`, `--gpu-min-batch`: interpolation batching/backend controls.
- `--disable-gpu`: force CPU-only benchmark.

Example output:

Timing batched circular-orbit visibility solves (precomputed interpolation)
queries_per_batch=11 (initial + 10 subsequent), scan_samples=41, root_max_iter=12
batch_n | interp_cpu_init | interp_cpu_sub | linear_cpu_init | linear_cpu_sub | interp_gpu_init | interp_gpu_sub | linear_gpu_init | linear_gpu_sub
--------+-----------------+----------------+-----------------+----------------+-----------------+----------------+-----------------+----------------
      1 |         57.1468 |         31.742 |         45.5975 |          3.750 |            43.0317 |         31.095 |            44.8230 |          3.739
      5 |         34.0241 |         29.182 |         32.1296 |          2.788 |            31.4768 |         29.336 |            66.0395 |          5.456
     25 |         38.3350 |         30.423 |         30.1325 |          2.682 |            29.8996 |         28.970 |            30.1067 |          6.982
    125 |         29.1623 |         28.530 |         28.8844 |          2.575 |            33.0696 |         29.287 |            33.1787 |          3.092
"""

from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, cp
from earliest_visible_interpolated_session import SampledTrajectory3D
from precompute_earliest_grid import PrecomputedEarliestInterpolator

# Reuse the exact batched solver implementations from the plotting script.
import plot_visibility_from_initial_states as pvis


def _powers_of_five(max_power: int) -> List[int]:
    if max_power < 0:
        return []
    return [5**k for k in range(max_power + 1)]


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


def _build_circular_batch(
    n_orbits: int,
    rs: float,
    start_radius_light_seconds: float,
    tmin: float,
    tmax: float,
    trajectory_samples: int,
) -> List[SampledTrajectory3D]:
    n = max(1, int(n_orbits))
    m = max(2, int(trajectory_samples))
    ts = np.linspace(float(tmin), float(tmax), m, dtype=float)
    r0 = float(start_radius_light_seconds) * C
    # Two full revolutions across the configured window.
    window = max(1e-9, float(tmax) - float(tmin))
    omega = 4.0 * pi / window

    out: List[SampledTrajectory3D] = []
    for i in range(n):
        phase0 = 2.0 * pi * (float(i) / float(n))
        phase = phase0 + omega * (ts - float(tmin))
        xs = r0 * np.cos(phase)
        ys = r0 * np.sin(phase)
        zs = np.zeros_like(xs)
        out.append(SampledTrajectory3D.from_arrays(ts=ts, xs=xs, ys=ys, zs=zs))
    return out


def _run_variant(
    solver_fn: Callable[..., List[Dict[str, object]]],
    interp: PrecomputedEarliestInterpolator,
    sampled_batch: List[SampledTrajectory3D],
    point_b: Tuple[float, float, float],
    t_seq: Sequence[float],
    tmin: float,
    tmax: float,
    use_gpu: bool,
    batch_size: int,
    gpu_min_batch: int,
    scan_samples: int,
    root_max_iter: int,
    root_tol_time: float,
) -> Tuple[float, float, float]:
    prev_batch: Optional[List[Dict[str, object]]] = None
    prev_t0: Optional[float] = None
    call_times: List[float] = []
    for t in t_seq:
        t_call = time.perf_counter()
        out = solver_fn(
            interp=interp,
            sampled_trajectories=sampled_batch,
            point_b=point_b,
            t0=float(t),
            tmin=float(tmin),
            tmax=float(tmax),
            scan_samples=int(scan_samples),
            root_max_iter=int(root_max_iter),
            root_tol_time=float(root_tol_time),
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
            gpu_min_batch=int(gpu_min_batch),
            previous_batch=prev_batch,
            previous_t0=prev_t0,
        )
        call_times.append(time.perf_counter() - t_call)
        prev_batch = out
        prev_t0 = float(t)
    total_s = float(np.sum(np.asarray(call_times, dtype=float)))
    if len(call_times) > 1:
        sub = np.asarray(call_times[1:], dtype=float)
        subsequent_per_call_s = float(np.mean(sub))
        subsequent_std_s = float(np.std(sub))
    else:
        subsequent_per_call_s = total_s
        subsequent_std_s = 0.0
    initial_call_s = float(call_times[0]) if call_times else 0.0
    return initial_call_s, subsequent_per_call_s, subsequent_std_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Head-to-head timing for batched circular-orbit visibility solves using precomputed interpolation. "
            "Runs batches n=1,5,25,...,5^k and prints a table row after each batch."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz table.",
    )
    parser.add_argument("--max-power", type=int, default=5, help="Largest exponent k in n=5^k (default: 5).")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=320.0)
    parser.add_argument(
        "--t0-start",
        type=float,
        default=320.0,
        help="Initial observer time for each benchmark batch (default: 320.0).",
    )
    parser.add_argument(
        "--query-dt-s",
        type=float,
        default=0.1,
        help="Delta-t between sequential samples (default: 0.1).",
    )
    parser.add_argument(
        "--subsequent-samples",
        type=int,
        default=10,
        help="Number of samples after the initial sample (default: 10).",
    )
    parser.add_argument(
        "--start-radius-light-seconds",
        type=float,
        default=5.0,
        help="Circular orbit radius used for synthetic trajectory batch (default: 5.0).",
    )
    parser.add_argument("--scan-samples", type=int, default=41)
    parser.add_argument("--root-max-iter", type=int, default=12)
    parser.add_argument("--root-tol-time", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--gpu-min-batch", type=int, default=256)
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=4097,
        help="Per-orbit sampled points used to build synthetic trajectories (default: 4097).",
    )
    parser.add_argument(
        "--disable-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip GPU variants even if CuPy is available.",
    )
    parser.add_argument(
        "--include-interpolated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also benchmark non-linear interpolated variants (default: false).",
    )
    args = parser.parse_args()

    counts = _powers_of_five(int(args.max_power))
    if not counts:
        raise ValueError("No batch sizes to test. Use --max-power >= 0.")

    input_path = _resolve_input_path(Path(args.input))
    interp = PrecomputedEarliestInterpolator.from_npz(input_path)
    interp.prepare_backend(use_gpu=False)
    gpu_available = bool((cp is not None) and (not bool(args.disable_gpu)))
    if gpu_available:
        interp.prepare_backend(use_gpu=True)

    rs = float(interp.rs_m)
    point_b = (2.0 * rs, 0.0, 0.0)

    n_queries = max(1, int(args.subsequent_samples) + 1)
    t_seq = [float(args.t0_start) + i * float(args.query_dt_s) for i in range(n_queries)]

    print("Timing batched circular-orbit visibility solves (precomputed interpolation)")
    print(
        f"queries_per_batch={n_queries} (initial + {max(0, n_queries - 1)} subsequent), "
        f"scan_samples={int(args.scan_samples)}, root_max_iter={int(args.root_max_iter)}"
    )
    if bool(args.include_interpolated):
        print(
            "batch_n | interp_cpu_init | interp_cpu_sub | linear_cpu_init | linear_cpu_sub | "
            "interp_gpu_init | interp_gpu_sub | linear_gpu_init | linear_gpu_sub"
        )
        print(
            "--------+-----------------+----------------+-----------------+----------------+"
            "-----------------+----------------+-----------------+----------------"
        )
    else:
        print("batch_n | linear_cpu_init | linear_cpu_sub | linear_gpu_init | linear_gpu_sub")
        print("--------+-----------------+----------------+-----------------+----------------")

    for n in counts:
        sampled_batch = _build_circular_batch(
            n_orbits=int(n),
            rs=rs,
            start_radius_light_seconds=float(args.start_radius_light_seconds),
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            trajectory_samples=int(args.trajectory_samples),
        )
        n_traj = float(max(1, int(n)))

        interp_cpu_init_ms_per_traj = float("nan")
        interp_cpu_sub_ms_per_traj = float("nan")
        interp_gpu_txt = "       n/a        "
        interp_gpu_sub_txt = "      n/a     "
        if bool(args.include_interpolated):
            interp_cpu_init_s, interp_cpu_sub_s, _interp_cpu_sub_std_s = _run_variant(
                solver_fn=pvis._solve_interpolated_batch_for_t0,
                interp=interp,
                sampled_batch=sampled_batch,
                point_b=point_b,
                t_seq=t_seq,
                tmin=float(args.tmin),
                tmax=float(args.tmax),
                use_gpu=False,
                batch_size=int(args.batch_size),
                gpu_min_batch=int(args.gpu_min_batch),
                scan_samples=int(args.scan_samples),
                root_max_iter=int(args.root_max_iter),
                root_tol_time=float(args.root_tol_time),
            )
            interp_cpu_init_ms_per_traj = (interp_cpu_init_s * 1000.0) / n_traj
            interp_cpu_sub_ms_per_traj = (interp_cpu_sub_s * 1000.0) / n_traj
        linear_cpu_init_s, linear_cpu_sub_s, linear_cpu_sub_std_s = _run_variant(
            solver_fn=pvis._solve_interpolated_linearized_batch_for_t0,
            interp=interp,
            sampled_batch=sampled_batch,
            point_b=point_b,
            t_seq=t_seq,
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            use_gpu=False,
            batch_size=int(args.batch_size),
            gpu_min_batch=int(args.gpu_min_batch),
            scan_samples=int(args.scan_samples),
            root_max_iter=int(args.root_max_iter),
            root_tol_time=float(args.root_tol_time),
        )

        linear_cpu_init_ms_per_traj = (linear_cpu_init_s * 1000.0) / n_traj
        linear_cpu_sub_ms_per_traj = (linear_cpu_sub_s * 1000.0) / n_traj
        _linear_cpu_sub_std_ms_per_traj = (linear_cpu_sub_std_s * 1000.0) / n_traj

        if gpu_available:
            if bool(args.include_interpolated):
                interp_gpu_init_s, interp_gpu_sub_s, interp_gpu_sub_std_s = _run_variant(
                    solver_fn=pvis._solve_interpolated_batch_for_t0,
                    interp=interp,
                    sampled_batch=sampled_batch,
                    point_b=point_b,
                    t_seq=t_seq,
                    tmin=float(args.tmin),
                    tmax=float(args.tmax),
                    use_gpu=True,
                    batch_size=int(args.batch_size),
                    gpu_min_batch=int(args.gpu_min_batch),
                    scan_samples=int(args.scan_samples),
                    root_max_iter=int(args.root_max_iter),
                    root_tol_time=float(args.root_tol_time),
                )
                interp_gpu_init_ms_per_traj = (interp_gpu_init_s * 1000.0) / n_traj
                interp_gpu_sub_ms_per_traj = (interp_gpu_sub_s * 1000.0) / n_traj
                interp_gpu_sub_std_ms_per_traj = (interp_gpu_sub_std_s * 1000.0) / n_traj
                interp_gpu_txt = f"{interp_gpu_init_ms_per_traj:18.4f}"
                interp_gpu_sub_txt = f"{interp_gpu_sub_ms_per_traj:14.3f}"
            linear_gpu_init_s, linear_gpu_sub_s, linear_gpu_sub_std_s = _run_variant(
                solver_fn=pvis._solve_interpolated_linearized_batch_for_t0,
                interp=interp,
                sampled_batch=sampled_batch,
                point_b=point_b,
                t_seq=t_seq,
                tmin=float(args.tmin),
                tmax=float(args.tmax),
                use_gpu=True,
                batch_size=int(args.batch_size),
                gpu_min_batch=int(args.gpu_min_batch),
                scan_samples=int(args.scan_samples),
                root_max_iter=int(args.root_max_iter),
                root_tol_time=float(args.root_tol_time),
            )
            linear_gpu_init_ms_per_traj = (linear_gpu_init_s * 1000.0) / n_traj
            linear_gpu_sub_ms_per_traj = (linear_gpu_sub_s * 1000.0) / n_traj
            _linear_gpu_sub_std_ms_per_traj = (linear_gpu_sub_std_s * 1000.0) / n_traj
            linear_gpu_txt = f"{linear_gpu_init_ms_per_traj:18.4f}"
            linear_gpu_sub_txt = f"{linear_gpu_sub_ms_per_traj:14.3f}"
        else:
            linear_gpu_txt = "       n/a        "
            linear_gpu_sub_txt = "      n/a     "

        if bool(args.include_interpolated):
            print(
                f"{int(n):7d} | {interp_cpu_init_ms_per_traj:15.4f} | "
                f"{interp_cpu_sub_ms_per_traj:14.3f} | "
                f"{linear_cpu_init_ms_per_traj:15.4f} | "
                f"{linear_cpu_sub_ms_per_traj:14.3f} | "
                f"{interp_gpu_txt} | {interp_gpu_sub_txt} | {linear_gpu_txt} | {linear_gpu_sub_txt}"
            )
        else:
            print(
                f"{int(n):7d} | {linear_cpu_init_ms_per_traj:15.4f} | "
                f"{linear_cpu_sub_ms_per_traj:14.3f} | "
                f"{linear_gpu_txt} | {linear_gpu_sub_txt}"
            )


if __name__ == "__main__":
    main()
