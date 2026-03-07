from __future__ import annotations

from math import pi
from pathlib import Path
import argparse
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import cp
from earliest_visible_interpolated_session import EarliestVisibleInterpolatedSession, SampledTrajectory3D
from precompute_earliest_grid import PrecomputedEarliestInterpolator


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


def _build_sampled_trajectory(rs: float, tmin: float, tmax: float, samples: int) -> SampledTrajectory3D:
    n = max(2, int(samples))
    ts = np.linspace(tmin, tmax, n, dtype=float)
    radius = 7.0 * rs
    omega = 0.12
    phase = omega * ts
    xs = radius * np.cos(phase)
    ys = radius * np.sin(phase)
    zs = 0.12 * radius * np.sin(0.37 * phase)
    return SampledTrajectory3D.from_arrays(ts=ts, xs=xs, ys=ys, zs=zs)


def _advance_b_sub_luminal(prev_b: np.ndarray, dt_s: float, speed_frac_c: float, step_index: int) -> np.ndarray:
    c = 299_792_458.0
    v = max(0.0, min(float(speed_frac_c), 0.999)) * c
    max_step = v * max(0.0, float(dt_s))
    r = float(np.linalg.norm(prev_b))
    if r <= 0.0 or max_step <= 0.0:
        return prev_b.copy()
    # Mixed tangential/radial motion (still capped by max_step) so B_r/rs changes over time.
    # Tangential component advances azimuth; radial component oscillates deterministically.
    tangential_frac = 0.8
    radial_frac = 0.2 * np.sin(0.37 * float(step_index))
    tangential_step = abs(tangential_frac) * max_step
    radial_step = radial_frac * max_step
    # Enforce total displacement budget.
    if tangential_step + abs(radial_step) > max_step and (tangential_step + abs(radial_step)) > 0.0:
        scale = max_step / (tangential_step + abs(radial_step))
        tangential_step *= scale
        radial_step *= scale
    dphi = tangential_step / r
    phi = float(np.arctan2(prev_b[1], prev_b[0])) + dphi
    r_new = max(1e-6, r + radial_step)
    z = float(prev_b[2])
    b_new = np.asarray([r_new * np.cos(phi), r_new * np.sin(phi), z], dtype=float)
    return b_new


def _run_timing(
    session: EarliestVisibleInterpolatedSession,
    counts: Sequence[int],
    tmin: float,
    tmax: float,
    t0: float,
    root_max_iter: int,
    root_tol_time: float,
    b_radius_rs: float,
    query_dt_s: float,
    b_speed_frac_c: float,
    scan_samples_fallback: int,
) -> None:
    mode = "GPU-batch" if (session.use_gpu and cp is not None) else "CPU-batch"
    print(f"Mode: {mode} | batch_size={session.batch_size}")
    print(f"Testing scan_samples: {', '.join(str(x) for x in counts)}")
    print("n       | compute_s | per_sample_ms | sides | t0_s      | B_r/rs")
    print("--------+-----------+---------------+-------+-----------+--------")

    rs = session.interp.rs_m
    b_cur = np.asarray([float(b_radius_rs) * rs, 0.0, 0.0], dtype=float)
    t0_cur = float(t0)
    dt_sign = -1.0 if t0_cur >= float(tmax) - 1e-12 else 1.0
    prev_result = None
    prev_t0 = None
    for n in counts:
        point_b = tuple(float(x) for x in b_cur.tolist())
        t0_i = float(np.clip(t0_cur, tmin, tmax))

        compute_t0 = time.perf_counter()
        try:
            result = session.solve_from_previous(
                point_b=point_b,
                t0=t0_i,
                tmin=tmin,
                tmax=tmax,
                previous_result=prev_result,
                previous_t0=prev_t0,
                scan_samples_fallback=max(int(scan_samples_fallback), int(n)),
                root_max_iter=root_max_iter,
                root_tol_time=root_tol_time,
                seed_base_step_s=float(query_dt_s),
            )
            compute_s = time.perf_counter() - compute_t0
            per_sample_ms = (compute_s / n) * 1000.0 if n > 0 else 0.0
            sides = int(result["plus"] is not None) + int(result["minus"] is not None)
            r_b = float(np.linalg.norm(np.asarray(point_b, dtype=float))) / rs
            print(f"{n:7d} | {compute_s:9.6f} | {per_sample_ms:13.4f} | {sides:5d} | {t0_i:9.4f} | {r_b:7.4f}")
            prev_result = result
            prev_t0 = t0_i
        except Exception as exc:
            compute_s = time.perf_counter() - compute_t0
            print(f"{n:7d} | {compute_s:9.6f} | {0.0:13.4f} | failed ({exc!r})")
            break

        # Advance observer spacetime state for next query.
        b_cur = _advance_b_sub_luminal(
            b_cur,
            dt_s=float(query_dt_s),
            speed_frac_c=float(b_speed_frac_c),
            step_index=int(n),
        )
        t0_next = t0_cur + dt_sign * float(query_dt_s)
        if t0_next > float(tmax):
            dt_sign = -1.0
            t0_next = t0_cur + dt_sign * float(query_dt_s)
        elif t0_next < float(tmin):
            dt_sign = 1.0
            t0_next = t0_cur + dt_sign * float(query_dt_s)
        t0_cur = t0_next


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Time earliest-visible solve using precomputed interpolation session for scan_samples=1,5,25,...\n"
            "One-time trajectory/session upload is measured separately; table reports only per-query solve time."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz table.",
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=7,
        help="Largest exponent k in n=5^k (default: 7 => up to 78125 samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Interpolation batch size (default: 5000).",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU interpolation when CuPy is available (default: true).",
    )
    parser.add_argument("--tmin", type=float, default=0.0, help="Trajectory start time.")
    parser.add_argument("--tmax", type=float, default=30.0, help="Trajectory end time.")
    parser.add_argument("--t0", type=float, default=30.0, help="Base observer time at B.")
    parser.add_argument("--b-radius-rs", type=float, default=8.0, help="Observer radius in rs units (default: 8.0).")
    parser.add_argument("--root-max-iter", type=int, default=36, help="Root bisection iterations (default: 36).")
    parser.add_argument("--root-tol-time", type=float, default=1e-6, help="Root time tolerance in seconds.")
    parser.add_argument(
        "--trajectory-upload-samples",
        type=int,
        default=4097,
        help="Number of samples used to pre-upload/cached trajectory (default: 4097).",
    )
    parser.add_argument(
        "--gpu-min-batch",
        type=int,
        default=256,
        help=(
            "When using GPU, interpolation calls smaller than this are run on CPU "
            "to avoid GPU launch overhead (default: 256)."
        ),
    )
    parser.add_argument(
        "--query-dt-s",
        type=float,
        default=0.1,
        help="Time step between consecutive B queries in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--b-speed-frac-c",
        type=float,
        default=0.1,
        help="Observer B speed as fraction of c between queries, constrained <1 (default: 0.1).",
    )
    parser.add_argument(
        "--scan-samples-fallback",
        type=int,
        default=257,
        help="Fallback full-scan samples when warm-start cannot bracket (default: 257).",
    )
    args = parser.parse_args()

    counts = _powers_of_five(args.max_power)
    if not counts:
        raise ValueError("No counts to test. Use --max-power >= 0.")
    counts = [max(3, n) for n in counts]

    input_path = _resolve_input_path(args.input)

    # One-time setup phase: load precompute metadata, sample trajectory, and upload/cache tables.
    setup_t0 = time.perf_counter()
    interp_meta = PrecomputedEarliestInterpolator.from_npz(input_path)
    sampled = _build_sampled_trajectory(
        rs=interp_meta.rs_m,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        samples=int(args.trajectory_upload_samples),
    )
    session = EarliestVisibleInterpolatedSession(
        precompute_npz=input_path,
        sampled_trajectory=sampled,
        use_gpu=bool(args.use_gpu),
        batch_size=int(args.batch_size),
        gpu_min_batch=int(args.gpu_min_batch),
    )
    setup_s = time.perf_counter() - setup_t0

    mode = "GPU-batch" if (bool(args.use_gpu) and cp is not None) else "CPU-batch"
    print(f"One-time setup complete: mode={mode}, setup_s={setup_s:.6f}, trajectory_samples={sampled.ts.size}")

    _run_timing(
        session=session,
        counts=counts,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        t0=float(args.t0),
        root_max_iter=int(args.root_max_iter),
        root_tol_time=float(args.root_tol_time),
        b_radius_rs=float(args.b_radius_rs),
        query_dt_s=float(args.query_dt_s),
        b_speed_frac_c=float(args.b_speed_frac_c),
        scan_samples_fallback=int(args.scan_samples_fallback),
    )


if __name__ == "__main__":
    main()
