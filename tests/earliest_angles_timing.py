from __future__ import annotations

from math import pi
from pathlib import Path
import argparse
import sys
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import SchwarzschildBlackHole, cp


def _powers_of_five(max_power: int) -> List[int]:
    if max_power < 0:
        return []
    return [5**k for k in range(max_power + 1)]


def _build_sampled_trajectory(
    n: int, rs: float, tmin: float, tmax: float
) -> Tuple[Callable[[float], Sequence[float]], Tuple[float, float, float]]:
    # Smooth deterministic trajectory around the hole.
    ts = np.linspace(tmin, tmax, n, dtype=float)
    radius = 7.0 * rs
    omega = 0.12
    phase = omega * ts
    xs = radius * np.cos(phase)
    ys = radius * np.sin(phase)
    zs = 0.12 * radius * np.sin(0.37 * phase)

    def trajectory(t: float) -> Sequence[float]:
        tt = float(np.clip(t, tmin, tmax))
        x = float(np.interp(tt, ts, xs))
        y = float(np.interp(tt, ts, ys))
        z = float(np.interp(tt, ts, zs))
        return (x, y, z)

    # Keep observer fixed and outside the source radius.
    point_b = (8.0 * rs, 0.0, 0.0)
    return trajectory, point_b


def _run_timing(
    bh: SchwarzschildBlackHole,
    counts: Sequence[int],
    use_gpu: bool,
    quality: str,
    tmin: float,
    tmax: float,
    t0: float,
) -> None:
    mode = "GPU-batch" if (use_gpu and cp is not None) else "CPU-batch"
    print(f"Mode: {mode} | quality={quality}")
    print(f"Testing scan_samples: {', '.join(str(x) for x in counts)}")
    print("n       | load_s    | compute_s | total_s   | per_sample_ms | sides")
    print("--------+-----------+-----------+-----------+---------------+------")

    rs = bh.schwarzschild_radius_m
    for n in counts:
        load_t0 = time.perf_counter()
        traj, point_b = _build_sampled_trajectory(n=n, rs=rs, tmin=tmin, tmax=tmax)
        load_s = time.perf_counter() - load_t0

        compute_t0 = time.perf_counter()
        try:
            result = bh.find_earliest_observed_angles_at_b(
                trajectory=traj,
                point_b=point_b,
                t0=t0,
                tmin=tmin,
                tmax=tmax,
                scan_samples=n,
                use_gpu=use_gpu,
            )
            compute_s = time.perf_counter() - compute_t0
            total_s = load_s + compute_s
            per_sample_ms = (compute_s / n) * 1000.0 if n > 0 else 0.0
            sides = int(result.plus is not None) + int(result.minus is not None)
            print(
                f"{n:7d} | {load_s:9.6f} | {compute_s:9.6f} | {total_s:9.6f} | "
                f"{per_sample_ms:13.4f} | {sides:5d}"
            )
        except Exception as exc:
            compute_s = time.perf_counter() - compute_t0
            total_s = load_s + compute_s
            print(
                f"{n:7d} | {load_s:9.6f} | {compute_s:9.6f} | {total_s:9.6f} | "
                f"{0.0:13.4f} | failed ({exc!r})"
            )
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time earliest-observed-angle solve at B for scan_samples=1,5,25,..."
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=4,
        help="Largest exponent k in n=5^k (default: 4 => up to 625 samples).",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU batch path when CuPy is available (default: true).",
    )
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "fast"],
        default="fast",
        help="Solver precision preset (default: fast).",
    )
    parser.add_argument("--tmin", type=float, default=0.0, help="Trajectory start time.")
    parser.add_argument("--tmax", type=float, default=30.0, help="Trajectory end time.")
    parser.add_argument("--t0", type=float, default=30.0, help="Observer time at B.")
    args = parser.parse_args()

    counts = _powers_of_five(args.max_power)
    if not counts:
        raise ValueError("No counts to test. Use --max-power >= 0.")

    # Avoid invalid scan sizes in the solver.
    counts = [max(3, n) for n in counts]

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    _run_timing(
        bh=bh,
        counts=counts,
        use_gpu=args.use_gpu,
        quality=args.quality,
        tmin=args.tmin,
        tmax=args.tmax,
        t0=args.t0,
    )


if __name__ == "__main__":
    main()
