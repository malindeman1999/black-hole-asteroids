from __future__ import annotations

from math import pi
from pathlib import Path
import argparse
import random
import sys
import time
from typing import List, Sequence, Tuple

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import SchwarzschildBlackHole, cp


def _make_point(r: float, theta: float, z_scale: float) -> Tuple[float, float, float]:
    from math import cos, sin

    return (r * cos(theta), r * sin(theta), z_scale * r)


def _build_point_pairs(
    n: int, rs: float, rng: random.Random
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for i in range(n):
        ra = rs * rng.uniform(4.8, 9.0)
        rb = rs * rng.uniform(4.8, 9.0)
        ta = rng.uniform(0.0, 2.0 * pi)
        if i < max(8, n // 4):
            tb = (ta + pi + rng.uniform(-0.22, 0.22)) % (2.0 * pi)
        else:
            tb = rng.uniform(0.0, 2.0 * pi)
        za = rng.uniform(-0.18, 0.18)
        zb = rng.uniform(-0.18, 0.18)
        pairs.append((_make_point(ra, ta, za), _make_point(rb, tb, zb)))
    return pairs


def _powers_of_five(max_power: int) -> List[int]:
    if max_power < 0:
        return []
    return [5**k for k in range(max_power + 1)]


def _run_timing(
    bh: SchwarzschildBlackHole,
    all_pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
    counts: Sequence[int],
    use_gpu: bool,
    quality: str,
) -> None:
    mode = "GPU-batch" if (use_gpu and cp is not None) else "CPU-batch"
    print(f"Mode: {mode} | quality={quality}")
    print(f"Testing counts: {', '.join(str(x) for x in counts)}")
    print("n       | load_s    | compute_s | total_s   | per_pair_ms | solved")
    print("--------+-----------+-----------+-----------+-------------+-------")

    for n in counts:
        load_t0 = time.perf_counter()
        # Measure host-side batch preparation/loading time separately.
        pairs = list(all_pairs[:n])
        load_s = time.perf_counter() - load_t0

        compute_t0 = time.perf_counter()
        try:
            results = bh.find_two_shortest_geodesics_batch(pairs, use_gpu=use_gpu)
            compute_s = time.perf_counter() - compute_t0
            total_s = load_s + compute_s
            per_pair_ms = (compute_s / n) * 1000.0 if n > 0 else 0.0
            print(
                f"{n:7d} | {load_s:9.6f} | {compute_s:9.6f} | {total_s:9.6f} | "
                f"{per_pair_ms:11.4f} | {len(results):6d}"
            )
        except Exception as exc:
            compute_s = time.perf_counter() - compute_t0
            total_s = load_s + compute_s
            print(
                f"{n:7d} | {load_s:9.6f} | {compute_s:9.6f} | {total_s:9.6f} | "
                f"{0.0:11.4f} | failed ({exc!r})"
            )
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time batched geodesic computation for N=1,5,25,... (no plotting)."
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=6,
        help="Largest exponent k in N=5^k (default: 6 => up to 15625 pairs).",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU batch path when CuPy is available (default: true).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic pair generation (default: 42).",
    )
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "fast"],
        default="fast",
        help="Solver precision preset (default: fast).",
    )
    args = parser.parse_args()

    counts = _powers_of_five(args.max_power)
    if not counts:
        raise ValueError("No counts to test. Use --max-power >= 0.")

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rng = random.Random(args.seed)
    max_n = max(counts)
    all_pairs = _build_point_pairs(max_n, bh.schwarzschild_radius_m, rng)

    _run_timing(bh, all_pairs, counts, use_gpu=args.use_gpu, quality=args.quality)


if __name__ == "__main__":
    main()
