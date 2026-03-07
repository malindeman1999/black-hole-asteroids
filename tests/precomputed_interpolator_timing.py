from __future__ import annotations

from math import pi
from pathlib import Path
import argparse
import random
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import cp
from precompute_earliest_grid import PrecomputedEarliestInterpolator


def _make_point(r: float, theta: float, z_scale: float) -> Tuple[float, float, float]:
    from math import cos, sin

    return (r * cos(theta), r * sin(theta), z_scale * r)


def _build_point_pairs(
    n: int, rs: float, rng: random.Random, rmin_rs: float, rmax_rs: float
) -> Tuple[np.ndarray, np.ndarray]:
    a_list: List[Tuple[float, float, float]] = []
    b_list: List[Tuple[float, float, float]] = []
    for i in range(n):
        ra = rs * rng.uniform(rmin_rs, rmax_rs)
        rb = rs * rng.uniform(rmin_rs, rmax_rs)
        ta = rng.uniform(0.0, 2.0 * pi)
        if i < max(8, n // 4):
            tb = (ta + pi + rng.uniform(-0.22, 0.22)) % (2.0 * pi)
        else:
            tb = rng.uniform(0.0, 2.0 * pi)
        za = rng.uniform(-0.18, 0.18)
        zb = rng.uniform(-0.18, 0.18)
        a_list.append(_make_point(ra, ta, za))
        b_list.append(_make_point(rb, tb, zb))
    return np.asarray(a_list, dtype=float), np.asarray(b_list, dtype=float)


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


def _run_timing(
    interp: PrecomputedEarliestInterpolator,
    all_a: np.ndarray,
    all_b: np.ndarray,
    counts: Sequence[int],
    use_gpu: bool,
    batch_size: int,
) -> None:
    mode = "GPU-batch" if (use_gpu and cp is not None) else "CPU-batch"
    print(f"Mode: {mode} | batch_size={batch_size}")
    print(f"Testing counts: {', '.join(str(x) for x in counts)}")
    print("n       | load_s    | compute_s | total_s   | per_pair_ms | ok_both")
    print("--------+-----------+-----------+-----------+-------------+--------")

    for n in counts:
        load_t0 = time.perf_counter()
        a_chunk = np.asarray(all_a[:n], dtype=float)
        b_chunk = np.asarray(all_b[:n], dtype=float)
        load_s = time.perf_counter() - load_t0

        compute_t0 = time.perf_counter()
        try:
            out = interp.interpolate_pairs_3d(
                a_points_m=a_chunk,
                b_points_m=b_chunk,
                use_gpu=use_gpu,
                batch_size=batch_size,
            )
            compute_s = time.perf_counter() - compute_t0
            total_s = load_s + compute_s
            per_pair_ms = (compute_s / n) * 1000.0 if n > 0 else 0.0
            ok_both = int(np.sum(out["ok_both"]))
            print(
                f"{n:7d} | {load_s:9.6f} | {compute_s:9.6f} | {total_s:9.6f} | "
                f"{per_pair_ms:11.4f} | {ok_both:7d}"
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
        description="Time precomputed interpolation for N=1,5,25,... on 3D (A,B) pairs."
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
        default=6,
        help="Largest exponent k in N=5^k (default: 6 => up to 15625 pairs).",
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
        help="Use GPU path when CuPy is available (default: true).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic pair generation (default: 42).",
    )
    parser.add_argument(
        "--rmin-rs",
        type=float,
        default=5.0,
        help="Minimum random radius for A/B generation in rs units (default: 5.0).",
    )
    parser.add_argument(
        "--rmax-rs",
        type=float,
        default=9.0,
        help="Maximum random radius for A/B generation in rs units (default: 9.0).",
    )
    args = parser.parse_args()

    counts = _powers_of_five(args.max_power)
    if not counts:
        raise ValueError("No counts to test. Use --max-power >= 0.")

    input_path = _resolve_input_path(args.input)
    interp = PrecomputedEarliestInterpolator.from_npz(input_path)

    rng = random.Random(args.seed)
    max_n = max(counts)
    all_a, all_b = _build_point_pairs(
        n=max_n,
        rs=interp.rs_m,
        rng=rng,
        rmin_rs=float(args.rmin_rs),
        rmax_rs=float(args.rmax_rs),
    )

    _run_timing(
        interp=interp,
        all_a=all_a,
        all_b=all_b,
        counts=counts,
        use_gpu=bool(args.use_gpu),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()
