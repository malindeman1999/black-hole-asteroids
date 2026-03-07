from __future__ import annotations

import argparse
from math import acos, cos, pi, sin, sqrt
from pathlib import Path
import random
import sys
import time
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import GeodesicSolution, SchwarzschildBlackHole, cp


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _bisection_root(func, lo: float, hi: float, max_iter: int = 80) -> float:
    flo = func(lo)
    fhi = func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("Root not bracketed")
    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = func(m)
        if abs(fm) < 1e-12:
            return m
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _cumulative_trapezoid(values: Sequence[float], x: Sequence[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(values)):
        dx = x[i] - x[i - 1]
        out.append(out[-1] + 0.5 * (values[i - 1] + values[i]) * dx)
    return out


def _find_turning_radius(bh: SchwarzschildBlackHole, b: float, r_cap: float) -> float:
    rs = bh.schwarzschild_radius_m
    rph = bh.photon_sphere_radius_m
    lo = rph * (1.0 + 1e-9)
    hi = r_cap * (1.0 - 1e-9)

    def f(r: float) -> float:
        return 1.0 / (b * b) - (1.0 - rs / r) / (r * r)

    # Locate a sign change robustly.
    nscan = 256
    prev_r = lo
    prev_f = f(prev_r)
    for i in range(1, nscan + 1):
        rr = lo + (hi - lo) * i / nscan
        ff = f(rr)
        if prev_f * ff <= 0.0:
            return _bisection_root(f, prev_r, rr)
        prev_r, prev_f = rr, ff
    raise RuntimeError("Failed to locate turning radius")


def _build_path_profile(
    bh: SchwarzschildBlackHole, r_start: float, r_end: float, path: GeodesicSolution, n: int = 1500
) -> Tuple[List[float], List[float]]:
    rs = bh.schwarzschild_radius_m
    b2 = path.impact_parameter_m * path.impact_parameter_m
    target_phi = path.target_azimuth_rad

    def phi_density(r: float) -> float:
        w = 1.0 / b2 - (1.0 - rs / r) / (r * r)
        if w < 0.0 and w > -1e-14:
            w = 0.0
        if w <= 0.0:
            return 0.0
        return 1.0 / (r * r * sqrt(w))

    if path.branch == "monotonic":
        s = [i / (n - 1) for i in range(n)]
        dr = r_end - r_start
        r_samples = [r_start + dr * si for si in s]
        dens = [abs(dr) * phi_density(r) for r in r_samples]
        phi_samples = _cumulative_trapezoid(dens, s)
    else:
        r_turn = _find_turning_radius(bh, path.impact_parameter_m, min(r_start, r_end))
        n1 = max(64, n // 2)
        n2 = max(64, n - n1 + 1)

        # Leg 1: start -> turning radius, substitution avoids endpoint singularity.
        s1 = [i / (n1 - 1) for i in range(n1)]
        d1 = r_start - r_turn
        r1 = [r_turn + d1 * (1.0 - si) * (1.0 - si) for si in s1]
        dens1 = [2.0 * abs(d1) * (1.0 - si) * phi_density(rr) for si, rr in zip(s1, r1)]
        phi1 = _cumulative_trapezoid(dens1, s1)

        # Leg 2: turning radius -> end.
        s2 = [i / (n2 - 1) for i in range(n2)]
        d2 = r_end - r_turn
        r2 = [r_turn + d2 * si * si for si in s2]
        dens2 = [2.0 * abs(d2) * si * phi_density(rr) for si, rr in zip(s2, r2)]
        phi2 = _cumulative_trapezoid(dens2, s2)
        phi2 = [x + phi1[-1] for x in phi2]

        r_samples = r1 + r2[1:]
        phi_samples = phi1 + phi2[1:]

    # Match the exact solved target angle.
    if phi_samples[-1] > 0.0:
        scale = target_phi / phi_samples[-1]
        phi_samples = [p * scale for p in phi_samples]
    else:
        phi_samples = [target_phi * i / max(1, len(phi_samples) - 1) for i in range(len(phi_samples))]

    return r_samples, phi_samples


def _angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    ra = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    rb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return acos(_clamp(dot / (ra * rb), -1.0, 1.0))


def _make_point(r: float, theta: float, z_scale: float) -> Tuple[float, float, float]:
    return (r * cos(theta), r * sin(theta), z_scale * r)


def _classify_path_pair(result) -> str:
    if len(result.paths) < 2:
        return "single"
    p1, p2 = result.paths[0], result.paths[1]
    db_rel = abs(p1.impact_parameter_m - p2.impact_parameter_m) / max(1.0, abs(p1.impact_parameter_m), abs(p2.impact_parameter_m))
    dt_abs = abs(p1.travel_time_s - p2.travel_time_s)
    dphi = abs(p1.target_azimuth_rad - p2.target_azimuth_rad)
    if db_rel < 1e-4 and dt_abs < 1e-3 and dphi < 1e-3 and p1.direction == p2.direction:
        return "near-duplicate"
    return "distinct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot two shortest geodesics for random point pairs.")
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "fast"],
        default="fast",
        help="Solver precision preset (default: fast).",
    )
    args = parser.parse_args()

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = bh.schwarzschild_radius_m
    rng = random.Random(42)

    # Build 20 deterministic random point pairs around the hole.
    # First 8 are forced near opposite sides; remaining 12 are fully random.
    point_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for i in range(20):
        ra = rs * rng.uniform(4.8, 9.0)
        rb = rs * rng.uniform(4.8, 9.0)
        ta = rng.uniform(0.0, 2.0 * pi)
        if i < 8:
            tb = (ta + pi + rng.uniform(-0.22, 0.22)) % (2.0 * pi)
        else:
            tb = rng.uniform(0.0, 2.0 * pi)
        za = rng.uniform(-0.18, 0.18)
        zb = rng.uniform(-0.18, 0.18)
        point_a = _make_point(ra, ta, za)
        point_b = _make_point(rb, tb, zb)
        point_pairs.append((point_a, point_b))

    # Compute only (no plotting) using the batch API.
    use_gpu = True
    gpu_active = bool(use_gpu and cp is not None)
    compute_mode = "GPU-batch" if gpu_active else "CPU-batch"
    compute_t0 = time.perf_counter()
    solved_results = bh.find_two_shortest_geodesics_batch(point_pairs, use_gpu=use_gpu)
    total_compute_s = time.perf_counter() - compute_t0

    solved = [(i, solved_results[i]) for i in range(len(solved_results))]
    print(f"Computed {len(solved)} point pairs with mode={compute_mode}.")
    print(f"Total batch compute time (excluding plotting): {total_compute_s:.6f} s")
    opposite_count = 0
    near_duplicate_count = 0
    for idx, result in solved:
        gamma_deg = result.separation_angle_rad * 180.0 / pi
        if gamma_deg > 150.0:
            opposite_count += 1
        pair_type = _classify_path_pair(result)
        if pair_type == "near-duplicate":
            near_duplicate_count += 1
        print(
            f"pair {idx + 1:02d}: "
            f"gamma={gamma_deg:6.2f} deg, paths={len(result.paths)}, "
            f"lag={result.lag_between_fastest_two_s:.6f} s, type={pair_type}"
        )
    print(f"Pairs near opposite sides (gamma > 150 deg): {opposite_count}")
    print(f"Pairs flagged as near-duplicate: {near_duplicate_count}")

    # Plot each pair in its own local center-A-B plane subplot.
    fig, axes = plt.subplots(4, 5, figsize=(18, 13))
    axes_list = list(axes.ravel())
    t = [2.0 * pi * i / 240 for i in range(241)]
    rph = bh.photon_sphere_radius_m
    colors = ["tab:blue", "tab:orange"]

    for ax, (idx, result) in zip(axes_list, solved):
        start = result.start_point
        end = result.end_point
        r_start = sqrt(start[0] * start[0] + start[1] * start[1] + start[2] * start[2])
        r_end = sqrt(end[0] * end[0] + end[1] * end[1] + end[2] * end[2])
        gamma = _angle_between(start, end)

        ax.plot([rs * cos(v) for v in t], [rs * sin(v) for v in t], "k-", lw=1.2)
        ax.plot([rph * cos(v) for v in t], [rph * sin(v) for v in t], "k--", lw=0.9)

        for path_idx, path in enumerate(result.paths):
            r_samples, phi_samples = _build_path_profile(bh, r_start, r_end, path, n=900)
            theta = [path.direction * p for p in phi_samples]
            xs = [r * cos(th) for r, th in zip(r_samples, theta)]
            ys = [r * sin(th) for r, th in zip(r_samples, theta)]
            ax.plot(xs, ys, color=colors[path_idx % len(colors)], lw=1.5, alpha=0.95)

        ax.scatter([r_start], [0.0], c="green", s=22)
        ax.scatter([r_end * cos(gamma)], [r_end * sin(gamma)], c="red", s=22)
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.2)
        gamma_deg = result.separation_angle_rad * 180.0 / pi
        ax.set_title(f"Pair {idx + 1} (batch, {gamma_deg:.1f} deg)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Two shortest null geodesics for 20 point pairs\n"
        f"compute only: {total_compute_s:.6f} s total (plotting not included)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
