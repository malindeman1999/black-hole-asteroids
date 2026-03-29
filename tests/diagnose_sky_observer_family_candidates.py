from __future__ import annotations

import argparse
from math import acos, cos, pi, sin
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole, cp
from precompute_earliest_grid import _arrival_direction_at_b_for_pair, _dir_world_to_local


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _unit_xy(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / n


def _local_dir_to_world(point_xy: np.ndarray, local_dir_xy: np.ndarray) -> np.ndarray:
    er = _unit_xy(np.asarray(point_xy[:2], dtype=float))
    ephi = np.asarray([-er[1], er[0]], dtype=float)
    d = float(local_dir_xy[0]) * er + float(local_dir_xy[1]) * ephi
    return _unit_xy(d)


def _solve_all_candidates(bh: SchwarzschildBlackHole, a_point: np.ndarray, b_point: np.ndarray, use_gpu: bool):
    a = np.asarray(a_point, dtype=float)
    b = np.asarray(b_point, dtype=float)
    r1 = float(np.linalg.norm(a))
    r2 = float(np.linalg.norm(b))
    gamma = acos(_clamp(float(np.dot(a, b)) / max(1e-12, r1 * r2), -1.0, 1.0))
    targets = [(+1, gamma, "short"), (-1, 2.0 * pi - gamma, "long")]
    out = []
    for direction, target, label in targets:
        if target <= 1e-12:
            continue
        sols = bh._solve_for_target_azimuth(r1, r2, target, use_gpu=use_gpu)  # intentional diagnostic use
        for impact_b, travel_time, branch in sols:
            gamma_b = float(bh._arrival_angle_at_b(b, impact_b))
            d_world = _arrival_direction_at_b_for_pair(a, b, gamma_b, direction, branch)
            d_local = _dir_world_to_local(b, d_world)
            out.append(
                {
                    "direction": int(direction),
                    "target_label": label,
                    "target_azimuth_rad": float(target),
                    "branch": str(branch),
                    "impact_parameter_m": float(impact_b),
                    "travel_time_s": float(travel_time),
                    "gamma_at_b_rad": gamma_b,
                    "dir_at_b_local_xy": np.asarray(d_local, dtype=float),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose all raw observer-direction candidates near a suspicious sky azimuth window."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
        help="Sky interpolation table to inspect.",
    )
    parser.add_argument("--quality", choices=["fast", "medium", "high"], default="fast")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--b-index", type=int, default=-1, help="B index to inspect; -1 means outermost B.")
    parser.add_argument("--phi-center-deg", type=float, default=105.0, help="Absolute source azimuth center.")
    parser.add_argument("--phi-half-width-deg", type=float, default=18.0, help="Half-width around +/- center.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests") / "diagnose_sky_observer_family_candidates.png",
        help="Output image path.",
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = np.load(args.input, allow_pickle=True)
    rs = float(data["rs_m"])
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_points = np.asarray(data["b_points_m"], dtype=float)
    dir_b_plus = np.asarray(data["dir_at_b_plus_local_xy"], dtype=float)
    dir_b_minus = np.asarray(data["dir_at_b_minus_local_xy"], dtype=float)
    ok_plus = np.asarray(data["ok_plus"], dtype=bool)
    ok_minus = np.asarray(data["ok_minus"], dtype=bool)

    b_index = int(args.b_index)
    if b_index < 0:
        b_index = int(np.argmax(np.linalg.norm(b_points[:, :2], axis=1)))
    if b_index < 0 or b_index >= b_points.shape[0]:
        raise IndexError(f"Invalid B index: {b_index}")

    b_point = np.asarray(b_points[b_index], dtype=float)
    a_phi_deg = np.degrees(np.arctan2(a_points[:, 1], a_points[:, 0]))
    center = abs(float(args.phi_center_deg))
    half = abs(float(args.phi_half_width_deg))
    mask = (np.abs(np.abs(a_phi_deg) - center) <= half)
    idx = np.where(mask)[0]
    idx = idx[np.argsort(a_phi_deg[idx])]

    bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs) / C).with_quality(args.quality)
    use_gpu_active = bool(args.use_gpu and cp is not None)

    rows = []
    for k, ai in enumerate(idx, start=1):
        print(f"Inspecting source {k} of {len(idx)} (A index {ai})...")
        candidates = _solve_all_candidates(bh=bh, a_point=a_points[ai], b_point=b_point, use_gpu=use_gpu_active)
        rows.append((int(ai), float(a_phi_deg[ai]), candidates))

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    ax0, ax1 = axes

    xs = []
    yp = []
    ym = []
    for ai, phi_deg, cand in rows:
        xs.append(phi_deg)
        lp = dir_b_plus[b_index, ai]
        lm = dir_b_minus[b_index, ai]
        yp.append(np.degrees(np.arctan2(float(lp[1]), float(lp[0]))) if ok_plus[b_index, ai] else np.nan)
        ym.append(np.degrees(np.arctan2(float(lm[1]), float(lm[0]))) if ok_minus[b_index, ai] else np.nan)
        for c in cand:
            ang = np.degrees(np.arctan2(float(c["dir_at_b_local_xy"][1]), float(c["dir_at_b_local_xy"][0])))
            marker = "o" if c["direction"] == +1 else "s"
            color = "tab:red" if c["branch"] == "monotonic" else "tab:blue"
            ax0.scatter([phi_deg], [ang], c=color, marker=marker, s=36, alpha=0.9)
            ax1.scatter([phi_deg], [float(c["travel_time_s"])], c=color, marker=marker, s=36, alpha=0.9)

    ax0.plot(xs, yp, color="black", lw=1.2, alpha=0.8, label="table plus")
    ax0.plot(xs, ym, color="0.5", lw=1.0, alpha=0.8, label="table minus")
    ax0.set_ylabel("Observer local angle (deg)")
    ax0.grid(alpha=0.22)
    ax0.legend(loc="best")

    ax1.set_xlabel("Source azimuth (deg)")
    ax1.set_ylabel("Travel time (s)")
    ax1.grid(alpha=0.22)

    fig.suptitle(
        f"Raw geodesic candidates near source azimuth +/-{center:.1f} deg for B index {b_index}\n"
        "marker: circle=short target(+1), square=long target(-1) | color: red=monotonic, blue=turning",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")

    for ai, phi_deg, cand in rows:
        print(f"\nA index {ai} | source azimuth {phi_deg:.3f} deg")
        for c in sorted(cand, key=lambda item: (item["target_label"], item["travel_time_s"])):
            ang = np.degrees(np.arctan2(float(c["dir_at_b_local_xy"][1]), float(c["dir_at_b_local_xy"][0])))
            print(
                f"  target={c['target_label']:<5} dir={c['direction']:+d} branch={c['branch']:<9} "
                f"obs_angle={ang:8.3f} deg dt={c['travel_time_s']:.6f} s"
            )

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
