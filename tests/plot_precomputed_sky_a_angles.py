from __future__ import annotations

import argparse
from math import ceil, cos, pi, sin, sqrt
from pathlib import Path
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _unit_xy(vec_xy: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec_xy))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return vec_xy / n


def _direction_from_angle_at_a(a_point: np.ndarray, b_point: np.ndarray, gamma_at_a: float, branch_side: int) -> np.ndarray:
    er = _unit_xy(a_point[:2])
    ephi = np.asarray([-er[1], er[0]], dtype=float)

    th_a = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_b = float(np.arctan2(float(b_point[1]), float(b_point[0])))
    dth_short = ((th_b - th_a + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dth_short >= 0.0 else -1.0
    orient_sign = short_sign if int(branch_side) == +1 else -short_sign

    to_b = _unit_xy(np.asarray(b_point[:2], dtype=float) - np.asarray(a_point[:2], dtype=float))
    d_out = _unit_xy((cos(gamma_at_a)) * er + (orient_sign * sin(gamma_at_a)) * ephi)
    d_in = _unit_xy((-cos(gamma_at_a)) * er + (orient_sign * sin(gamma_at_a)) * ephi)
    align_out = float(np.dot(d_out, to_b))
    align_in = float(np.dot(d_in, to_b))

    r_a = float(np.linalg.norm(a_point[:2]))
    r_b = float(np.linalg.norm(b_point[:2]))
    if r_a < r_b and int(branch_side) == -1:
        return d_in if float(np.dot(d_in, er)) <= float(np.dot(d_out, er)) else d_out

    return d_out if align_out >= align_in else d_in


def _direction_from_angle_at_b(b_point: np.ndarray, a_point: np.ndarray, gamma_at_b: float, branch_side: int) -> np.ndarray:
    er = _unit_xy(b_point[:2])
    ephi = np.asarray([-er[1], er[0]], dtype=float)

    # Match the same branch-orientation convention used by precompute_earliest_grid:
    # build local (x,y) at B with B on +x, then use signed shortest angular change A->B.
    a_xy = np.asarray(a_point[:2], dtype=float)
    x_a = float(np.dot(a_xy, er))
    y_a = float(np.dot(a_xy, ephi))
    a_phi = float(np.arctan2(y_a, x_a))
    dth_short = ((-a_phi + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dth_short >= 0.0 else -1.0
    orient_sign = short_sign if int(branch_side) == +1 else -short_sign

    r_b = float(np.linalg.norm(b_point[:2]))
    r_a = float(np.linalg.norm(a_point[:2]))
    radial_sign = +1.0 if r_a <= r_b else -1.0
    d = (radial_sign * cos(gamma_at_b)) * er + (orient_sign * sin(gamma_at_b)) * ephi
    return _unit_xy(d)


def _local_dir_to_world(point_xy: np.ndarray, local_dir_xy: np.ndarray) -> np.ndarray:
    er = _unit_xy(np.asarray(point_xy[:2], dtype=float))
    ephi = np.asarray([-er[1], er[0]], dtype=float)
    d = float(local_dir_xy[0]) * er + float(local_dir_xy[1]) * ephi
    return _unit_xy(d)


def _panel_layout(n: int) -> Tuple[int, int]:
    ncols = int(ceil(sqrt(n)))
    nrows = int(ceil(n / ncols))
    return nrows, ncols


def _ray_start_distance_beyond_radius(origin_xy: np.ndarray, direction_xy: np.ndarray, min_radius: float) -> float:
    # Find smallest s >= 0 such that ||origin + s*direction|| >= min_radius.
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    r0 = sqrt(ox * ox + oy * oy)
    if r0 >= min_radius:
        return 0.0

    bdotd = ox * dx + oy * dy
    c = ox * ox + oy * oy - (min_radius * min_radius)
    disc = bdotd * bdotd - c
    if disc < 0.0:
        return 0.0
    root = sqrt(disc)
    s1 = -bdotd - root
    s2 = -bdotd + root
    candidates = [s for s in (s1, s2) if s >= 0.0]
    if not candidates:
        return 0.0
    return min(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sky-table precomputed A-angle direction fields (+1/-1) for selected B radial positions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
        help="Path to sky precomputed .npz data.",
    )
    parser.add_argument(
        "--vector-length-rs",
        type=float,
        default=30.0,
        help="Vector length in units of rs (default: 30.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests") / "precomputed_sky_a_angle_panels.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display figure interactively when backend supports it (default: true).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        fallback_paths = [
            Path("earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz"),
            Path("tests") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
        ]
        found = next((p for p in fallback_paths if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Input file not found: {args.input}")
        print(f"Input not found at {args.input}; using fallback: {found}")
        args.input = found

    data = np.load(args.input, allow_pickle=True)
    required = [
        "rs_m",
        "a_points_m",
        "b_points_m",
        "dir_at_a_plus_local_xy",
        "dir_at_a_minus_local_xy",
        "dir_at_b_plus_local_xy",
        "dir_at_b_minus_local_xy",
        "ok_plus",
        "ok_minus",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required arrays in npz: {missing}")

    rs = float(data["rs_m"])
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_points = np.asarray(data["b_points_m"], dtype=float)
    dir_a_plus = np.asarray(data["dir_at_a_plus_local_xy"], dtype=float)
    dir_a_minus = np.asarray(data["dir_at_a_minus_local_xy"], dtype=float)
    dir_b_plus = np.asarray(data["dir_at_b_plus_local_xy"], dtype=float)
    dir_b_minus = np.asarray(data["dir_at_b_minus_local_xy"], dtype=float)
    ok_plus = np.asarray(data["ok_plus"], dtype=bool)
    ok_minus = np.asarray(data["ok_minus"], dtype=bool)

    n_b, n_a = ok_plus.shape
    if a_points.shape[0] != n_a or b_points.shape[0] != n_b:
        raise ValueError("Array shape mismatch between points and solution tables.")
    if n_b <= 0:
        raise ValueError("No B positions found in precompute table.")
    if n_b == 1:
        selected_b = [0]
    elif n_b == 2:
        selected_b = [0, 1]
    elif n_b == 3:
        selected_b = [0, 1, 2]
    else:
        selected_b = sorted(set([0, n_b // 3, (2 * n_b) // 3, n_b - 1]))

    vec_len = float(args.vector_length_rs) * rs
    r_max = 0.0
    sky_radius = 0.0
    if a_points.size > 0:
        sky_radius = float(np.max(np.linalg.norm(a_points[:, :2], axis=1)))
        r_max = max(r_max, sky_radius)
    if b_points.size > 0:
        r_max = max(r_max, float(np.max(np.linalg.norm(b_points[:, :2], axis=1))))
    lim = 1.12 * (r_max + vec_len)

    nrows, ncols = _panel_layout(len(selected_b))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    axes_list = np.asarray(axes).ravel().tolist()

    t = np.linspace(0.0, 2.0 * pi, 241)
    hole_x = rs * np.cos(t)
    hole_y = rs * np.sin(t)
    total_panels = len(selected_b)

    for panel_i, b_i in enumerate(selected_b):
        print(f"Plotting panel {panel_i + 1}/{total_panels} (B index {b_i})...")
        ax = axes_list[panel_i]
        ax.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.45", alpha=0.65)
        ax.plot(hole_x, hole_y, "k-", lw=1.2)

        bx, by = float(b_points[b_i, 0]), float(b_points[b_i, 1])
        r_b = sqrt(bx * bx + by * by)
        for a_i in range(n_a):
            a = a_points[a_i]
            axx, axy = float(a[0]), float(a[1])
            has_a_plus_dir = np.all(np.isfinite(dir_a_plus[b_i, a_i]))
            has_b_plus_dir = np.all(np.isfinite(dir_b_plus[b_i, a_i]))
            has_a_minus_dir = np.all(np.isfinite(dir_a_minus[b_i, a_i]))
            has_b_minus_dir = np.all(np.isfinite(dir_b_minus[b_i, a_i]))

            if ok_plus[b_i, a_i] and has_a_plus_dir:
                d = _local_dir_to_world(a, dir_a_plus[b_i, a_i])
                ax.plot([axx, axx + vec_len * d[0]], [axy, axy + vec_len * d[1]], color="red", lw=1.0, alpha=0.9)

                if has_b_plus_dir:
                    d_obs = _local_dir_to_world(b_points[b_i], dir_b_plus[b_i, a_i])
                else:
                    d_obs = None
                if d_obs is not None:
                    d_view = -d_obs
                    hx = bx + vec_len * float(d_view[0])
                    hy = by + vec_len * float(d_view[1])
                    ax.plot([bx, hx], [by, hy], color="red", lw=1.0, alpha=0.9)
                    ax.plot([hx, axx], [hy, axy], color="black", lw=0.8, ls="--", alpha=0.7)
            else:
                ax.scatter([axx], [axy], c="red", s=16, alpha=0.9)
            if ok_minus[b_i, a_i] and has_a_minus_dir:
                d = _local_dir_to_world(a, dir_a_minus[b_i, a_i])
                ax.plot([axx, axx + vec_len * d[0]], [axy, axy + vec_len * d[1]], color="blue", lw=1.0, alpha=0.9)
                if has_b_minus_dir:
                    d_obs = _local_dir_to_world(b_points[b_i], dir_b_minus[b_i, a_i])
                else:
                    d_obs = None
                if d_obs is not None:
                    d_view = -d_obs
                    tail_offset = _ray_start_distance_beyond_radius(
                        b_points[b_i][:2], d_view, sky_radius + 0.05 * vec_len
                    )
                    tx = bx + tail_offset * float(d_view[0])
                    ty = by + tail_offset * float(d_view[1])
                    hx = tx + vec_len * float(d_view[0])
                    hy = ty + vec_len * float(d_view[1])
                    ax.plot([tx, hx], [ty, hy], color="blue", lw=1.0, alpha=0.9)
                    ax.plot([tx, axx], [ty, axy], color="black", lw=0.8, ls="--", alpha=0.7)
            else:
                ax.scatter([axx], [axy], c="blue", s=28, alpha=0.9)

        ax.scatter([bx], [by], c="green", s=64, marker="x", linewidths=2.2, zorder=10)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"B index {b_i} | r_B/rs={r_b/rs:.3f}", fontsize=9)

    for ax in axes_list[len(selected_b):]:
        ax.axis("off")

    fig.suptitle(
        "Sky-table precomputed A-angle direction field by selected B radial positions\n"
        "red=+1 vectors, blue=-1 vectors (+ and viewer), black dashed=source linkage",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved figure: {args.output}")
    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
