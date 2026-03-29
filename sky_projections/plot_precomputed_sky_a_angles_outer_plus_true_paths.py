from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys

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


def _local_dir_to_world(point_xy: np.ndarray, local_dir_xy: np.ndarray) -> np.ndarray:
    er = _unit_xy(np.asarray(point_xy[:2], dtype=float))
    ephi = np.asarray([-er[1], er[0]], dtype=float)
    d = float(local_dir_xy[0]) * er + float(local_dir_xy[1]) * ephi
    return _unit_xy(d)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot saved true +family paths from sky-table: outermost B only."
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
        "--max-paths",
        type=int,
        default=20,
        help="Maximum number of + paths to solve/plot via even subsampling (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sky_projections") / "precomputed_sky_a_angle_outermost_plus_true_paths.png",
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
            Path("sky_projections") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
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
        "dir_at_b_plus_local_xy",
        "ok_plus",
        "true_path_plus_xy_m",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required arrays in npz: {missing}")

    rs = float(data["rs_m"])
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_points = np.asarray(data["b_points_m"], dtype=float)
    dir_a_plus = np.asarray(data["dir_at_a_plus_local_xy"], dtype=float)
    dir_b_plus = np.asarray(data["dir_at_b_plus_local_xy"], dtype=float)
    ok_plus = np.asarray(data["ok_plus"], dtype=bool)

    n_b, n_a = ok_plus.shape
    if a_points.shape[0] != n_a or b_points.shape[0] != n_b:
        raise ValueError("Array shape mismatch between points and solution tables.")

    b_r = np.linalg.norm(b_points[:, :2], axis=1)
    b_i = int(np.argmax(b_r))
    b_point = np.asarray(b_points[b_i], dtype=float)

    vec_len = float(args.vector_length_rs) * rs
    r_max = max(
        float(np.max(np.linalg.norm(a_points[:, :2], axis=1))) if a_points.size > 0 else 0.0,
        float(np.max(np.linalg.norm(b_points[:, :2], axis=1))) if b_points.size > 0 else 0.0,
    )
    lim = 1.12 * (r_max + vec_len)

    valid_idx = [
        i for i in range(n_a) if bool(ok_plus[b_i, i]) and np.all(np.isfinite(dir_a_plus[b_i, i])) and np.all(np.isfinite(dir_b_plus[b_i, i]))
    ]
    if int(args.max_paths) > 0 and len(valid_idx) > int(args.max_paths):
        sel = np.linspace(0, len(valid_idx) - 1, int(args.max_paths), dtype=int)
        valid_idx = [valid_idx[j] for j in sel]
    true_path_plus = np.asarray(data["true_path_plus_xy_m"], dtype=float)
    total_pairs = len(valid_idx)
    for k, _a_i in enumerate(valid_idx, start=1):
        print(f"Loading saved path {k} of {total_pairs}...")

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0))

    t = np.linspace(0.0, 2.0 * pi, 241)
    ax.plot(rs * np.cos(t), rs * np.sin(t), "k-", lw=1.2)
    ax.plot(1.5 * rs * np.cos(t), 1.5 * rs * np.sin(t), "k--", lw=1.0, alpha=0.9)
    ax.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.45", alpha=0.45)

    bx, by = float(b_point[0]), float(b_point[1])
    ax.scatter([bx], [by], c="green", s=72, marker="x", linewidths=2.2, zorder=10)

    plotted_curves = 0
    plotted_samples = 0
    for local_i, a_i in enumerate(valid_idx):
        a = np.asarray(a_points[a_i], dtype=float)
        axx, axy = float(a[0]), float(a[1])

        d_src = _local_dir_to_world(a, dir_a_plus[b_i, a_i])
        ax.plot([axx, axx + vec_len * d_src[0]], [axy, axy + vec_len * d_src[1]], color="blue", lw=1.0, alpha=0.85)

        d_obs = _local_dir_to_world(b_point, dir_b_plus[b_i, a_i])
        d_view = -d_obs
        ax.plot([bx, bx + vec_len * d_view[0]], [by, by + vec_len * d_view[1]], color="red", lw=1.0, alpha=0.45)

        path_xy = np.asarray(true_path_plus[b_i, a_i, :, :], dtype=float)
        if path_xy.ndim != 2 or path_xy.shape[1] != 2:
            continue
        mask = np.all(np.isfinite(path_xy), axis=1)
        if int(np.count_nonzero(mask)) < 2:
            continue
        # Plot raw saved sample points so spacing directly reflects solver output.
        ax.scatter(path_xy[mask, 0], path_xy[mask, 1], s=4.0, c="black", alpha=0.42, linewidths=0.0)
        plotted_curves += 1
        plotted_samples += int(np.count_nonzero(mask))

    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Outermost B only (index {b_i}, r_B/rs={float(b_r[b_i]/rs):.3f}) | + branch only\n"
        f"blue=source vectors, red=observer vectors, black=raw saved solver samples | "
        f"curves={plotted_curves}/{len(valid_idx)} samples={plotted_samples}",
        fontsize=10,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")
    print(
        f"B index={b_i}, valid_plus={len(valid_idx)}, curves_plotted={plotted_curves}, "
        f"sample_points_plotted={plotted_samples}, path_source=saved"
    )

    if args.show and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
