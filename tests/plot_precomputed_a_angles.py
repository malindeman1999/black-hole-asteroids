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
    # Local basis at A in the orbital plane.
    er = _unit_xy(a_point[:2])
    ephi = np.asarray([-er[1], er[0]], dtype=float)

    # Map (+1 short / -1 long branch) to the actual global angular orientation
    # in this A->B geometry. This keeps upper/lower-half solutions antisymmetric.
    th_a = float(np.arctan2(float(a_point[1]), float(a_point[0])))
    th_b = float(np.arctan2(float(b_point[1]), float(b_point[0])))
    dth_short = ((th_b - th_a + pi) % (2.0 * pi)) - pi
    short_sign = +1.0 if dth_short >= 0.0 else -1.0
    orient_sign = short_sign if int(branch_side) == +1 else -short_sign

    # gamma_at_a is measured from local radial direction. Resolve the remaining
    # inward/outward ambiguity by choosing the candidate more aligned with A->B.
    to_b = _unit_xy(np.asarray(b_point[:2], dtype=float) - np.asarray(a_point[:2], dtype=float))
    d_out = _unit_xy((cos(gamma_at_a)) * er + (orient_sign * sin(gamma_at_a)) * ephi)
    d_in = _unit_xy((-cos(gamma_at_a)) * er + (orient_sign * sin(gamma_at_a)) * ephi)
    align_out = float(np.dot(d_out, to_b))
    align_in = float(np.dot(d_in, to_b))

    r_a = float(np.linalg.norm(a_point[:2]))
    r_b = float(np.linalg.norm(b_point[:2]))
    if r_a < r_b and int(branch_side) == -1:
        # For inner A -> outer B, the long branch should initially head inward
        # toward a near-hole turning path, not directly outward toward B.
        return d_in if float(np.dot(d_in, er)) <= float(np.dot(d_out, er)) else d_out

    return d_out if align_out >= align_in else d_in


def _panel_layout(n: int) -> Tuple[int, int]:
    ncols = int(ceil(sqrt(n)))
    nrows = int(ceil(n / ncols))
    return nrows, ncols


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot precomputed A-angle direction fields (+1/-1) for each B radial position "
            "from an earliest-angle precompute .npz file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz data.",
    )
    parser.add_argument(
        "--vector-length-rs",
        type=float,
        default=0.5,
        help="Vector length in units of rs (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests") / "precomputed_a_angle_panels.png",
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
            Path("earliest_angles_precompute_10rs.npz"),
            Path("tests") / "earliest_angles_precompute_10rs.npz",
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
        "gamma_at_a_plus_rad",
        "gamma_at_a_minus_rad",
        "ok_plus",
        "ok_minus",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required arrays in npz: {missing}")

    rs = float(data["rs_m"])
    a_points = np.asarray(data["a_points_m"], dtype=float)
    b_points = np.asarray(data["b_points_m"], dtype=float)
    gamma_a_plus = np.asarray(data["gamma_at_a_plus_rad"], dtype=float)
    gamma_a_minus = np.asarray(data["gamma_at_a_minus_rad"], dtype=float)
    ok_plus = np.asarray(data["ok_plus"], dtype=bool)
    ok_minus = np.asarray(data["ok_minus"], dtype=bool)

    n_b, n_a = ok_plus.shape
    if a_points.shape[0] != n_a or b_points.shape[0] != n_b:
        raise ValueError("Array shape mismatch between points and solution tables.")

    vec_len = float(args.vector_length_rs) * rs

    r_max = 0.0
    if a_points.size > 0:
        r_max = max(r_max, float(np.max(np.linalg.norm(a_points[:, :2], axis=1))))
    if b_points.size > 0:
        r_max = max(r_max, float(np.max(np.linalg.norm(b_points[:, :2], axis=1))))
    lim = 1.12 * (r_max + vec_len)

    nrows, ncols = _panel_layout(n_b)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    axes_list = np.asarray(axes).ravel().tolist()

    t = np.linspace(0.0, 2.0 * pi, 241)
    hole_x = rs * np.cos(t)
    hole_y = rs * np.sin(t)

    for b_i in range(n_b):
        ax = axes_list[b_i]

        # Mark all A points.
        ax.scatter(a_points[:, 0], a_points[:, 1], s=8, c="0.45", alpha=0.65)

        # Event horizon circle.
        ax.plot(hole_x, hole_y, "k-", lw=1.2)

        # Mark B position for this panel.
        bx, by = float(b_points[b_i, 0]), float(b_points[b_i, 1])
        r_b = sqrt(bx * bx + by * by)
        ax.scatter([bx], [by], c="green", s=40, marker="x", linewidths=1.6)

        for a_i in range(n_a):
            a = a_points[a_i]
            axx, axy = float(a[0]), float(a[1])

            if ok_plus[b_i, a_i] and np.isfinite(gamma_a_plus[b_i, a_i]):
                d = _direction_from_angle_at_a(a, b_points[b_i], float(gamma_a_plus[b_i, a_i]), branch_side=+1)
                ax.plot([axx, axx + vec_len * d[0]], [axy, axy + vec_len * d[1]], color="red", lw=1.0, alpha=0.9)
            else:
                ax.scatter([axx], [axy], c="red", s=16, alpha=0.9)

            if ok_minus[b_i, a_i] and np.isfinite(gamma_a_minus[b_i, a_i]):
                d = _direction_from_angle_at_a(a, b_points[b_i], float(gamma_a_minus[b_i, a_i]), branch_side=-1)
                ax.plot([axx, axx + vec_len * d[0]], [axy, axy + vec_len * d[1]], color="blue", lw=1.0, alpha=0.9)
            else:
                ax.scatter([axx], [axy], c="blue", s=28, alpha=0.9)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"B index {b_i} | r_B/rs={r_b/rs:.3f}", fontsize=9)

    for ax in axes_list[n_b:]:
        ax.axis("off")

    fig.suptitle(
        "Precomputed A-angle direction field by B radial position\n"
        "red=+1 vectors, blue=-1 vectors, red/blue dots indicate missing solutions",
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
