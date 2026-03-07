from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import cp
from earliest_visible_interpolated_session import EarliestVisibleInterpolatedSession, SampledTrajectory3D


def _build_sampled_trajectory_callable(rs: float, tmin: float, tmax: float):
    # Smooth deterministic trajectory around the hole.
    ts = np.linspace(tmin, tmax, 2049, dtype=float)
    radius = 7.0 * rs
    omega = 0.12
    phase = omega * ts
    xs = radius * np.cos(phase)
    ys = radius * np.sin(phase)
    zs = 0.12 * radius * np.sin(0.37 * phase)

    def trajectory(t: float):
        tt = float(np.clip(t, tmin, tmax))
        x = float(np.interp(tt, ts, xs))
        y = float(np.interp(tt, ts, ys))
        z = float(np.interp(tt, ts, zs))
        return (x, y, z)

    return trajectory


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find earliest visibility from B at time t0 using precomputed interpolation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Path to precomputed .npz table.",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU interpolation when CuPy is available (default: true).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Interpolation batch size (default: 5000).",
    )
    parser.add_argument(
        "--gpu-min-batch",
        type=int,
        default=256,
        help="Use CPU for interpolation calls smaller than this size to reduce GPU overhead (default: 256).",
    )
    parser.add_argument("--tmin", type=float, default=-30.0, help="Trajectory start time (default: -30).")
    parser.add_argument("--tmax", type=float, default=0.0, help="Trajectory end time (default: 0).")
    parser.add_argument("--t0", type=float, default=0.0, help="Observer time at B (default: 0).")
    parser.add_argument(
        "--scan-samples",
        type=int,
        default=257,
        help="Samples for bracket scan (default: 257).",
    )
    parser.add_argument(
        "--root-max-iter",
        type=int,
        default=36,
        help="Root bisection iterations (default: 36).",
    )
    parser.add_argument(
        "--root-tol-time",
        type=float,
        default=1e-6,
        help="Root time tolerance in seconds (default: 1e-6).",
    )
    parser.add_argument(
        "--b-radius-rs",
        type=float,
        default=8.0,
        help="Observer radius in rs units (default: 8.0).",
    )
    parser.add_argument(
        "--b-phi-rad",
        type=float,
        default=0.0,
        help="Observer azimuth angle in radians in the x-y plane (default: 0).",
    )
    parser.add_argument(
        "--b-z-scale",
        type=float,
        default=0.0,
        help="Observer z = b_z_scale * b_radius (default: 0).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)

    # Build and cache trajectory once.
    # The session loads/interpolator tables once and keeps backend arrays resident.
    # Repeated solve() calls for new (B, t0) avoid re-uploading large precompute tables.
    from precompute_earliest_grid import PrecomputedEarliestInterpolator

    tmp_interp = PrecomputedEarliestInterpolator.from_npz(input_path)
    rs = tmp_interp.rs_m
    traj_callable = _build_sampled_trajectory_callable(rs=rs, tmin=float(args.tmin), tmax=float(args.tmax))
    sampled_traj = SampledTrajectory3D.from_callable(
        trajectory=traj_callable,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        samples=max(2049, 4 * max(3, int(args.scan_samples))),
    )

    session = EarliestVisibleInterpolatedSession(
        precompute_npz=input_path,
        sampled_trajectory=sampled_traj,
        use_gpu=bool(args.use_gpu),
        batch_size=int(args.batch_size),
        gpu_min_batch=int(args.gpu_min_batch),
    )

    rb = float(args.b_radius_rs) * rs
    point_b = (
        rb * np.cos(float(args.b_phi_rad)),
        rb * np.sin(float(args.b_phi_rad)),
        float(args.b_z_scale) * rb,
    )

    use_gpu_active = bool(args.use_gpu and cp is not None)
    mode = "GPU-batch" if use_gpu_active else "CPU-batch"
    print(f"Mode: {mode}")
    print(f"B = ({point_b[0]:.6e}, {point_b[1]:.6e}, {point_b[2]:.6e}) m")

    result = session.solve(
        point_b=point_b,
        t0=float(args.t0),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        scan_samples=max(3, int(args.scan_samples)),
        root_max_iter=int(args.root_max_iter),
        root_tol_time=float(args.root_tol_time),
    )

    print(
        f"Window: t in [{float(args.tmin):.3f}, {min(float(args.tmax), float(args.t0)):.3f}], "
        f"observer t0={float(args.t0):.3f}, scan_samples={max(3, int(args.scan_samples))}"
    )

    def _print_side(label: str, obs) -> None:
        if obs is None:
            print(f"{label}: no root in window")
            return
        print(
            f"{label}: te={obs['emission_time_s']:.9f} s, dt={obs['travel_time_s']:.9f} s, "
            f"arr={obs['arrival_time_s']:.9f} s, gamma_B={obs['gamma_at_b_rad']:.9f} rad"
        )

    _print_side("plus", result["plus"])
    _print_side("minus", result["minus"])


if __name__ == "__main__":
    main()
