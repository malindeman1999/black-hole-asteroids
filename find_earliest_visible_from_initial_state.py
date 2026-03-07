from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole
from initial_state_visibility import TimelikeVisibilitySession


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Earliest visibility solve from a timelike body defined by initial position/velocity."
        )
    )
    parser.add_argument("--quality", choices=["high", "medium", "fast"], default="fast")
    parser.add_argument(
        "--integrator",
        choices=["symplectic", "rk4", "euler"],
        default="symplectic",
        help="Timelike trajectory integrator (default: symplectic).",
    )
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tmin", type=float, default=-20.0, help="Earliest emission time to search.")
    parser.add_argument("--tmax", type=float, default=20.0, help="Latest emission time to search.")
    parser.add_argument("--t0", type=float, default=6.0, help="Observer coordinate time.")
    parser.add_argument("--dtau", type=float, default=1e-3, help="Proper-time step for mass trajectory integration.")
    parser.add_argument("--root-max-iter", type=int, default=12)
    parser.add_argument("--root-tol-time", type=float, default=1e-6)
    parser.add_argument("--scan-samples-fallback", type=int, default=65)

    parser.add_argument("--x0-rs", type=float, default=10.0)
    parser.add_argument("--y0-rs", type=float, default=0.0)
    parser.add_argument("--z0-rs", type=float, default=0.0)
    parser.add_argument("--vx-frac-c", type=float, default=0.0)
    parser.add_argument("--vy-frac-c", type=float, default=0.20)
    parser.add_argument("--vz-frac-c", type=float, default=0.0)

    parser.add_argument("--b-x-rs", type=float, default=2.0)
    parser.add_argument("--b-y-rs", type=float, default=0.0)
    parser.add_argument("--b-z-rs", type=float, default=0.0)
    args = parser.parse_args()

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = bh.schwarzschild_radius_m

    initial_position = (float(args.x0_rs) * rs, float(args.y0_rs) * rs, float(args.z0_rs) * rs)
    initial_velocity = (
        float(args.vx_frac_c) * C,
        float(args.vy_frac_c) * C,
        float(args.vz_frac_c) * C,
    )
    observer_b = (float(args.b_x_rs) * rs, float(args.b_y_rs) * rs, float(args.b_z_rs) * rs)

    session = TimelikeVisibilitySession(
        bh=bh,
        initial_position_m=initial_position,
        initial_velocity_m_s=initial_velocity,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        proper_time_step_s=float(args.dtau),
        integrator=str(args.integrator),
        use_gpu=bool(args.use_gpu),
    )
    result = session.solve(
        observer_point_b=observer_b,
        observer_time_s=float(args.t0),
        root_max_iter=int(args.root_max_iter),
        root_tol_time=float(args.root_tol_time),
        fallback_scan_samples=int(args.scan_samples_fallback),
    )

    print(
        f"initial_pos_rs=({args.x0_rs:.3f},{args.y0_rs:.3f},{args.z0_rs:.3f}) "
        f"initial_vel_c=({args.vx_frac_c:.3f},{args.vy_frac_c:.3f},{args.vz_frac_c:.3f})"
    )
    print(f"observer_B_rs=({args.b_x_rs:.3f},{args.b_y_rs:.3f},{args.b_z_rs:.3f}) t0={float(args.t0):.6f}s")
    for side in ("plus", "minus", "earliest"):
        obs = result.get(side)
        if not isinstance(obs, dict):
            print(f"{side}: none")
            continue
        pa = obs["point_a_m"]
        print(
            f"{side}: te={obs['emission_time_s']:.9f}s dt={obs['travel_time_s']:.9f}s "
            f"A=({pa[0]:.6e},{pa[1]:.6e},{pa[2]:.6e})"
        )


if __name__ == "__main__":
    main()
