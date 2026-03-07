from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole, TimelikeTrajectoryResult


def _build_initial_velocity(speed_m_s: float, direction_deg: float) -> Tuple[float, float, float]:
    th = direction_deg * pi / 180.0
    # At x>0 on the x-axis: radial outward is +x, tangential is +y.
    return (speed_m_s * cos(th), speed_m_s * sin(th), 0.0)


def _run_case(
    bh: SchwarzschildBlackHole,
    initial_position_m: Tuple[float, float, float],
    initial_velocity_m_s: Tuple[float, float, float],
    dtau: float,
    max_tau: float,
    max_steps: int,
    escape_radius_m: float,
    integrator: str,
) -> TimelikeTrajectoryResult:
    return bh.integrate_timelike_trajectory(
        initial_position_m=initial_position_m,
        initial_velocity_m_s=initial_velocity_m_s,
        proper_time_step_s=dtau,
        max_proper_time_s=max_tau,
        max_steps=max_steps,
        escape_radius_m=escape_radius_m,
        integrator=integrator,
    )


def _local_circular_speed_schwarzschild(r_m: float, rs_m: float) -> float:
    if r_m <= rs_m:
        raise ValueError("Circular orbit requires r > rs.")
    # Local physical circular speed measured by a static observer.
    return C * (rs_m / (2.0 * (r_m - rs_m))) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot several timelike trajectories around a Schwarzschild black hole.\n"
            "All trajectories start at r = 5 light-seconds with different launch directions."
        )
    )
    parser.add_argument(
        "--diameter-light-seconds",
        type=float,
        default=1.0,
        help="Black-hole Schwarzschild diameter in light-seconds (default: 1.0).",
    )
    parser.add_argument(
        "--start-radius-light-seconds",
        type=float,
        default=5.0,
        help="Initial radius for all trajectories in light-seconds (default: 5.0).",
    )
    parser.add_argument(
        "--speed-frac-c",
        type=float,
        default=0.20,
        help="Initial speed magnitude as fraction of c (default: 0.20).",
    )
    parser.add_argument("--dtau", type=float, default=1e-3, help="Proper-time step in seconds.")
    parser.add_argument("--max-tau", type=float, default=25.0, help="Max proper time in seconds.")
    parser.add_argument("--max-steps", type=int, default=300000, help="Max integration steps.")
    parser.add_argument(
        "--integrator",
        choices=["symplectic", "rk4", "euler"],
        default="symplectic",
        help="Trajectory integration method (default: symplectic).",
    )
    parser.add_argument(
        "--escape-radius-rs",
        type=float,
        default=80.0,
        help="Escape threshold radius in units of rs (default: 80).",
    )
    args = parser.parse_args()

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(args.diameter_light_seconds).with_quality("fast")
    rs = bh.schwarzschild_radius_m
    r0 = args.start_radius_light_seconds * C
    speed = max(0.0, min(float(args.speed_frac_c), 0.999)) * C

    # Launch directions in the local static frame at (r0, 0, 0):
    # 0 deg = radial outward, 90 deg = tangential, 180 deg = radial inward.
    cases = [
        ("Outward (0 deg)", 0.0),
        ("Mostly Tangential (70 deg)", 70.0),
        ("Tangential (90 deg)", 90.0),
        ("Inward-angled (140 deg)", 140.0),
        ("Inward (180 deg)", 180.0),
    ]

    initial_position = (r0, 0.0, 0.0)
    escape_radius_m = float(args.escape_radius_rs) * rs
    results: List[Tuple[str, TimelikeTrajectoryResult]] = []

    # Add one explicitly circular case and integrate for 2 full orbits in azimuth.
    v_circ = _local_circular_speed_schwarzschild(r0, rs)
    gamma_circ = 1.0 / (1.0 - (v_circ / C) ** 2) ** 0.5
    dphi_dtau_circ = gamma_circ * v_circ / r0
    tau_for_two_orbits = (4.0 * pi) / dphi_dtau_circ
    circular_max_tau = tau_for_two_orbits
    circular_dtau = min(float(args.dtau), tau_for_two_orbits / 20_000.0)
    out_circ = _run_case(
        bh=bh,
        initial_position_m=initial_position,
        initial_velocity_m_s=(0.0, v_circ, 0.0),
        dtau=circular_dtau,
        max_tau=circular_max_tau,
        max_steps=max(int(args.max_steps), int(circular_max_tau / circular_dtau) + 10),
        escape_radius_m=escape_radius_m,
        integrator=str(args.integrator),
    )
    results.append(("Circular (90 deg @ v_circ, 2 orbits target)", out_circ))

    for label, deg in cases:
        v0 = _build_initial_velocity(speed, deg)
        out = _run_case(
            bh=bh,
            initial_position_m=initial_position,
            initial_velocity_m_s=v0,
            dtau=float(args.dtau),
            max_tau=float(args.max_tau),
            max_steps=int(args.max_steps),
            escape_radius_m=escape_radius_m,
            integrator=str(args.integrator),
        )
        results.append((label, out))

    fig, ax = plt.subplots(figsize=(9.5, 9.5))

    # Horizon and photon sphere in the orbital plane.
    t = [2.0 * pi * i / 600 for i in range(601)]
    rph = bh.photon_sphere_radius_m
    ax.fill([rs * cos(v) for v in t], [rs * sin(v) for v in t], color="black", alpha=0.22, label="Event horizon")
    ax.plot([rs * cos(v) for v in t], [rs * sin(v) for v in t], color="black", lw=1.2)
    ax.plot([rph * cos(v) for v in t], [rph * sin(v) for v in t], "k--", lw=1.0, label="Photon sphere")

    colors = ["tab:cyan", "tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]
    r_plot_max = max(r0, rs * 4.0)
    for i, (label, out) in enumerate(results):
        if not out.samples:
            continue
        xs = [s.position_xyz_m[0] for s in out.samples]
        ys = [s.position_xyz_m[1] for s in out.samples]
        r_plot_max = max(r_plot_max, max((x * x + y * y) ** 0.5 for x, y in zip(xs, ys)))
        status_label = f"{label}: {out.status}"
        ax.plot(xs, ys, color=colors[i % len(colors)], lw=1.5, label=status_label)
        ax.scatter([xs[0]], [ys[0]], color=colors[i % len(colors)], s=22)
        ax.scatter([xs[-1]], [ys[-1]], color=colors[i % len(colors)], s=26, marker="x")

        if "Circular" in label and out.samples:
            phi_span = out.samples[-1].azimuth_rad - out.samples[0].azimuth_rad
            r_vals = [s.radius_m for s in out.samples]
            r_mean = sum(r_vals) / len(r_vals)
            r_drift = max(abs(r - r_mean) for r in r_vals)
            print(
                "circular-check: "
                f"phi_span={phi_span:.6f} rad ({phi_span/(2*pi):.6f} orbits), "
                f"mean_r/rs={r_mean/rs:.9f}, max_abs_radial_drift_m={r_drift:.6e}"
            )

    lim = 1.1 * max(r_plot_max, rs * 2.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Timelike trajectories in Schwarzschild coordinates\n"
        f"r0={args.start_radius_light_seconds:.2f} light-seconds, |v0|={args.speed_frac_c:.3f} c, "
        f"rs={rs/C:.3f} light-seconds, integrator={args.integrator}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
