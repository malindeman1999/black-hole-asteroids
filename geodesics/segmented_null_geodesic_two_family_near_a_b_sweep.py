from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geodesics.segmented_null_geodesic_two_family_sweep import (
    _build_b_radii_rs,
    _build_black_hole,
    solve_b_radial_series,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Segmented null-path sweep where A radius also steps through the same radial grid as B. "
            "For each A radius in that grid, solve a full B-radial sweep over A-phi."
        )
    )
    p.add_argument(
        "--rs-m",
        type=float,
        default=1.0,
        help=(
            "Schwarzschild radius in meters (legacy input). "
            "A SchwarzschildBlackHole object is still built from this value."
        ),
    )
    p.add_argument(
        "--diameter-light-seconds",
        type=float,
        default=None,
        help=(
            "Black-hole diameter in light-seconds (preferred project-wide input). "
            "If provided, overrides --rs-m."
        ),
    )
    p.add_argument("--b-r-min-rs", type=float, default=1.6, help="Min B radius in Rs.")
    p.add_argument("--b-r-max-rs", type=float, default=10.0, help="Max B radius in Rs.")
    p.add_argument("--b-r-count", type=int, default=12, help="Number of radii in the shared A/B grid.")
    p.add_argument(
        "--b-spacing",
        choices=["r3", "linear"],
        default="r3",
        help="Shared radial spacing for A and B grids.",
    )
    p.add_argument("--a-phi-count", type=int, default=97, help="A angles around ring.")
    p.add_argument(
        "--a-phi-second-rad",
        type=float,
        default=None,
        help=(
            "Optional explicit second A angle (radians) used for debug continuation. "
            "When set, solver uses exactly two A paths: 0 and this value."
        ),
    )
    p.add_argument(
        "--a-phi-step-rad",
        type=float,
        default=None,
        help=(
            "Optional A-angle step in radians. When set (and --a-phi-second-rad is not set), "
            "A angles are generated as i*step for i=0..a-phi-count-1."
        ),
    )
    p.add_argument("--node-count", type=int, default=20, help="Nodes along A->B path.")
    p.add_argument(
        "--node-spacing",
        choices=["r3", "linear", "log"],
        default="r3",
        help="Path node radial spacing (same meaning as base segmented sweep).",
    )
    p.add_argument("--optimizer", choices=["scipy", "coord"], default="scipy")
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--opt-ftol", type=float, default=1e-9)
    p.add_argument("--opt-gtol", type=float, default=1e-6)
    p.add_argument(
        "--adaptive-iter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt per-angle max iterations from previous continuation solve (default: on).",
    )
    p.add_argument(
        "--fail-fast-series",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop A-phi continuation immediately after first failed path in a ring (default: on).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="B-ring worker threads inside each fixed-A sweep. 0=auto, 1=serial, >1=parallel.",
    )
    p.add_argument("--debug-show-rings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug-pause-rings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "segmented_null_geodesic_two_family_near_a_b_sweep.npz",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    bh = _build_black_hole(args)
    rs_m = float(bh.schwarzschild_radius_m)
    if int(args.b_r_count) < 1:
        raise ValueError("--b-r-count must be >= 1")
    if float(args.b_r_max_rs) <= float(args.b_r_min_rs):
        raise ValueError("--b-r-max-rs must be > --b-r-min-rs")

    shared_radii_rs = _build_b_radii_rs(
        r_min_rs=float(args.b_r_min_rs),
        r_max_rs=float(args.b_r_max_rs),
        n=int(args.b_r_count),
        spacing=str(args.b_spacing),
    )
    shared_radii_m = shared_radii_rs * rs_m

    first = solve_b_radial_series(
        rs_m=rs_m,
        a_radius_m=float(shared_radii_m[0]),
        b_radii_m=shared_radii_m,
        a_phi_count=int(args.a_phi_count),
        a_phi_second_rad=(None if args.a_phi_second_rad is None else float(args.a_phi_second_rad)),
        a_phi_step_rad=(None if args.a_phi_step_rad is None else float(args.a_phi_step_rad)),
        node_count=int(args.node_count),
        node_spacing=str(args.node_spacing),
        optimizer=str(args.optimizer),
        max_iter=int(args.max_iter),
        opt_ftol=float(args.opt_ftol),
        opt_gtol=float(args.opt_gtol),
        adaptive_iter=bool(args.adaptive_iter),
        fail_fast_series=bool(args.fail_fast_series),
        workers=int(args.workers),
        debug_show_rings=bool(args.debug_show_rings),
        debug_pause_rings=bool(args.debug_pause_rings),
    )

    n_a_r = int(shared_radii_m.size)
    n_b = int(first["b_radii_m"].size)
    n_a_phi = int(first["a_phi_rad"].size)
    n_nodes = int(first["path_plus_xy_m"].shape[2])

    path_plus = np.full((n_a_r, n_b, n_a_phi, n_nodes, 2), np.nan, dtype=float)
    path_minus = np.full_like(path_plus, np.nan)
    t_plus = np.full((n_a_r, n_b, n_a_phi), np.nan, dtype=float)
    t_minus = np.full((n_a_r, n_b, n_a_phi), np.nan, dtype=float)
    tau_plus = np.full((n_a_r, n_b, n_a_phi), np.nan, dtype=float)
    tau_minus = np.full((n_a_r, n_b, n_a_phi), np.nan, dtype=float)
    emit_plus = np.full((n_a_r, n_b, n_a_phi, 2), np.nan, dtype=float)
    emit_minus = np.full((n_a_r, n_b, n_a_phi, 2), np.nan, dtype=float)
    from_plus = np.full((n_a_r, n_b, n_a_phi, 2), np.nan, dtype=float)
    from_minus = np.full((n_a_r, n_b, n_a_phi, 2), np.nan, dtype=float)
    ok_plus = np.zeros((n_a_r, n_b, n_a_phi), dtype=bool)
    photon_safe_plus = np.zeros((n_a_r, n_b, n_a_phi), dtype=bool)
    min_radius_plus_m = np.full((n_a_r, n_b, n_a_phi), np.nan, dtype=float)
    iters_plus = np.zeros((n_a_r, n_b, n_a_phi), dtype=np.int32)

    def _assign(ai: int, out_one: dict[str, np.ndarray]) -> None:
        path_plus[ai] = np.asarray(out_one["path_plus_xy_m"], dtype=float)
        path_minus[ai] = np.asarray(out_one["path_minus_xy_m"], dtype=float)
        t_plus[ai] = np.asarray(out_one["time_plus_s"], dtype=float)
        t_minus[ai] = np.asarray(out_one["time_minus_s"], dtype=float)
        tau_plus[ai] = np.asarray(out_one["proper_time_plus_s"], dtype=float)
        tau_minus[ai] = np.asarray(out_one["proper_time_minus_s"], dtype=float)
        emit_plus[ai] = np.asarray(out_one["emit_dir_plus_xy"], dtype=float)
        emit_minus[ai] = np.asarray(out_one["emit_dir_minus_xy"], dtype=float)
        from_plus[ai] = np.asarray(out_one["arrive_from_plus_xy"], dtype=float)
        from_minus[ai] = np.asarray(out_one["arrive_from_minus_xy"], dtype=float)
        ok_plus[ai] = np.asarray(out_one["success_plus"], dtype=bool)
        photon_safe_plus[ai] = np.asarray(out_one["photon_safe_plus"], dtype=bool)
        min_radius_plus_m[ai] = np.asarray(out_one["min_radius_plus_m"], dtype=float)
        iters_plus[ai] = np.asarray(out_one["iters_plus"], dtype=np.int32)

    _assign(0, first)
    print(
        f"[A-radius 1/{n_a_r}] A={shared_radii_rs[0]:.6f} Rs complete "
        f"(B rings={n_b}, A phi={n_a_phi}).",
        flush=True,
    )

    for ai in range(1, n_a_r):
        out_one = solve_b_radial_series(
            rs_m=rs_m,
            a_radius_m=float(shared_radii_m[ai]),
            b_radii_m=shared_radii_m,
            a_phi_count=int(args.a_phi_count),
            a_phi_second_rad=(None if args.a_phi_second_rad is None else float(args.a_phi_second_rad)),
            a_phi_step_rad=(None if args.a_phi_step_rad is None else float(args.a_phi_step_rad)),
            node_count=int(args.node_count),
            node_spacing=str(args.node_spacing),
            optimizer=str(args.optimizer),
            max_iter=int(args.max_iter),
            opt_ftol=float(args.opt_ftol),
            opt_gtol=float(args.opt_gtol),
            adaptive_iter=bool(args.adaptive_iter),
            fail_fast_series=bool(args.fail_fast_series),
            workers=int(args.workers),
            debug_show_rings=bool(args.debug_show_rings),
            debug_pause_rings=bool(args.debug_pause_rings),
        )
        _assign(ai, out_one)
        print(
            f"[A-radius {ai+1}/{n_a_r}] A={shared_radii_rs[ai]:.6f} Rs complete "
            f"(B rings={n_b}, A phi={n_a_phi}).",
            flush=True,
        )

    out = {
        "rs_m": np.asarray(rs_m, dtype=float),
        "a_radii_m": np.asarray(shared_radii_m, dtype=float),
        "a_radii_rs": np.asarray(shared_radii_rs, dtype=float),
        "b_radii_m": np.asarray(shared_radii_m, dtype=float),
        "b_radii_rs": np.asarray(shared_radii_rs, dtype=float),
        "a_phi_rad": np.asarray(first["a_phi_rad"], dtype=float),
        "path_plus_xy_m": path_plus,
        "path_minus_xy_m": path_minus,
        "time_plus_s": t_plus,
        "time_minus_s": t_minus,
        "proper_time_plus_s": tau_plus,
        "proper_time_minus_s": tau_minus,
        "emit_dir_plus_xy": emit_plus,
        "emit_dir_minus_xy": emit_minus,
        "arrive_from_plus_xy": from_plus,
        "arrive_from_minus_xy": from_minus,
        "success_plus": ok_plus,
        "photon_safe_plus": photon_safe_plus,
        "min_radius_plus_m": min_radius_plus_m,
        "iters_plus": iters_plus,
        "bh_mass_kg": np.asarray(bh.mass_kg, dtype=float),
        "bh_diameter_light_seconds": np.asarray(bh.diameter_light_seconds, dtype=float),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"Saved: {args.output}")
    n_total = int(out["success_plus"].size)
    n_ok = int(np.count_nonzero(out["success_plus"]))
    n_photo_viol = int(np.count_nonzero(~np.asarray(out["photon_safe_plus"], dtype=bool)))
    print(
        "Solve summary: "
        f"A_radii={n_a_r}, B_rings={n_b}, A_phi={n_a_phi}, nodes={int(args.node_count)}, "
        f"success={n_ok}/{n_total}, photon_violations={n_photo_viol}"
    )


if __name__ == "__main__":
    main()
