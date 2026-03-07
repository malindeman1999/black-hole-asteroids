from __future__ import annotations

import argparse
from pathlib import Path
import sys
from math import pi

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole
from precompute_earliest_grid import PrecomputedEarliestInterpolator
from sky_projections.icosphere_mesh import build_icosphere, xyz_to_uv


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "earliest_angles_precompute_10rs.npz",
        Path("earliest_angles_precompute_10rs.npz"),
        Path("tests") / "earliest_angles_precompute_10rs.npz",
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Precompute input not found: {path}")
    return found


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Precompute sky-vertex visibility table for icosphere corners at fixed source radius "
            "(default 100 Rs) for each observer B radius in the interpolation table."
        )
    )
    p.add_argument("--input", type=Path, default=Path("data") / "earliest_angles_precompute_10rs.npz")
    p.add_argument("--output", type=Path, default=Path("data") / "sky_vertices_precompute_100rs_sub2.npz")
    p.add_argument("--sky-radius-rs", type=float, default=100.0)
    p.add_argument("--subdivisions", type=int, default=2)
    p.add_argument("--flip-v", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="fast")
    p.add_argument("--backend", choices=["serial", "thread", "process"], default="process")
    p.add_argument("--batch-size", type=int, default=5000)
    return p.parse_args()


def _arrival_gamma_at_b(rs: float, r_b: float, impact_parameter_m: float) -> float:
    s = impact_parameter_m * np.sqrt(max(0.0, 1.0 - rs / r_b)) / r_b
    s = max(-1.0, min(1.0, float(s)))
    return float(np.arcsin(s))


def _local_basis_for_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    eps = 1e-12
    r_b = float(np.linalg.norm(b))
    er = b / max(r_b, eps)
    x_a = float(np.dot(a, er))
    a_perp = a - x_a * er
    n_perp = float(np.linalg.norm(a_perp))
    if n_perp > eps:
        ephi = a_perp / n_perp
    else:
        ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(er[2])) > 0.9:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        ephi = np.cross(ref, er)
        ephi = ephi / max(float(np.linalg.norm(ephi)), eps)
    y_a = float(np.dot(a, ephi))
    a_phi = float(np.arctan2(y_a, x_a))
    dtheta_short = float(np.mod(-a_phi + pi, 2.0 * pi) - pi)
    short_sign = 1.0 if dtheta_short >= 0.0 else -1.0
    return er, ephi, short_sign


def main() -> None:
    args = _parse_args()
    npz_in = _resolve_input(Path(args.input))
    interp = PrecomputedEarliestInterpolator.from_npz(npz_in)
    rs = float(interp.rs_m)
    bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs) / C).with_quality(str(args.quality))

    verts_unit, faces = build_icosphere(subdivisions=int(args.subdivisions))
    uv = xyz_to_uv(verts_unit, flip_v=bool(args.flip_v))
    sky_radius_m = float(args.sky_radius_rs) * rs
    verts_m = verts_unit * sky_radius_m

    b_r = np.asarray(interp.b_r_m, dtype=float)
    n_b = b_r.size
    n_v = verts_m.shape[0]
    arr_p = np.full((n_b, n_v, 3), np.nan, dtype=float)
    arr_m = np.full((n_b, n_v, 3), np.nan, dtype=float)
    dt_p = np.full((n_b, n_v), np.nan, dtype=float)
    dt_m = np.full((n_b, n_v), np.nan, dtype=float)
    ok_p = np.zeros((n_b, n_v), dtype=bool)
    ok_m = np.zeros((n_b, n_v), dtype=bool)

    for i, rb in enumerate(b_r):
        b_points = np.zeros((n_v, 3), dtype=float)
        b_points[:, 0] = float(rb)
        pairs = [(tuple(float(v) for v in verts_m[k]), tuple(float(v) for v in b_points[k])) for k in range(n_v)]
        rr = bh.find_two_shortest_geodesics_batch(
            point_pairs=pairs,
            a_before_b=True,
            backend=str(args.backend),
            use_gpu=bool(args.use_gpu),
        )
        for k in range(n_v):
            a = np.asarray(verts_m[k], dtype=float)
            b = np.asarray(b_points[k], dtype=float)
            r_a = float(np.linalg.norm(a))
            r_b = float(np.linalg.norm(b))
            er, ephi, short_sign = _local_basis_for_pair(a, b)
            p_plus = None
            p_minus = None
            for p in rr[k].paths:
                if int(p.direction) == +1:
                    p_plus = p
                elif int(p.direction) == -1:
                    p_minus = p

            if p_plus is not None:
                gamma = _arrival_gamma_at_b(rs=rs, r_b=r_b, impact_parameter_m=float(p_plus.impact_parameter_m))
                radial_sign = (+1.0 if r_a > r_b else -1.0) if str(p_plus.branch) == "turning" else (
                    -1.0 if r_a > r_b else +1.0
                )
                orient_sign = short_sign
                vv = radial_sign * np.cos(gamma) * er + orient_sign * np.sin(gamma) * ephi
                vv = vv / max(float(np.linalg.norm(vv)), 1e-12)
                arr_p[i, k, :] = vv
                dt_p[i, k] = float(p_plus.travel_time_s)
                ok_p[i, k] = np.all(np.isfinite(vv)) and np.isfinite(dt_p[i, k])

            if p_minus is not None:
                gamma = _arrival_gamma_at_b(rs=rs, r_b=r_b, impact_parameter_m=float(p_minus.impact_parameter_m))
                radial_sign = (+1.0 if r_a > r_b else -1.0) if str(p_minus.branch) == "turning" else (
                    -1.0 if r_a > r_b else +1.0
                )
                orient_sign = -short_sign
                vv = radial_sign * np.cos(gamma) * er + orient_sign * np.sin(gamma) * ephi
                vv = vv / max(float(np.linalg.norm(vv)), 1e-12)
                arr_m[i, k, :] = vv
                dt_m[i, k] = float(p_minus.travel_time_s)
                ok_m[i, k] = np.all(np.isfinite(vv)) and np.isfinite(dt_m[i, k])
        print(f"row {i + 1}/{n_b}: B={rb / rs:.3f} Rs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rs_m=np.asarray(rs, dtype=float),
        input_precompute=str(npz_in),
        b_r_m=b_r,
        sky_radius_m=np.asarray(sky_radius_m, dtype=float),
        subdivisions=np.asarray(int(args.subdivisions), dtype=np.int32),
        vertices_unit=verts_unit.astype(np.float32),
        vertices_m=verts_m.astype(np.float32),
        faces=faces.astype(np.int32),
        uv=uv.astype(np.float32),
        arrival_dir_plus_xyz=arr_p.astype(np.float32),
        arrival_dir_minus_xyz=arr_m.astype(np.float32),
        delta_t_plus_s=dt_p.astype(np.float32),
        delta_t_minus_s=dt_m.astype(np.float32),
        ok_plus=ok_p,
        ok_minus=ok_m,
    )
    print(f"Saved sky vertex table: {args.output}")


if __name__ == "__main__":
    main()
