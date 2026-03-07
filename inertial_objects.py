from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from blackhole_geodesics import C
from earliest_visible_interpolated_session import SampledTrajectory3D
from precompute_earliest_grid import PrecomputedEarliestInterpolator


def _rotation_matrix_xyz_deg(angles_deg: Sequence[float]) -> np.ndarray:
    ax, ay, az = (float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2]))
    rx = np.deg2rad(ax)
    ry = np.deg2rad(ay)
    rz = np.deg2rad(az)

    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)

    # Intrinsic XYZ rotation (roll->pitch->yaw style).
    rx_m = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_m = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_m = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_m @ ry_m @ rx_m


def _first_branch_fields(ok_plus: np.ndarray, ok_minus: np.ndarray, dt_plus: np.ndarray, dt_minus: np.ndarray) -> np.ndarray:
    first_plus = ok_plus & ((~ok_minus) | (dt_plus <= dt_minus))
    out = np.zeros(ok_plus.shape, dtype=int)
    out[first_plus] = +1
    out[(~first_plus) & ok_minus] = -1
    return out


def _interpolate_visibility(
    interpolator: PrecomputedEarliestInterpolator,
    a_points_m: np.ndarray,
    observer_point_b: Sequence[float],
    use_gpu: bool,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    a = np.asarray(a_points_m, dtype=float).reshape(-1, 3)
    b = np.asarray(observer_point_b, dtype=float).reshape(1, 3)
    b_all = np.repeat(b, int(a.shape[0]), axis=0)
    out = interpolator.interpolate_pairs_3d(
        a_points_m=a,
        b_points_m=b_all,
        use_gpu=bool(use_gpu),
        batch_size=int(batch_size),
    )
    ok_plus = np.asarray(out["ok_plus"], dtype=bool)
    ok_minus = np.asarray(out["ok_minus"], dtype=bool)
    dt_plus = np.asarray(out["delta_t_plus_s"], dtype=float)
    dt_minus = np.asarray(out["delta_t_minus_s"], dtype=float)
    gb_plus = np.asarray(out["gamma_at_b_plus_rad"], dtype=float)
    gb_minus = np.asarray(out["gamma_at_b_minus_rad"], dtype=float)
    dir_plus = np.asarray(out.get("arrival_dir_plus_xyz", np.full((a.shape[0], 3), np.nan)), dtype=float)
    dir_minus = np.asarray(out.get("arrival_dir_minus_xyz", np.full((a.shape[0], 3), np.nan)), dtype=float)

    first_dir = _first_branch_fields(ok_plus=ok_plus, ok_minus=ok_minus, dt_plus=dt_plus, dt_minus=dt_minus)
    first_gamma = np.full(ok_plus.shape, np.nan, dtype=float)
    first_dt = np.full(ok_plus.shape, np.nan, dtype=float)
    first_arrival_dir = np.full((a.shape[0], 3), np.nan, dtype=float)
    plus_mask = first_dir == +1
    minus_mask = first_dir == -1
    first_gamma[plus_mask] = gb_plus[plus_mask]
    first_gamma[minus_mask] = gb_minus[minus_mask]
    first_dt[plus_mask] = dt_plus[plus_mask]
    first_dt[minus_mask] = dt_minus[minus_mask]
    first_arrival_dir[plus_mask, :] = dir_plus[plus_mask, :]
    first_arrival_dir[minus_mask, :] = dir_minus[minus_mask, :]

    return {
        "ok_plus": ok_plus,
        "ok_minus": ok_minus,
        "delta_t_plus_s": dt_plus,
        "delta_t_minus_s": dt_minus,
        "gamma_at_b_plus_rad": gb_plus,
        "gamma_at_b_minus_rad": gb_minus,
        "arrival_dir_plus_xyz": dir_plus,
        "arrival_dir_minus_xyz": dir_minus,
        "first_direction": first_dir,
        "first_gamma_at_b_rad": first_gamma,
        "first_delta_t_s": first_dt,
        "first_arrival_dir_xyz": first_arrival_dir,
    }


@dataclass
class InertialPointMass:
    """
    Inertial point-mass trajectory wrapper.

    The center is directly given by `sampled_trajectory`. No shape offsets are used.
    """

    sampled_trajectory: SampledTrajectory3D

    def center_at(self, t: float) -> np.ndarray:
        p = self.sampled_trajectory.eval_points(np.asarray([float(t)], dtype=float))[0]
        return np.asarray(p, dtype=float)

    def points_at(self, t: float) -> np.ndarray:
        return self.center_at(float(t)).reshape(1, 3)

    def triangles_at(self, t: float) -> np.ndarray:
        del t
        return np.zeros((0, 3, 3), dtype=float)

    def visibility_angles_from_points(
        self,
        points_m: Sequence[Sequence[float]],
        observer_point_b: Sequence[float],
        interpolator: PrecomputedEarliestInterpolator,
        use_gpu: bool = False,
        batch_size: int = 5000,
    ) -> Dict[str, object]:
        corners = _interpolate_visibility(
            interpolator=interpolator,
            a_points_m=np.asarray(points_m, dtype=float),
            observer_point_b=observer_point_b,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )
        return {"corners": corners, "faces": None}

    def visibility_angles_at_time(
        self,
        t: float,
        observer_point_b: Sequence[float],
        interpolator: PrecomputedEarliestInterpolator,
        use_gpu: bool = False,
        batch_size: int = 5000,
    ) -> Dict[str, object]:
        return self.visibility_angles_from_points(
            points_m=self.points_at(float(t)),
            observer_point_b=observer_point_b,
            interpolator=interpolator,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )


@dataclass
class InertialTetrahedron:
    """
    Inertial tetrahedron trajectory wrapper.

    Current implementation uses simple rigid offsets from the center trajectory.
    Relativistic contraction/rotation dynamics are intentionally not applied yet.
    """

    sampled_trajectory: SampledTrajectory3D
    size_light_seconds: float = 0.1
    rotation_angles_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.size_m = float(self.size_light_seconds) * C
        if self.size_m <= 0.0:
            raise ValueError("size_light_seconds must be > 0.")
        self._rmat = _rotation_matrix_xyz_deg(self.rotation_angles_deg)
        self._face_indices = np.asarray(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            dtype=int,
        )
        # Regular tetrahedron points centered at origin.
        base = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=float,
        )
        base /= np.sqrt(3.0)
        self._local_vertices_m = (self._rmat @ (self.size_m * base).T).T

    def center_at(self, t: float) -> np.ndarray:
        p = self.sampled_trajectory.eval_points(np.asarray([float(t)], dtype=float))[0]
        return np.asarray(p, dtype=float)

    def points_at(self, t: float) -> np.ndarray:
        c = self.center_at(float(t)).reshape(1, 3)
        return c + self._local_vertices_m

    def triangles_at(self, t: float) -> np.ndarray:
        verts = self.points_at(float(t))
        return verts[self._face_indices]

    def face_centroids_at(self, t: float) -> np.ndarray:
        tris = self.triangles_at(float(t))
        if tris.size == 0:
            return np.zeros((0, 3), dtype=float)
        return np.mean(tris, axis=1)

    def triangle_indices(self) -> np.ndarray:
        return self._face_indices.copy()

    def visibility_angles_from_points(
        self,
        points_m: Sequence[Sequence[float]],
        observer_point_b: Sequence[float],
        interpolator: PrecomputedEarliestInterpolator,
        use_gpu: bool = False,
        batch_size: int = 5000,
    ) -> Dict[str, object]:
        corners_xyz = np.asarray(points_m, dtype=float).reshape(4, 3)
        triangles = corners_xyz[self._face_indices]
        face_centroids = np.mean(triangles, axis=1)

        corners = _interpolate_visibility(
            interpolator=interpolator,
            a_points_m=corners_xyz,
            observer_point_b=observer_point_b,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )
        faces = _interpolate_visibility(
            interpolator=interpolator,
            a_points_m=face_centroids,
            observer_point_b=observer_point_b,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )
        return {
            "corners": corners,
            "faces": faces,
            "triangle_vertices_m": triangles,
            "triangle_indices": self._face_indices.copy(),
        }

    def visibility_angles_at_time(
        self,
        t: float,
        observer_point_b: Sequence[float],
        interpolator: PrecomputedEarliestInterpolator,
        use_gpu: bool = False,
        batch_size: int = 5000,
    ) -> Dict[str, object]:
        return self.visibility_angles_from_points(
            points_m=self.points_at(float(t)),
            observer_point_b=observer_point_b,
            interpolator=interpolator,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )

    def apparent_rays_at_time(
        self,
        t: float,
        observer_point_b: Sequence[float],
        interpolator: PrecomputedEarliestInterpolator,
        use_gpu: bool = False,
        batch_size: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Return apparent arrival ray directions at observer B using earliest-branch selection.

        This uses precomputed interpolated geodesic data (+/- branches), then picks the first
        visible branch per corner/face by earliest arrival (minimum travel time).
        """
        vis = self.visibility_angles_at_time(
            t=float(t),
            observer_point_b=observer_point_b,
            interpolator=interpolator,
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )
        corners = vis["corners"]
        faces = vis["faces"]
        return {
            "corner_first_direction": np.asarray(corners["first_direction"], dtype=int),
            "corner_first_arrival_dir_xyz": np.asarray(corners["first_arrival_dir_xyz"], dtype=float),
            "corner_first_gamma_at_b_rad": np.asarray(corners["first_gamma_at_b_rad"], dtype=float),
            "face_first_direction": np.asarray(faces["first_direction"], dtype=int),
            "face_first_arrival_dir_xyz": np.asarray(faces["first_arrival_dir_xyz"], dtype=float),
            "face_first_gamma_at_b_rad": np.asarray(faces["first_gamma_at_b_rad"], dtype=float),
            "triangle_indices": self._face_indices.copy(),
            "vertices_m": self.points_at(float(t)),
        }


__all__ = [
    "InertialPointMass",
    "InertialTetrahedron",
]
