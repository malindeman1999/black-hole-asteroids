from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _base_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    v = np.array(
        [
            [-1, +phi, 0],
            [+1, +phi, 0],
            [-1, -phi, 0],
            [+1, -phi, 0],
            [0, -1, +phi],
            [0, +1, +phi],
            [0, -1, -phi],
            [0, +1, -phi],
            [+phi, 0, -1],
            [+phi, 0, +1],
            [-phi, 0, -1],
            [-phi, 0, +1],
        ],
        dtype=float,
    )
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )
    return _normalize_rows(v), f


def _subdivide(points: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    midpoint_cache: dict[tuple[int, int], int] = {}
    pts = points.tolist()
    new_faces = []

    def midpoint_index(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        p = 0.5 * (points[i] + points[j])
        p = p / max(np.linalg.norm(p), 1e-12)
        idx = len(pts)
        pts.append(p.tolist())
        midpoint_cache[key] = idx
        return idx

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        a = midpoint_index(i0, i1)
        b = midpoint_index(i1, i2)
        c = midpoint_index(i2, i0)
        new_faces.extend(
            [
                [i0, a, c],
                [i1, b, a],
                [i2, c, b],
                [a, b, c],
            ]
        )
    return np.asarray(pts, dtype=float), np.asarray(new_faces, dtype=np.int32)


def build_icosphere(subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    points, faces = _base_icosahedron()
    for _ in range(max(0, int(subdivisions))):
        points, faces = _subdivide(points, faces)
    return _normalize_rows(points), faces


def xyz_to_uv(points: np.ndarray, flip_v: bool = False) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = np.clip(points[:, 2], -1.0, 1.0)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)
    u = (lon + np.pi) / (2.0 * np.pi)
    v = 0.5 - (lat / np.pi)
    if flip_v:
        v = 1.0 - v
    return np.stack([u, v], axis=1)

