from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import sys

import numpy as np

try:
    from sky_projections.icosphere_mesh import build_icosphere, xyz_to_uv
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sky_projections.icosphere_mesh import build_icosphere, xyz_to_uv


def _expand_faces_with_seam_fix(points: np.ndarray, faces: np.ndarray, flip_v: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_pts = points[faces.reshape(-1)]
    tri_uv = xyz_to_uv(tri_pts, flip_v=flip_v)

    # Per-triangle seam unwrap: avoid interpolation across u=0/1 seam.
    # Keep unwrapped u (>1) on wrapped triangles; renderer repeat mode handles it.
    tri_uv = tri_uv.reshape(-1, 3, 2)
    u = tri_uv[:, :, 0]
    umax = np.max(u, axis=1, keepdims=True)
    umin = np.min(u, axis=1, keepdims=True)
    wraps = (umax - umin) > 0.5
    adjust = wraps & (u < 0.5)
    u = np.where(adjust, u + 1.0, u)
    tri_uv[:, :, 0] = u
    tri_uv = tri_uv.reshape(-1, 2)

    n_tri = faces.shape[0]
    tri_faces = np.arange(n_tri * 3, dtype=np.int32).reshape(n_tri, 3)
    return tri_pts, tri_faces, tri_uv


def _build_polydata(points: np.ndarray, faces: np.ndarray, uv: np.ndarray):
    import pyvista as pv

    face_cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).astype(np.int32).ravel()
    mesh = pv.PolyData(points, face_cells)
    mesh.active_texture_coordinates = uv.astype(np.float32)
    return mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU-rendered low-poly textured sphere from cylindrical sky image.")
    p.add_argument("--image", type=Path, default=Path(__file__).resolve().parent / "checkerboard_equirect.png")
    p.add_argument("--subdivisions", type=int, default=2, help="Icosphere subdivisions (0=>20 tris, 1=>80, 2=>320...).")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--flip-v", action="store_true", help="Flip vertical texture coordinate if image appears upside down.")
    p.add_argument("--show-edges", action="store_true", default=True)
    p.add_argument("--outside-view", action="store_true", help="Use external camera view instead of default inside view.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("pyvista is required. Install with: pip install pyvista") from exc

    points, faces = build_icosphere(subdivisions=int(args.subdivisions))
    points = points * float(args.radius)
    tri_pts, tri_faces, tri_uv = _expand_faces_with_seam_fix(points, faces, flip_v=bool(args.flip_v))
    mesh = _build_polydata(tri_pts, tri_faces, tri_uv)

    tex = pv.read_texture(str(args.image))
    try:
        tex.repeat = True
    except Exception:
        pass
    plotter = pv.Plotter(window_size=(1280, 820))
    plotter.set_background("black")
    plotter.add_mesh(
        mesh,
        texture=tex,
        smooth_shading=False,
        lighting=False,
        show_edges=bool(args.show_edges),
        edge_color="red",
        line_width=3.0,
        culling=("back" if bool(args.outside_view) else None),
    )
    if bool(args.outside_view):
        plotter.camera_position = "iso"
    else:
        # Default: view from inside the sphere.
        r = float(args.radius)
        cam = plotter.camera
        cam.position = (0.0, 0.0, 0.0)
        cam.focal_point = (1.0 * r, 0.0, 0.0)
        cam.up = (0.0, 0.0, 1.0)
        # Keep interior shell inside clip range from center.
        cam.clipping_range = (max(1e-6, 1e-4 * r), max(2.0 * r, 10.0))
        # Lock camera at center; keep mouse interaction as look rotation.
        def _lock_camera_position(_obj=None, _event=None):
            c = plotter.camera
            # Preserve current look direction from interaction, only pin position.
            try:
                dop = np.asarray(c.direction, dtype=float)
            except Exception:
                dop = np.asarray(c.GetDirectionOfProjection(), dtype=float)
            vec = dop
            n = float(np.linalg.norm(vec))
            if n < 1e-12:
                vec = np.asarray([1.0, 0.0, 0.0], dtype=float)
                n = 1.0
            vec = vec / n
            c.position = (0.0, 0.0, 0.0)
            c.focal_point = tuple((vec * r).tolist())
            c.clipping_range = (max(1e-6, 1e-4 * r), max(2.0 * r, 10.0))
        plotter.iren.add_observer("InteractionEvent", _lock_camera_position)
        plotter.iren.add_observer("MouseWheelForwardEvent", _lock_camera_position)
        plotter.iren.add_observer("MouseWheelBackwardEvent", _lock_camera_position)
    plotter.add_text(
        f"triangles={faces.shape[0]} (GPU via VTK)",
        font_size=10,
        color="white",
    )
    plotter.show()


if __name__ == "__main__":
    main()
