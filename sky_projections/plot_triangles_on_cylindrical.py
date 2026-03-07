from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from sky_projections.icosphere_mesh import build_icosphere, xyz_to_uv
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sky_projections.icosphere_mesh import build_icosphere, xyz_to_uv


def _load_rgb_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        arr = np.asarray(plt.imread(str(path)))
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return arr


def _unwrap_uv_for_face(uv3: np.ndarray) -> np.ndarray:
    out = np.asarray(uv3, dtype=float).copy()
    u = out[:, 0]
    if float(np.max(u) - np.min(u)) > 0.5:
        u = np.where(u < 0.5, u + 1.0, u)
        out[:, 0] = u
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Display icosphere triangles on top of a 2D cylindrical/equirectangular image."
    )
    p.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent / "checkerboard_equirect.png",
        help="Path to cylindrical/equirectangular image.",
    )
    p.add_argument("--subdivisions", type=int, default=2, help="Icosphere subdivisions.")
    p.add_argument("--flip-v", action="store_true", help="Flip V if needed for your image convention.")
    p.add_argument("--line-color", default="red", help="Triangle edge color.")
    p.add_argument("--line-width", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = _load_rgb_image(args.image)
    h, w, _ = img.shape
    verts, faces = build_icosphere(subdivisions=int(args.subdivisions))
    uv = xyz_to_uv(verts, flip_v=bool(args.flip_v))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.imshow(img, origin="upper", extent=[0.0, 1.0, 1.0, 0.0])

    for tri in faces:
        idx = np.asarray(tri, dtype=int)
        uv3 = _unwrap_uv_for_face(uv[idx])
        u = uv3[:, 0]
        v = uv3[:, 1]
        for k in range(3):
            u0, v0 = u[k], v[k]
            u1, v1 = u[(k + 1) % 3], v[(k + 1) % 3]
            # Draw wrapped and primary copies so seam-crossing edges appear continuous.
            for off in (-1.0, 0.0, 1.0):
                ax.plot(
                    [u0 + off, u1 + off],
                    [v0, v1],
                    color=args.line_color,
                    lw=float(args.line_width),
                    alpha=float(args.alpha),
                )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("u (longitude / 2pi)")
    ax.set_ylabel("v (latitude)")
    ax.set_title(f"Icosphere Triangles on Cylindrical Projection ({args.image.name}, subdiv={int(args.subdivisions)})")
    plt.show()


if __name__ == "__main__":
    main()
