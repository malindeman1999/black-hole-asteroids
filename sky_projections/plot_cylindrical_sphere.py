from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_image_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        img = plt.imread(path)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr


def plot_cylindrical_texture_on_sphere(image_path: Path, stride: int = 2) -> None:
    """
    Plot an equirectangular/cylindrical sky image on a sphere.

    Convention:
    - Image top row    => latitude +90 deg (north pole)
    - Image bottom row => latitude -90 deg (south pole)
    - Horizontal axis maps full longitude range [-180, +180) deg.
    """
    tex = _load_image_rgb(image_path)
    h, w, _ = tex.shape

    y_idx = np.arange(0, h, max(1, int(stride)), dtype=float)
    x_idx = np.arange(0, w, max(1, int(stride)), dtype=float)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")

    lon = (xx / max(1.0, float(w - 1))) * (2.0 * np.pi) - np.pi
    lat = 0.5 * np.pi - (yy / max(1.0, float(h - 1))) * np.pi

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    tex_sub = tex[:: max(1, int(stride)), :: max(1, int(stride)), :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=tex_sub,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.set_title(f"Spherical Mapping: {image_path.name}")
    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map a cylindrical/equirectangular sky image onto a sphere.")
    p.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent / "milkyway.jpg",
        help="Path to cylindrical/equirectangular image.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Downsample stride for plotting mesh (larger = faster).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    plot_cylindrical_texture_on_sphere(image_path=image_path, stride=int(args.stride))


if __name__ == "__main__":
    main()
