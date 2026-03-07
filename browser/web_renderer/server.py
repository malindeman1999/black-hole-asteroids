from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from blackhole_geodesics import C
from earliest_visible_interpolated_session import SampledTrajectory3D
from inertial_objects import InertialTetrahedron
from precompute_earliest_grid import PrecomputedEarliestInterpolator


class FrameRequest(BaseModel):
    t: float = Field(default=0.0, description="Coordinate time in seconds.")
    observer_point_b: Optional[List[float]] = Field(
        default=None,
        description="Observer position [x,y,z] meters. Defaults to [2*rs,0,0].",
    )
    use_gpu: bool = Field(default=False, description="Use GPU interpolation backend when available.")


@dataclass
class RendererContext:
    interpolator: PrecomputedEarliestInterpolator
    tetra: InertialTetrahedron
    rs_m: float


def _resolve_precompute_path(path: Path) -> Path:
    if path.exists():
        return path
    fallbacks = [
        Path("earliest_angles_precompute_10rs.npz"),
        Path("tests") / "earliest_angles_precompute_10rs.npz",
        Path("data") / "earliest_angles_precompute_10rs.npz",
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Precompute file not found: {path}")
    return found


def _build_center_trajectory(rs_m: float, tmin: float = -8.0, tmax: float = 320.0, n: int = 4097) -> SampledTrajectory3D:
    ts = np.linspace(float(tmin), float(tmax), max(2, int(n)), dtype=float)
    r0 = 5.0 * C
    window = max(1e-9, float(tmax - tmin))
    omega = 4.0 * np.pi / window
    phase = omega * (ts - float(tmin))
    xs = r0 * np.cos(phase)
    ys = r0 * np.sin(phase)
    zs = np.zeros_like(xs)
    return SampledTrajectory3D.from_arrays(ts=ts, xs=xs, ys=ys, zs=zs)


def _create_context(precompute_path: Path) -> RendererContext:
    interp = PrecomputedEarliestInterpolator.from_npz(precompute_path)
    interp.prepare_backend(use_gpu=False)
    sampled = _build_center_trajectory(rs_m=interp.rs_m)
    tetra = InertialTetrahedron(
        sampled_trajectory=sampled,
        size_light_seconds=0.1,
        rotation_angles_deg=(0.0, 0.0, 0.0),
    )
    return RendererContext(interpolator=interp, tetra=tetra, rs_m=float(interp.rs_m))


def create_app(precompute_npz: Path | str = Path("data") / "earliest_angles_precompute_10rs.npz") -> FastAPI:
    app = FastAPI(title="Black Hole Tetra Renderer API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved = _resolve_precompute_path(Path(precompute_npz))
    ctx = _create_context(resolved)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "precompute": str(resolved), "rs_m": ctx.rs_m}

    @app.post("/api/frame")
    def frame(req: FrameRequest) -> dict:
        observer = req.observer_point_b if req.observer_point_b is not None else [2.0 * ctx.rs_m, 0.0, 0.0]
        t = float(req.t)
        corners = ctx.tetra.points_at(t)
        triangles = ctx.tetra.triangle_indices()
        vis = ctx.tetra.visibility_angles_from_points(
            points_m=corners,
            observer_point_b=observer,
            interpolator=ctx.interpolator,
            use_gpu=bool(req.use_gpu),
            batch_size=5000,
        )

        corner = vis["corners"]
        face = vis["faces"]
        return {
            "t": t,
            "rs_m": ctx.rs_m,
            "observer_point_b": [float(v) for v in observer],
            "center_m": [float(v) for v in ctx.tetra.center_at(t).tolist()],
            "vertices_m": [[float(v) for v in p] for p in corners.tolist()],
            "triangles": [[int(v) for v in tri] for tri in triangles.tolist()],
            "corner_first_direction": [int(v) for v in np.asarray(corner["first_direction"], dtype=int).tolist()],
            "corner_first_gamma_at_b_rad": [float(v) for v in np.asarray(corner["first_gamma_at_b_rad"], dtype=float).tolist()],
            "face_first_direction": [int(v) for v in np.asarray(face["first_direction"], dtype=int).tolist()],
            "face_first_gamma_at_b_rad": [float(v) for v in np.asarray(face["first_gamma_at_b_rad"], dtype=float).tolist()],
        }

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_renderer.server:app", host="127.0.0.1", port=8000, reload=False)
