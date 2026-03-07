from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional, Tuple
import sys
from math import pi, sqrt

import numpy as np
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

from blackhole_geodesics import C, SchwarzschildBlackHole
from earliest_visible_interpolated_session import SampledTrajectory3D
from inertial_objects import InertialTetrahedron
from precompute_earliest_grid import PrecomputedEarliestInterpolator

TESTS_DIR = Path(__file__).resolve().parent / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import plot_visibility_from_initial_states as pvis


def _bisection_root(func, lo: float, hi: float, max_iter: int = 80) -> float:
    flo = func(lo)
    fhi = func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("Root not bracketed")
    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = func(m)
        if abs(fm) < 1e-12:
            return m
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _cumulative_trapezoid(values, x):
    out = [0.0]
    for i in range(1, len(values)):
        dx = x[i] - x[i - 1]
        out.append(out[-1] + 0.5 * (values[i - 1] + values[i]) * dx)
    return out


def _find_turning_radius(bh: SchwarzschildBlackHole, b: float, r_cap: float) -> float:
    rs = bh.schwarzschild_radius_m
    rph = bh.photon_sphere_radius_m
    lo = rph * (1.0 + 1e-9)
    hi = r_cap * (1.0 - 1e-9)

    def f(r: float) -> float:
        return 1.0 / (b * b) - (1.0 - rs / r) / (r * r)

    nscan = 256
    prev_r = lo
    prev_f = f(prev_r)
    for i in range(1, nscan + 1):
        rr = lo + (hi - lo) * i / nscan
        ff = f(rr)
        if prev_f * ff <= 0.0:
            return _bisection_root(f, prev_r, rr)
        prev_r, prev_f = rr, ff
    raise RuntimeError("Failed to locate turning radius")


def _build_path_profile(
    bh: SchwarzschildBlackHole, r_start: float, r_end: float, impact_b: float, target_phi: float, branch: str, n: int = 500
):
    rs = bh.schwarzschild_radius_m
    b2 = impact_b * impact_b

    def phi_density(r: float) -> float:
        w = 1.0 / b2 - (1.0 - rs / r) / (r * r)
        if w < 0.0 and w > -1e-14:
            w = 0.0
        if w <= 0.0:
            return 0.0
        return 1.0 / (r * r * sqrt(w))

    if branch == "monotonic":
        s = [i / (n - 1) for i in range(n)]
        dr = r_end - r_start
        r_samples = [r_start + dr * si for si in s]
        dens = [abs(dr) * phi_density(r) for r in r_samples]
        phi_samples = _cumulative_trapezoid(dens, s)
    else:
        r_turn = _find_turning_radius(bh, impact_b, min(r_start, r_end))
        n1 = max(64, n // 2)
        n2 = max(64, n - n1 + 1)
        s1 = [i / (n1 - 1) for i in range(n1)]
        d1 = r_start - r_turn
        r1 = [r_turn + d1 * (1.0 - si) * (1.0 - si) for si in s1]
        dens1 = [2.0 * abs(d1) * (1.0 - si) * phi_density(rr) for si, rr in zip(s1, r1)]
        phi1 = _cumulative_trapezoid(dens1, s1)

        s2 = [i / (n2 - 1) for i in range(n2)]
        d2 = r_end - r_turn
        r2 = [r_turn + d2 * si * si for si in s2]
        dens2 = [2.0 * abs(d2) * si * phi_density(rr) for si, rr in zip(s2, r2)]
        phi2 = _cumulative_trapezoid(dens2, s2)
        phi2 = [x + phi1[-1] for x in phi2]
        r_samples = r1 + r2[1:]
        phi_samples = phi1 + phi2[1:]

    if phi_samples[-1] > 0.0:
        scale = target_phi / phi_samples[-1]
        phi_samples = [p * scale for p in phi_samples]
    else:
        phi_samples = [target_phi * i / max(1, len(phi_samples) - 1) for i in range(len(phi_samples))]
    return r_samples, phi_samples


def _resolve_precompute_path(path: Path) -> Path:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "earliest_angles_precompute_10rs.npz",
        Path("earliest_angles_precompute_10rs.npz"),
        Path("tests") / "earliest_angles_precompute_10rs.npz",
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Could not find precompute file at {path} or fallback locations.")
    return found


def _resolve_sky_vertices_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "sky_vertices_precompute_100rs_sub2.npz",
        Path("sky_vertices_precompute_100rs_sub2.npz"),
    ]
    return next((p for p in fallbacks if p.exists()), None)


def _resolve_sky_image(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("sky_projections") / "checkerboard_equirect.png",
        Path("sky_projections") / "milkyway.jpg",
    ]
    return next((p for p in fallbacks if p.exists()), None)


def _build_center_trajectory(tmin: float, tmax: float, samples: int, radius_m: float) -> SampledTrajectory3D:
    ts = np.linspace(float(tmin), float(tmax), max(2, int(samples)), dtype=float)
    window = max(1e-9, float(tmax - tmin))
    omega = (4.0 * np.pi) / window
    phase = omega * (ts - float(tmin))
    xs = radius_m * np.cos(phase)
    ys = radius_m * np.sin(phase)
    zs = np.zeros_like(xs)
    return SampledTrajectory3D.from_arrays(ts=ts, xs=xs, ys=ys, zs=zs)


class InertialViewerApp:
    def __init__(self, root: tk.Tk, precompute_npz: Path, sky_vertices_npz: Optional[Path], sky_image_path: Optional[Path]) -> None:
        self.root = root
        self.root.title("Inertial Tetra Viewer")
        self.root.geometry("1400x850")

        self.interp = PrecomputedEarliestInterpolator.from_npz(precompute_npz)
        self.interp.prepare_backend(use_gpu=False)
        self.rs = float(self.interp.rs_m)
        self.bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality("fast")

        traj = _build_center_trajectory(
            tmin=-200.0,
            tmax=320.0,
            samples=4097,
            radius_m=5.0 * self.rs,
        )
        self.tetra = InertialTetrahedron(
            sampled_trajectory=traj,
            size_light_seconds=0.1,
            rotation_angles_deg=(15.0, 25.0, 10.0),
        )

        self.observer_b = np.asarray([9.0 * self.rs, 0.0, 0.0], dtype=float)
        # Yaw=pi means initial look direction is -X in the map plane.
        self.view_yaw_rad = float(np.pi)
        self.t = 0.0
        self.running = False
        self.after_id: Optional[str] = None
        self._mouse_drag_last_y: Optional[int] = None

        self.dt_s = tk.DoubleVar(value=0.25)
        self.speed_scale = tk.DoubleVar(value=1.0)
        self.use_gpu = tk.BooleanVar(value=False)
        self.show_sky_both = tk.BooleanVar(value=True)
        self._sky_status = "sky: disabled"

        self.sky_faces = np.zeros((0, 3), dtype=int)
        self.sky_uv = np.zeros((0, 2), dtype=float)
        self.sky_b_r = np.zeros((0,), dtype=float)
        self.sky_arr_p = np.zeros((0, 0, 3), dtype=float)
        self.sky_arr_m = np.zeros((0, 0, 3), dtype=float)
        self.sky_dt_p = np.zeros((0, 0), dtype=float)
        self.sky_dt_m = np.zeros((0, 0), dtype=float)
        self.sky_ok_p = np.zeros((0, 0), dtype=bool)
        self.sky_ok_m = np.zeros((0, 0), dtype=bool)
        self.sky_tex = np.zeros((1, 1, 3), dtype=np.uint8)
        self._load_sky_data(sky_vertices_npz=sky_vertices_npz, sky_image_path=sky_image_path)

        self.corner_traj, self.face_traj = self._build_tetra_point_trajectories()
        self.prev_corner_batch = None
        self.prev_face_batch = None
        self.prev_corner_t0: Optional[float] = None
        self.prev_face_t0: Optional[float] = None

        self._build_layout()
        self._draw_frame()

    def _offset_trajectory(self, offset_xyz: np.ndarray) -> SampledTrajectory3D:
        s = self.tetra.sampled_trajectory
        off = np.asarray(offset_xyz, dtype=float).reshape(3)
        return SampledTrajectory3D.from_arrays(
            ts=np.asarray(s.ts, dtype=float),
            xs=np.asarray(s.xs, dtype=float) + float(off[0]),
            ys=np.asarray(s.ys, dtype=float) + float(off[1]),
            zs=np.asarray(s.zs, dtype=float) + float(off[2]),
        )

    def _build_tetra_point_trajectories(self) -> tuple[list[SampledTrajectory3D], list[SampledTrajectory3D]]:
        # Corner trajectories are center trajectory with fixed offsets.
        corner_offsets = np.asarray(self.tetra._local_vertices_m, dtype=float)
        corner_traj = [self._offset_trajectory(off) for off in corner_offsets]
        tri_idx = np.asarray(self.tetra.triangle_indices(), dtype=int)
        face_offsets = np.mean(corner_offsets[tri_idx], axis=1)
        face_traj = [self._offset_trajectory(off) for off in face_offsets]
        return corner_traj, face_traj

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, weight=3, uniform="cols")
        self.root.grid_columnconfigure(1, weight=1, uniform="cols")
        self.root.grid_rowconfigure(0, weight=1)

        self.viewer = tk.Canvas(self.root, bg="#050c15", highlightthickness=0)
        self.viewer.grid(row=0, column=0, sticky="nsew")
        self.viewer.bind("<Configure>", lambda _e: self._draw_frame())
        self.viewer.bind("<ButtonPress-1>", self._on_mouse_down)
        self.viewer.bind("<B1-Motion>", self._on_mouse_drag)
        self.viewer.bind("<ButtonRelease-1>", self._on_mouse_up)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=1, sticky="nsew")

        title = ttk.Label(controls, text="Controls", font=("Segoe UI", 13, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.btn_start = ttk.Button(controls, text="Start", command=self._toggle_start)
        self.btn_start.grid(row=1, column=0, sticky="ew", pady=4)

        ttk.Button(controls, text="Step", command=self._step_once).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(controls, text="Reset", command=self._reset).grid(row=3, column=0, sticky="ew", pady=4)

        ttk.Label(controls, text="dt (s)").grid(row=4, column=0, sticky="w", pady=(12, 2))
        ttk.Scale(controls, from_=0.02, to=2.0, variable=self.dt_s, orient="horizontal").grid(
            row=5, column=0, sticky="ew"
        )

        ttk.Label(controls, text="speed x").grid(row=6, column=0, sticky="w", pady=(12, 2))
        ttk.Scale(controls, from_=0.1, to=5.0, variable=self.speed_scale, orient="horizontal").grid(
            row=7, column=0, sticky="ew"
        )

        ttk.Checkbutton(controls, text="Use GPU interpolation", variable=self.use_gpu).grid(
            row=8, column=0, sticky="w", pady=(12, 4)
        )
        ttk.Checkbutton(controls, text="Sky: show + and -", variable=self.show_sky_both, command=self._draw_frame).grid(
            row=9, column=0, sticky="w", pady=(4, 2)
        )

        self.lbl_t = ttk.Label(controls, text="t = 0.000 s")
        self.lbl_t.grid(row=10, column=0, sticky="w", pady=(8, 2))

        self.lbl_dir = ttk.Label(controls, text="face first dirs: []")
        self.lbl_dir.grid(row=11, column=0, sticky="w", pady=(2, 2))

        self.lbl_status = ttk.Label(controls, text="ready")
        self.lbl_status.grid(row=12, column=0, sticky="w", pady=(8, 2))
        self.lbl_sky = ttk.Label(controls, text=self._sky_status)
        self.lbl_sky.grid(row=13, column=0, sticky="w", pady=(2, 2))

        ttk.Separator(controls, orient="horizontal").grid(row=14, column=0, sticky="ew", pady=(8, 8))
        ttk.Label(controls, text="Top-Down Map").grid(row=15, column=0, sticky="w", pady=(0, 4))
        self.map_canvas = tk.Canvas(controls, bg="#0b1624", highlightthickness=1, highlightbackground="#2b3d52")
        self.map_canvas.grid(row=16, column=0, sticky="nsew")

        controls.grid_columnconfigure(0, weight=1)
        controls.grid_rowconfigure(16, weight=1)

    def _load_sky_data(self, sky_vertices_npz: Optional[Path], sky_image_path: Optional[Path]) -> None:
        if sky_vertices_npz is None or (not sky_vertices_npz.exists()):
            self._sky_status = "sky: precompute missing"
            return
        try:
            data = np.load(sky_vertices_npz, allow_pickle=False)
            self.sky_faces = np.asarray(data["faces"], dtype=int)
            self.sky_uv = np.asarray(data["uv"], dtype=float)
            self.sky_b_r = np.asarray(data["b_r_m"], dtype=float)
            self.sky_arr_p = np.asarray(data["arrival_dir_plus_xyz"], dtype=float)
            self.sky_arr_m = np.asarray(data["arrival_dir_minus_xyz"], dtype=float)
            self.sky_dt_p = np.asarray(data["delta_t_plus_s"], dtype=float)
            self.sky_dt_m = np.asarray(data["delta_t_minus_s"], dtype=float)
            self.sky_ok_p = np.asarray(data["ok_plus"], dtype=bool)
            self.sky_ok_m = np.asarray(data["ok_minus"], dtype=bool)
        except Exception:
            self._sky_status = "sky: precompute load failed"
            return

        if sky_image_path is None or (not sky_image_path.exists()):
            self._sky_status = "sky: image missing"
            return
        try:
            if Image is not None:
                self.sky_tex = np.asarray(Image.open(sky_image_path).convert("RGB"), dtype=np.uint8)
            else:
                import matplotlib.pyplot as plt

                img = np.asarray(plt.imread(str(sky_image_path)))
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                if img.shape[2] > 3:
                    img = img[:, :, :3]
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
                self.sky_tex = img
        except Exception:
            self._sky_status = "sky: image load failed"
            return
        self._sky_status = f"sky: loaded ({self.sky_faces.shape[0]} tris)"

    def _interp_sky_rows(self, r_b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.sky_b_r.size < 1:
            n_v = 0 if self.sky_uv.ndim < 2 else self.sky_uv.shape[0]
            z3 = np.full((n_v, 3), np.nan, dtype=float)
            z1 = np.full((n_v,), np.nan, dtype=float)
            zb = np.zeros((n_v,), dtype=bool)
            return z3, z3, z1, z1, zb, zb
        if self.sky_b_r.size == 1:
            return (
                self.sky_arr_p[0],
                self.sky_arr_m[0],
                self.sky_dt_p[0],
                self.sky_dt_m[0],
                self.sky_ok_p[0],
                self.sky_ok_m[0],
            )
        rb = float(np.clip(r_b, float(self.sky_b_r[0]), float(self.sky_b_r[-1])))
        i0 = int(np.searchsorted(self.sky_b_r, rb, side="right") - 1)
        i0 = max(0, min(i0, self.sky_b_r.size - 2))
        i1 = i0 + 1
        d = float(self.sky_b_r[i1] - self.sky_b_r[i0])
        w = 0.0 if d <= 0.0 else (rb - float(self.sky_b_r[i0])) / d

        ap = (1.0 - w) * self.sky_arr_p[i0] + w * self.sky_arr_p[i1]
        am = (1.0 - w) * self.sky_arr_m[i0] + w * self.sky_arr_m[i1]
        dp = (1.0 - w) * self.sky_dt_p[i0] + w * self.sky_dt_p[i1]
        dm = (1.0 - w) * self.sky_dt_m[i0] + w * self.sky_dt_m[i1]
        op = self.sky_ok_p[i0] & self.sky_ok_p[i1] & np.all(np.isfinite(ap), axis=1)
        om = self.sky_ok_m[i0] & self.sky_ok_m[i1] & np.all(np.isfinite(am), axis=1)
        return ap, am, dp, dm, op, om

    def _sample_tex_color(self, uv: np.ndarray, branch: str) -> str:
        if self.sky_tex.size < 3:
            return "#202020"
        h, w, _ = self.sky_tex.shape
        u = float(uv[0] % 1.0)
        v = float(np.clip(uv[1], 0.0, 1.0))
        x = int(np.clip(round(u * (w - 1)), 0, w - 1))
        y = int(np.clip(round(v * (h - 1)), 0, h - 1))
        c = self.sky_tex[y, x].astype(float)
        if branch == "minus":
            c = 0.75 * c
        c = np.clip(c, 0, 255).astype(int)
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def _on_mouse_down(self, event: tk.Event) -> None:
        self._mouse_drag_last_y = int(event.y)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._mouse_drag_last_y is None:
            self._mouse_drag_last_y = int(event.y)
            return
        dy = int(event.y) - int(self._mouse_drag_last_y)
        self._mouse_drag_last_y = int(event.y)
        # Moving mouse forward/back (up/down) rotates yaw in map plane.
        self.view_yaw_rad += float(-dy) * 0.01
        self._draw_frame()

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._mouse_drag_last_y = None

    def _toggle_start(self) -> None:
        self.running = not self.running
        self.btn_start.configure(text="Stop" if self.running else "Start")
        if self.running:
            self._schedule_next()
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        # Force immediate repaint so paused-mode diagnostics (light paths/vectors)
        # appear as soon as Stop is pressed.
        self._draw_frame()

    def _schedule_next(self) -> None:
        self.after_id = self.root.after(30, self._tick)

    def _tick(self) -> None:
        if not self.running:
            return
        self._step_once()
        self._schedule_next()

    def _step_once(self) -> None:
        self.t += float(self.dt_s.get()) * float(self.speed_scale.get())
        self._draw_frame()

    def _reset(self) -> None:
        self.t = 0.0
        self.prev_corner_batch = None
        self.prev_face_batch = None
        self.prev_corner_t0 = None
        self.prev_face_t0 = None
        self._draw_frame()

    def _project_topdown(self, pts_xyz: np.ndarray, w: int, h: int, scale_world_m: float) -> np.ndarray:
        pts = np.asarray(pts_xyz, dtype=float)
        cx = w * 0.5
        cy = h * 0.5
        s = min(w, h) * 0.45 / max(scale_world_m, 1e-9)
        out = np.zeros((pts.shape[0], 2), dtype=float)
        out[:, 0] = cx + s * pts[:, 0]
        out[:, 1] = cy - s * pts[:, 1]
        return out

    def _project_observer_view(self, pts_xyz: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(pts_xyz, dtype=float)
        rel = pts - self.observer_b.reshape(1, 3)
        # Rotate rel coords by -yaw around z so camera forward stays +x in camera frame.
        cy = np.cos(self.view_yaw_rad)
        sy = np.sin(self.view_yaw_rad)
        x_cam = cy * rel[:, 0] + sy * rel[:, 1]
        y_cam = -sy * rel[:, 0] + cy * rel[:, 1]
        z_cam = rel[:, 2]

        depth = x_cam
        near = 1e3
        visible = depth > near
        focal = 0.65 * min(w, h)
        out = np.zeros((pts.shape[0], 2), dtype=float)
        out[:, 0] = (w * 0.5) + focal * (y_cam / np.maximum(depth, near))
        out[:, 1] = (h * 0.5) - focal * (z_cam / np.maximum(depth, near))
        # Rotate screen projection 90 deg clockwise so map-up aligns with view-up.
        cx = 0.5 * w
        cy = 0.5 * h
        u = out[:, 0] - cx
        v = out[:, 1] - cy
        out[:, 0] = cx - v
        out[:, 1] = cy + u
        return out, visible

    def _project_observer_rays(self, dirs_xyz: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        # Interpolator returns incoming ray directions at B (photon momentum toward observer).
        # For sky projection we need the opposite direction (where to look to see the source).
        dirs = -np.asarray(dirs_xyz, dtype=float)
        # Rotate world ray directions by -yaw about z into camera frame.
        cy = np.cos(self.view_yaw_rad)
        sy = np.sin(self.view_yaw_rad)
        x_cam = cy * dirs[:, 0] + sy * dirs[:, 1]
        y_cam = -sy * dirs[:, 0] + cy * dirs[:, 1]
        z_cam = dirs[:, 2]
        near = 1e-6
        visible = np.isfinite(x_cam) & np.isfinite(y_cam) & np.isfinite(z_cam) & (x_cam > near)
        focal = 0.65 * min(w, h)
        out = np.zeros((dirs.shape[0], 2), dtype=float)
        out[:, 0] = (w * 0.5) + focal * (y_cam / np.maximum(x_cam, near))
        out[:, 1] = (h * 0.5) - focal * (z_cam / np.maximum(x_cam, near))
        # Rotate screen projection 90 deg clockwise so map-up aligns with view-up.
        cx = 0.5 * w
        cy = 0.5 * h
        u = out[:, 0] - cx
        v = out[:, 1] - cy
        out[:, 0] = cx - v
        out[:, 1] = cy + u
        return out, visible

    def _draw_frame(self) -> None:
        w = max(10, int(self.viewer.winfo_width()))
        h = max(10, int(self.viewer.winfo_height()))
        self.viewer.delete("all")

        center = self.tetra.center_at(self.t)
        verts = self.tetra.points_at(self.t)
        tris_idx = self.tetra.triangle_indices()
        tmin = float(np.min(np.asarray(self.tetra.sampled_trajectory.ts, dtype=float)))
        tmax = float(np.max(np.asarray(self.tetra.sampled_trajectory.ts, dtype=float)))
        point_b = tuple(float(v) for v in self.observer_b.tolist())
        scan_samples_now = 257 if (self.prev_corner_batch is None or self.prev_face_batch is None) else 41
        corner_batch = pvis._solve_interpolated_linearized_batch_for_t0(
            interp=self.interp,
            sampled_trajectories=self.corner_traj,
            point_b=point_b,
            t0=float(self.t),
            tmin=tmin,
            tmax=tmax,
            scan_samples=int(scan_samples_now),
            root_max_iter=12,
            root_tol_time=1e-6,
            use_gpu=bool(self.use_gpu.get()),
            batch_size=5000,
            gpu_min_batch=256,
            previous_batch=self.prev_corner_batch,
            previous_t0=self.prev_corner_t0,
        )
        face_batch = pvis._solve_interpolated_linearized_batch_for_t0(
            interp=self.interp,
            sampled_trajectories=self.face_traj,
            point_b=point_b,
            t0=float(self.t),
            tmin=tmin,
            tmax=tmax,
            scan_samples=int(scan_samples_now),
            root_max_iter=12,
            root_tol_time=1e-6,
            use_gpu=bool(self.use_gpu.get()),
            batch_size=5000,
            gpu_min_batch=256,
            previous_batch=self.prev_face_batch,
            previous_t0=self.prev_face_t0,
        )
        self.prev_corner_batch = corner_batch
        self.prev_face_batch = face_batch
        self.prev_corner_t0 = float(self.t)
        self.prev_face_t0 = float(self.t)

        corner_arrival_plus = np.full((4, 3), np.nan, dtype=float)
        corner_arrival_minus = np.full((4, 3), np.nan, dtype=float)
        # Keep emission points NaN until a branch is actually solved for that corner.
        corner_emit_plus = np.full((4, 3), np.nan, dtype=float)
        corner_emit_minus = np.full((4, 3), np.nan, dtype=float)
        for i in range(min(4, len(corner_batch))):
            ep = corner_batch[i].get("plus")
            if isinstance(ep, dict):
                ad = ep.get("arrival_dir_xyz")
                if isinstance(ad, (tuple, list)) and len(ad) == 3:
                    corner_arrival_plus[i, :] = np.asarray(ad, dtype=float)
                pa = ep.get("point_a_m")
                if isinstance(pa, (tuple, list)) and len(pa) == 3:
                    corner_emit_plus[i, :] = np.asarray(pa, dtype=float)
            em = corner_batch[i].get("minus")
            if isinstance(em, dict):
                ad = em.get("arrival_dir_xyz")
                if isinstance(ad, (tuple, list)) and len(ad) == 3:
                    corner_arrival_minus[i, :] = np.asarray(ad, dtype=float)
                pa = em.get("point_a_m")
                if isinstance(pa, (tuple, list)) and len(pa) == 3:
                    corner_emit_minus[i, :] = np.asarray(pa, dtype=float)

        face_dirs = np.zeros(4, dtype=int)
        for i in range(min(4, len(face_batch))):
            e = face_batch[i].get("earliest")
            if isinstance(e, dict):
                face_dirs[i] = int(e.get("direction", 0))

        def _safe_max_xy_norm(arr: np.ndarray, fallback: float) -> float:
            arr2 = np.asarray(arr, dtype=float)
            mask = np.all(np.isfinite(arr2[:, :2]), axis=1)
            if not np.any(mask):
                return float(fallback)
            vals = np.linalg.norm(arr2[mask, :2], axis=1)
            return float(np.max(vals))

        extent = max(
            float(np.linalg.norm(center[:2])) + 2.0 * self.rs,
            _safe_max_xy_norm(corner_emit_plus, fallback=0.0) + 1.5 * self.rs,
            _safe_max_xy_norm(corner_emit_minus, fallback=0.0) + 1.5 * self.rs,
            2.5 * self.rs,
        )

        # Main view: observer camera looking along +x, projected from apparent
        # arrival directions computed by earliest-angle interpolation.
        verts_px_plus, verts_vis_plus = self._project_observer_rays(np.asarray(corner_arrival_plus, dtype=float), w, h)
        verts_px_minus, verts_vis_minus = self._project_observer_rays(
            np.asarray(corner_arrival_minus, dtype=float), w, h
        )
        self.viewer.create_rectangle(0, 0, w, h, fill="#000000", outline="")
        r_b_now = float(np.linalg.norm(self.observer_b))
        sky_arr_p, sky_arr_m, sky_dt_p, sky_dt_m, sky_ok_p, sky_ok_m = self._interp_sky_rows(r_b=r_b_now)
        if sky_arr_p.shape[0] > 0 and self.sky_faces.size > 0:
            sky_px_p, sky_vis_p = self._project_observer_rays(np.asarray(sky_arr_p, dtype=float), w, h)
            sky_px_m, sky_vis_m = self._project_observer_rays(np.asarray(sky_arr_m, dtype=float), w, h)

            if bool(self.show_sky_both.get()):
                for branch_name, px, vis, ok in (
                    ("plus", sky_px_p, sky_vis_p, sky_ok_p),
                    ("minus", sky_px_m, sky_vis_m, sky_ok_m),
                ):
                    for tri in self.sky_faces:
                        idx = np.asarray(tri, dtype=int)
                        if not bool(np.all(ok[idx] & vis[idx])):
                            continue
                        poly = px[idx]
                        color = self._sample_tex_color(np.mean(self.sky_uv[idx], axis=0), branch=branch_name)
                        coords = [coord for p in poly for coord in (float(p[0]), float(p[1]))]
                        self.viewer.create_polygon(coords, fill=color, outline="")
            else:
                for tri in self.sky_faces:
                    idx = np.asarray(tri, dtype=int)
                    vp = bool(np.all(sky_ok_p[idx] & sky_vis_p[idx]))
                    vm = bool(np.all(sky_ok_m[idx] & sky_vis_m[idx]))
                    if (not vp) and (not vm):
                        continue
                    if vp and (not vm):
                        px = sky_px_p
                        br = "plus"
                    elif vm and (not vp):
                        px = sky_px_m
                        br = "minus"
                    else:
                        t_plus = float(np.nanmean(sky_dt_p[idx]))
                        t_minus = float(np.nanmean(sky_dt_m[idx]))
                        if t_plus <= t_minus:
                            px = sky_px_p
                            br = "plus"
                        else:
                            px = sky_px_m
                            br = "minus"
                    poly = px[idx]
                    color = self._sample_tex_color(np.mean(self.sky_uv[idx], axis=0), branch=br)
                    coords = [coord for p in poly for coord in (float(p[0]), float(p[1]))]
                    self.viewer.create_polygon(coords, fill=color, outline="")

        # Sort faces by depth (far -> near), simple painter's algorithm.
        face_depth: list[tuple[float, int]] = []
        for i, tri in enumerate(tris_idx):
            idx = np.asarray(tri, dtype=int)
            rel = verts[idx] - self.observer_b.reshape(1, 3)
            face_depth.append((float(np.mean(rel[:, 0])), i))
        face_depth.sort(reverse=True)

        for _, i in face_depth:
            tri = tris_idx[i]
            idx = np.asarray(tri, dtype=int)
            if bool(np.all(verts_vis_plus[idx])):
                poly = verts_px_plus[idx]
                coords = [coord for p in poly for coord in (float(p[0]), float(p[1]))]
                self.viewer.create_polygon(coords, fill="#50d0ff", outline="#d8f2ff", width=1, stipple="gray50")
            if bool(np.all(verts_vis_minus[idx])):
                poly = verts_px_minus[idx]
                coords = [coord for p in poly for coord in (float(p[0]), float(p[1]))]
                self.viewer.create_polygon(coords, fill="#ff9f5c", outline="#ffe2c8", width=1, stipple="gray50")

        for i in range(4):
            if bool(verts_vis_plus[i]):
                p = verts_px_plus[i]
                r = 3
                self.viewer.create_oval(
                    float(p[0] - r),
                    float(p[1] - r),
                    float(p[0] + r),
                    float(p[1] + r),
                    fill="#83e3ff",
                    outline="",
                )
            if bool(verts_vis_minus[i]):
                p = verts_px_minus[i]
                r = 3
                self.viewer.create_oval(
                    float(p[0] - r),
                    float(p[1] - r),
                    float(p[0] + r),
                    float(p[1] + r),
                    fill="#ffbc7a",
                    outline="",
                )

        # Observer-view guide reticle.
        cx, cy = 0.5 * w, 0.5 * h
        self.viewer.create_line(cx - 10, cy, cx + 10, cy, fill="#ff5fd7")
        self.viewer.create_line(cx, cy - 10, cx, cy + 10, fill="#ff5fd7")
        yaw_deg = float(np.rad2deg(self.view_yaw_rad))
        self.viewer.create_text(12, 12, anchor="nw", fill="#9ec3e8", text=f"Observer view (yaw={yaw_deg:.1f} deg)")

        # Bottom-right panel map (top-down).
        mw = max(10, int(self.map_canvas.winfo_width()))
        mh = max(10, int(self.map_canvas.winfo_height()))
        self.map_canvas.delete("all")
        map_extent = 10.0 * self.rs
        theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        horizon_xy = np.stack([self.rs * np.cos(theta), self.rs * np.sin(theta), np.zeros_like(theta)], axis=1)
        hpx = self._project_topdown(horizon_xy, mw, mh, map_extent)
        hcoords = [coord for p in hpx for coord in (float(p[0]), float(p[1]))]
        self.map_canvas.create_polygon(hcoords, fill="#0f0f14", outline="")
        self.map_canvas.create_line(*hcoords, hcoords[0], hcoords[1], fill="#8ca2bd", width=1, dash=(1, 3))
        rph = 1.5 * self.rs
        photon_xy = np.stack([rph * np.cos(theta), rph * np.sin(theta), np.zeros_like(theta)], axis=1)
        ppx = self._project_topdown(photon_xy, mw, mh, map_extent)
        pcoords = [coord for p in ppx for coord in (float(p[0]), float(p[1]))]
        self.map_canvas.create_line(*pcoords, pcoords[0], pcoords[1], fill="#cfe4ff", width=1)
        verts_map_plus = self._project_topdown(corner_emit_plus, mw, mh, map_extent)
        verts_map_minus = self._project_topdown(corner_emit_minus, mw, mh, map_extent)
        for tri in tris_idx:
            idx = np.asarray(tri, dtype=int)
            poly_p = verts_map_plus[idx]
            if np.all(np.isfinite(poly_p)):
                coords_p = [coord for p in poly_p for coord in (float(p[0]), float(p[1]))]
                self.map_canvas.create_polygon(coords_p, fill="", outline="#8ed8ff", width=1)
            poly_m = verts_map_minus[idx]
            if np.all(np.isfinite(poly_m)):
                coords_m = [coord for p in poly_m for coord in (float(p[0]), float(p[1]))]
                self.map_canvas.create_polygon(coords_m, fill="", outline="#ffb17a", width=1)
        obs_px = self._project_topdown(self.observer_b.reshape(1, 3), mw, mh, map_extent)[0]
        self.map_canvas.create_oval(obs_px[0] - 4, obs_px[1] - 4, obs_px[0] + 4, obs_px[1] + 4, fill="#ff5fd7", outline="")
        # Observer look direction arrow on map using current yaw in map plane.
        look_vec = np.asarray([np.cos(self.view_yaw_rad), np.sin(self.view_yaw_rad), 0.0], dtype=float)
        o2 = self._project_topdown((self.observer_b + (1.2 * self.rs) * look_vec).reshape(1, 3), mw, mh, map_extent)[0]
        self.map_canvas.create_line(obs_px[0], obs_px[1], o2[0], o2[1], fill="#ff5fd7", width=2, arrow="last")

        # Diagnostics when paused: draw geodesic light paths and observer arrival vectors.
        if not self.running:
            b_point = tuple(float(v) for v in self.observer_b.tolist())
            for i in range(min(4, len(corner_batch))):
                for side_name, color in (("plus", "#6fd6ff"), ("minus", "#ffb17a")):
                    ev = corner_batch[i].get(side_name)
                    if not isinstance(ev, dict):
                        continue
                    pa = ev.get("point_a_m")
                    if not (isinstance(pa, (tuple, list)) and len(pa) == 3):
                        continue
                    a_point = tuple(float(v) for v in pa)
                    try:
                        rr = self.bh.find_two_shortest_geodesics(a_point, b_point, a_before_b=True, use_gpu=False)
                    except Exception:
                        continue
                    direction = +1 if side_name == "plus" else -1
                    path = None
                    for p in rr.paths:
                        if int(p.direction) == int(direction):
                            path = p
                            break
                    if path is None:
                        continue

                    r_start = float(np.linalg.norm(np.asarray(a_point, dtype=float)))
                    r_end = float(np.linalg.norm(self.observer_b))
                    th_start = float(np.arctan2(a_point[1], a_point[0]))
                    th_end = float(np.arctan2(self.observer_b[1], self.observer_b[0]))
                    dtheta_short = ((th_end - th_start + pi) % (2.0 * pi)) - pi
                    short_sign = +1.0 if dtheta_short >= 0.0 else -1.0
                    gamma_short = abs(dtheta_short)
                    gamma_long = 2.0 * pi - gamma_short
                    is_short = abs(float(path.target_azimuth_rad) - gamma_short) <= abs(
                        float(path.target_azimuth_rad) - gamma_long
                    )
                    orient_sign = short_sign if is_short else -short_sign
                    try:
                        r_samples, phi_samples = _build_path_profile(
                            bh=self.bh,
                            r_start=r_start,
                            r_end=r_end,
                            impact_b=float(path.impact_parameter_m),
                            target_phi=float(path.target_azimuth_rad),
                            branch=str(path.branch),
                            n=360,
                        )
                    except Exception:
                        continue
                    theta_s = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
                    x = np.asarray(r_samples, dtype=float) * np.cos(theta_s)
                    y = np.asarray(r_samples, dtype=float) * np.sin(theta_s)
                    xyz = np.stack([x, y, np.zeros_like(x)], axis=1)
                    pxy = self._project_topdown(xyz, mw, mh, map_extent)
                    coords = [coord for p in pxy for coord in (float(p[0]), float(p[1]))]
                    self.map_canvas.create_line(*coords, fill=color, width=1)

                    ad = ev.get("arrival_dir_xyz")
                    if isinstance(ad, (tuple, list)) and len(ad) == 3:
                        arr = np.asarray(ad, dtype=float)
                        arr_xy = np.asarray([arr[0], arr[1], 0.0], dtype=float)
                        nrm = float(np.linalg.norm(arr_xy[:2]))
                        if nrm > 1e-12:
                            arr_xy = arr_xy / nrm
                            # Draw look-to-source vector (opposite incoming ray) at observer.
                            p2 = self.observer_b + (-0.9 * self.rs) * arr_xy
                            vv = self._project_topdown(np.stack([self.observer_b, p2], axis=0), mw, mh, map_extent)
                            self.map_canvas.create_line(
                                float(vv[0, 0]),
                                float(vv[0, 1]),
                                float(vv[1, 0]),
                                float(vv[1, 1]),
                                fill=color,
                                width=2,
                                arrow="last",
                            )

        self.lbl_t.configure(text=f"t = {self.t:.3f} s")
        plus_count = int(np.sum(np.all(np.isfinite(corner_arrival_plus), axis=1)))
        minus_count = int(np.sum(np.all(np.isfinite(corner_arrival_minus), axis=1)))
        self.lbl_dir.configure(text=f"corners visible (+/-): {plus_count}/{minus_count}")
        self.lbl_status.configure(text="running" if self.running else "paused")
        self.lbl_sky.configure(text=self._sky_status)


def main() -> None:
    precompute = _resolve_precompute_path(Path("data") / "earliest_angles_precompute_10rs.npz")
    sky_vertices = _resolve_sky_vertices_path(Path("data") / "sky_vertices_precompute_100rs_sub2.npz")
    sky_image = _resolve_sky_image(Path("sky_projections") / "checkerboard_equirect.png")
    root = tk.Tk()
    app = InertialViewerApp(root, precompute_npz=precompute, sky_vertices_npz=sky_vertices, sky_image_path=sky_image)
    del app
    root.mainloop()


if __name__ == "__main__":
    main()
