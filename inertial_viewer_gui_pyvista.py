from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from earliest_visible_interpolated_session import SampledTrajectory3D
from inertial_objects import InertialTetrahedron
from precompute_earliest_grid import PrecomputedEarliestInterpolator

TESTS_DIR = Path(__file__).resolve().parent / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import plot_visibility_from_initial_states as pvis


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


def _resolve_sky_interp_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz",
        Path("earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz"),
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


def _offset_trajectory(base: SampledTrajectory3D, offset_xyz: np.ndarray) -> SampledTrajectory3D:
    off = np.asarray(offset_xyz, dtype=float).reshape(3)
    return SampledTrajectory3D.from_arrays(
        ts=np.asarray(base.ts, dtype=float),
        xs=np.asarray(base.xs, dtype=float) + float(off[0]),
        ys=np.asarray(base.ys, dtype=float) + float(off[1]),
        zs=np.asarray(base.zs, dtype=float) + float(off[2]),
    )


def _expand_faces_with_seam_fix(points: np.ndarray, faces: np.ndarray, uv: np.ndarray):
    tri_pts = np.asarray(points, dtype=float)[faces.reshape(-1)]
    tri_uv = np.asarray(uv, dtype=float)[faces.reshape(-1)]
    tri_uv = tri_uv.reshape(-1, 3, 2)
    u = tri_uv[:, :, 0]
    wraps = (np.max(u, axis=1, keepdims=True) - np.min(u, axis=1, keepdims=True)) > 0.5
    adjust = wraps & (u < 0.5)
    u = np.where(adjust, u + 1.0, u)
    tri_uv[:, :, 0] = u
    tri_uv = tri_uv.reshape(-1, 2)
    n_tri = int(faces.shape[0])
    tri_faces = np.arange(n_tri * 3, dtype=np.int32).reshape(n_tri, 3)
    return tri_pts.astype(np.float32), tri_faces.astype(np.int32), tri_uv.astype(np.float32)


class InertialViewerPyVista:
    def __init__(
        self,
        precompute_npz: Path,
        sky_vertices_npz: Optional[Path],
        sky_interp_npz: Optional[Path],
        sky_image_path: Optional[Path],
    ) -> None:
        try:
            from PyQt5 import QtCore, QtWidgets
            from pyvistaqt import QtInteractor
            import pyvista as pv
        except Exception as exc:
            raise RuntimeError("Install GUI deps: pip install pyvista pyvistaqt PyQt5") from exc

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.QtInteractor = QtInteractor
        self.pv = pv

        self.interp = PrecomputedEarliestInterpolator.from_npz(precompute_npz)
        self.interp.prepare_backend(use_gpu=False)
        self.sky_interp = self.interp
        self.rs = float(self.interp.rs_m)

        traj = _build_center_trajectory(tmin=-200.0, tmax=320.0, samples=4097, radius_m=5.0 * self.rs)
        self.tetra = InertialTetrahedron(
            sampled_trajectory=traj,
            size_light_seconds=0.1,
            rotation_angles_deg=(15.0, 25.0, 10.0),
        )
        self.corner_traj, self.face_traj = self._build_tetra_point_trajectories()
        self.prev_corner_batch = None
        self.prev_face_batch = None
        self.prev_corner_t0: Optional[float] = None
        self.prev_face_t0: Optional[float] = None

        self.observer_b = np.asarray([9.0 * self.rs, 0.0, 0.0], dtype=float)
        self.t = 0.0
        self.running = False

        self.sky_faces = np.zeros((0, 3), dtype=np.int32)
        self.sky_vertices = np.zeros((0, 3), dtype=float)
        self.sky_radius_m = 0.0
        self.sky_uv = np.zeros((0, 2), dtype=np.float32)
        self.sky_status = "sky: disabled"
        self.sky_texture = None
        self._load_sky_data(sky_vertices_npz, sky_interp_npz, sky_image_path)

        self.app = self.QtWidgets.QApplication.instance() or self.QtWidgets.QApplication(sys.argv)
        self.window = self.QtWidgets.QMainWindow()
        self.window.setWindowTitle("Inertial Viewer (PyVista)")

        central = self.QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = self.QtWidgets.QHBoxLayout(central)

        self.plotter = self.QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 3)

        panel = self.QtWidgets.QWidget()
        panel_layout = self.QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel, 1)

        self.btn_start = self.QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self._toggle_start)
        panel_layout.addWidget(self.btn_start)

        self.btn_step = self.QtWidgets.QPushButton("Step")
        self.btn_step.clicked.connect(self._step_once)
        panel_layout.addWidget(self.btn_step)

        self.btn_reset = self.QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        panel_layout.addWidget(self.btn_reset)

        panel_layout.addWidget(self.QtWidgets.QLabel("dt (s)"))
        self.spin_dt = self.QtWidgets.QDoubleSpinBox()
        self.spin_dt.setRange(0.02, 2.0)
        self.spin_dt.setValue(0.25)
        self.spin_dt.setSingleStep(0.01)
        panel_layout.addWidget(self.spin_dt)

        panel_layout.addWidget(self.QtWidgets.QLabel("speed x"))
        self.spin_speed = self.QtWidgets.QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 5.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSingleStep(0.1)
        panel_layout.addWidget(self.spin_speed)

        self.chk_gpu = self.QtWidgets.QCheckBox("Use GPU interpolation")
        self.chk_gpu.setChecked(False)
        panel_layout.addWidget(self.chk_gpu)

        self.chk_sky_both = self.QtWidgets.QCheckBox("Sky: show + and -")
        self.chk_sky_both.setChecked(True)
        self.chk_sky_both.stateChanged.connect(self._draw_frame)
        panel_layout.addWidget(self.chk_sky_both)

        self.lbl_t = self.QtWidgets.QLabel("t = 0.000 s")
        panel_layout.addWidget(self.lbl_t)
        self.lbl_vis = self.QtWidgets.QLabel("corners visible (+/-): 0/0")
        panel_layout.addWidget(self.lbl_vis)
        self.lbl_status = self.QtWidgets.QLabel("paused")
        panel_layout.addWidget(self.lbl_status)
        self.lbl_sky = self.QtWidgets.QLabel(self.sky_status)
        panel_layout.addWidget(self.lbl_sky)
        panel_layout.addStretch(1)

        self.timer = self.QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

        self._setup_scene()
        self._draw_frame()

    def _build_tetra_point_trajectories(self):
        corner_offsets = np.asarray(self.tetra._local_vertices_m, dtype=float)
        base = self.tetra.sampled_trajectory
        corner_traj = [_offset_trajectory(base, off) for off in corner_offsets]
        tri_idx = np.asarray(self.tetra.triangle_indices(), dtype=int)
        face_offsets = np.mean(corner_offsets[tri_idx], axis=1)
        face_traj = [_offset_trajectory(base, off) for off in face_offsets]
        return corner_traj, face_traj

    def _load_sky_data(
        self,
        sky_vertices_npz: Optional[Path],
        sky_interp_npz: Optional[Path],
        sky_image_path: Optional[Path],
    ) -> None:
        if sky_vertices_npz is None or not sky_vertices_npz.exists():
            self.sky_status = "sky: precompute missing"
            return
        try:
            data = np.load(sky_vertices_npz, allow_pickle=False)
            self.sky_faces = np.asarray(data["faces"], dtype=np.int32)
            self.sky_vertices = np.asarray(data["vertices_m"], dtype=float)
            if self.sky_vertices.size > 0:
                self.sky_radius_m = float(np.nanmedian(np.linalg.norm(self.sky_vertices, axis=1)))
            self.sky_uv = np.asarray(data["uv"], dtype=np.float32)
        except Exception:
            self.sky_status = "sky: precompute load failed"
            return

        if sky_interp_npz is not None and sky_interp_npz.exists():
            try:
                self.sky_interp = PrecomputedEarliestInterpolator.from_npz(sky_interp_npz)
                self.sky_interp.prepare_backend(use_gpu=False)
            except Exception:
                self.sky_interp = self.interp
                self.sky_status = "sky: interp table load failed (fallback to main table)"
        else:
            self.sky_interp = self.interp
            self.sky_status = "sky: interp table missing (fallback to main table)"

        if sky_image_path is None or not sky_image_path.exists():
            if "fallback" not in self.sky_status:
                self.sky_status = "sky: image missing"
            return
        try:
            self.sky_texture = self.pv.read_texture(str(sky_image_path))
            try:
                self.sky_texture.repeat = True
            except Exception:
                pass
        except Exception:
            if "fallback" not in self.sky_status:
                self.sky_status = "sky: image load failed"
            return
        if "fallback" in self.sky_status:
            self.sky_status = f"sky: loaded texture/mesh ({self.sky_faces.shape[0]} tris), {self.sky_status}"
        else:
            self.sky_status = f"sky: loaded ({self.sky_faces.shape[0]} tris, fixed-radius interp)"

    def _compute_sky_visibility_interp(self):
        if self.sky_vertices.size == 0:
            n_v = 0
            z3 = np.full((n_v, 3), np.nan, dtype=float)
            z1 = np.full((n_v,), np.nan, dtype=float)
            zb = np.zeros((n_v,), dtype=bool)
            return z3, z3, z1, z1, zb, zb

        b_all = np.repeat(self.observer_b.reshape(1, 3), self.sky_vertices.shape[0], axis=0)
        out = self.sky_interp.interpolate_pairs_3d(
            a_points_m=self.sky_vertices,
            b_points_m=b_all,
            use_gpu=bool(self.chk_gpu.isChecked()),
            batch_size=5000,
        )
        ap = np.asarray(out["arrival_dir_plus_xyz"], dtype=float)
        am = np.asarray(out["arrival_dir_minus_xyz"], dtype=float)
        dp = np.asarray(out["delta_t_plus_s"], dtype=float)
        dm = np.asarray(out["delta_t_minus_s"], dtype=float)
        op = np.asarray(out["ok_plus"], dtype=bool) & np.all(np.isfinite(ap), axis=1)
        om = np.asarray(out["ok_minus"], dtype=bool) & np.all(np.isfinite(am), axis=1)
        return ap, am, dp, dm, op, om

    def _setup_scene(self) -> None:
        self.plotter.set_background("black")
        self.plotter.add_mesh(
            self.pv.Sphere(radius=float(self.rs), center=(0.0, 0.0, 0.0)),
            style="wireframe",
            color="#808080",
            opacity=0.6,
            name="horizon",
        )
        self.plotter.add_mesh(
            self.pv.Sphere(radius=float(1.5 * self.rs), center=(0.0, 0.0, 0.0)),
            style="wireframe",
            color="#b0c4de",
            opacity=0.4,
            name="photon",
        )
        self.plotter.add_points(
            self.observer_b.reshape(1, 3),
            color="#ff44cc",
            point_size=16,
            render_points_as_spheres=True,
            name="observer",
        )
        self.plotter.camera.position = tuple(self.observer_b.tolist())
        self.plotter.camera.focal_point = tuple((self.observer_b + np.asarray([-1.0 * self.rs, 0.0, 0.0])).tolist())
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        self.plotter.camera.clipping_range = (0.01 * self.rs, 1000.0 * self.rs)

    def _toggle_start(self) -> None:
        self.running = not self.running
        self.btn_start.setText("Stop" if self.running else "Start")
        if self.running:
            self.timer.start()
        else:
            self.timer.stop()
        self._draw_frame()

    def _tick(self) -> None:
        if not self.running:
            return
        self._step_once()

    def _step_once(self) -> None:
        self.t += float(self.spin_dt.value()) * float(self.spin_speed.value())
        self._draw_frame()

    def _reset(self) -> None:
        self.t = 0.0
        self.prev_corner_batch = None
        self.prev_face_batch = None
        self.prev_corner_t0 = None
        self.prev_face_t0 = None
        self._draw_frame()

    def _orient_sky_look_dirs(self, arr_dirs: np.ndarray, source_points: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr_dirs, dtype=float)
        src = np.asarray(source_points, dtype=float)
        look = -arr
        if arr.shape != src.shape:
            return look
        target = src - self.observer_b.reshape(1, 3)
        tn = np.linalg.norm(target, axis=1, keepdims=True)
        t_hat = np.where(tn > 1e-12, target / np.maximum(tn, 1e-12), 0.0)
        dotv = np.sum(look * t_hat, axis=1)
        flip = np.isfinite(dotv) & (dotv < 0.0)
        if np.any(flip):
            look[flip] *= -1.0
        return look

    def _build_ray_mesh(self, arr_dirs: np.ndarray, valid_mask: np.ndarray, faces: np.ndarray):
        look = self._orient_sky_look_dirs(arr_dirs=arr_dirs, source_points=self.sky_vertices)
        pts_world = np.full_like(look, np.nan, dtype=float)
        r = float(max(self.sky_radius_m, 1e-9))
        o = np.asarray(self.observer_b, dtype=float)
        oo = float(np.dot(o, o))
        c = oo - (r * r)
        for i in range(look.shape[0]):
            d = look[i]
            if not np.all(np.isfinite(d)):
                continue
            dn = float(np.linalg.norm(d))
            if dn <= 1e-12:
                continue
            d = d / dn
            b = 2.0 * float(np.dot(o, d))
            disc = b * b - 4.0 * c
            if disc < 0.0:
                continue
            sdisc = float(np.sqrt(max(0.0, disc)))
            t1 = 0.5 * (-b - sdisc)
            t2 = 0.5 * (-b + sdisc)
            t_pos = [tt for tt in (t1, t2) if tt > 0.0]
            if not t_pos:
                continue
            t_hit = float(min(t_pos))
            pts_world[i, :] = o + t_hit * d
        good_faces = []
        for tri in np.asarray(faces, dtype=np.int32):
            idx = np.asarray(tri, dtype=np.int32)
            if bool(np.all(valid_mask[idx])) and bool(np.all(np.isfinite(pts_world[idx]))):
                good_faces.append(idx)
        if not good_faces:
            return None
        good_faces = np.asarray(good_faces, dtype=np.int32)
        tri_pts, tri_faces, tri_uv = _expand_faces_with_seam_fix(pts_world, good_faces, self.sky_uv)
        face_cells = np.hstack([np.full((tri_faces.shape[0], 1), 3, dtype=np.int32), tri_faces]).ravel()
        mesh = self.pv.PolyData(tri_pts, face_cells.astype(np.int32))
        mesh.active_texture_coordinates = tri_uv
        return mesh

    def _build_tetra_branch_mesh(self, arrival_dirs: np.ndarray, tri_idx: np.ndarray, scale: float):
        d = np.asarray(arrival_dirs, dtype=float)
        ok = np.all(np.isfinite(d), axis=1)
        if not np.any(ok):
            return None
        pts = self.observer_b.reshape(1, 3) + scale * (-d)
        faces = []
        for tri in np.asarray(tri_idx, dtype=np.int32):
            idx = np.asarray(tri, dtype=np.int32)
            if bool(np.all(ok[idx])):
                faces.append(idx)
        if not faces:
            return None
        faces = np.asarray(faces, dtype=np.int32)
        face_cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
        return self.pv.PolyData(pts.astype(np.float32), face_cells.astype(np.int32))

    def _draw_frame(self) -> None:
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
            use_gpu=bool(self.chk_gpu.isChecked()),
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
            use_gpu=bool(self.chk_gpu.isChecked()),
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
        for i in range(min(4, len(corner_batch))):
            ep = corner_batch[i].get("plus")
            if isinstance(ep, dict):
                ad = ep.get("arrival_dir_xyz")
                if isinstance(ad, (tuple, list)) and len(ad) == 3:
                    corner_arrival_plus[i, :] = np.asarray(ad, dtype=float)
            em = corner_batch[i].get("minus")
            if isinstance(em, dict):
                ad = em.get("arrival_dir_xyz")
                if isinstance(ad, (tuple, list)) and len(ad) == 3:
                    corner_arrival_minus[i, :] = np.asarray(ad, dtype=float)

        tri_idx = np.asarray(self.tetra.triangle_indices(), dtype=np.int32)
        plus_mesh = self._build_tetra_branch_mesh(corner_arrival_plus, tri_idx, scale=3.0 * self.rs)
        minus_mesh = self._build_tetra_branch_mesh(corner_arrival_minus, tri_idx, scale=3.0 * self.rs)
        if plus_mesh is not None:
            self.plotter.add_mesh(
                plus_mesh,
                color="#50d0ff",
                opacity=0.45,
                show_edges=True,
                edge_color="#d8f2ff",
                name="tetra_plus",
                lighting=False,
            )
        else:
            self.plotter.remove_actor("tetra_plus", render=False)
        if minus_mesh is not None:
            self.plotter.add_mesh(
                minus_mesh,
                color="#ff9f5c",
                opacity=0.45,
                show_edges=True,
                edge_color="#ffe2c8",
                name="tetra_minus",
                lighting=False,
            )
        else:
            self.plotter.remove_actor("tetra_minus", render=False)

        sky_arr_p, sky_arr_m, sky_dt_p, sky_dt_m, sky_ok_p, sky_ok_m = self._compute_sky_visibility_interp()
        if self.sky_texture is not None and self.sky_faces.size > 0:
            if bool(self.chk_sky_both.isChecked()):
                mesh_p = self._build_ray_mesh(sky_arr_p, sky_ok_p, self.sky_faces)
                mesh_m = self._build_ray_mesh(sky_arr_m, sky_ok_m, self.sky_faces)
                if mesh_p is not None:
                    self.plotter.add_mesh(
                        mesh_p,
                        texture=self.sky_texture,
                        opacity=0.50,
                        show_edges=False,
                        name="sky_plus",
                        lighting=False,
                    )
                else:
                    self.plotter.remove_actor("sky_plus", render=False)
                if mesh_m is not None:
                    self.plotter.add_mesh(
                        mesh_m,
                        texture=self.sky_texture,
                        opacity=0.35,
                        show_edges=False,
                        name="sky_minus",
                        lighting=False,
                    )
                else:
                    self.plotter.remove_actor("sky_minus", render=False)
            else:
                pick_plus = sky_ok_p & ((~sky_ok_m) | (sky_dt_p <= sky_dt_m))
                pick_minus = sky_ok_m & (~pick_plus)
                arr_one = np.where(pick_plus[:, None], sky_arr_p, sky_arr_m)
                ok_one = pick_plus | pick_minus
                mesh_one = self._build_ray_mesh(arr_one, ok_one, self.sky_faces)
                if mesh_one is not None:
                    self.plotter.add_mesh(
                        mesh_one,
                        texture=self.sky_texture,
                        opacity=0.55,
                        show_edges=False,
                        name="sky_single",
                        lighting=False,
                    )
                else:
                    self.plotter.remove_actor("sky_single", render=False)
                self.plotter.remove_actor("sky_plus", render=False)
                self.plotter.remove_actor("sky_minus", render=False)
        else:
            self.plotter.remove_actor("sky_plus", render=False)
            self.plotter.remove_actor("sky_minus", render=False)
            self.plotter.remove_actor("sky_single", render=False)

        plus_count = int(np.sum(np.all(np.isfinite(corner_arrival_plus), axis=1)))
        minus_count = int(np.sum(np.all(np.isfinite(corner_arrival_minus), axis=1)))
        self.lbl_t.setText(f"t = {self.t:.3f} s")
        self.lbl_vis.setText(f"corners visible (+/-): {plus_count}/{minus_count}")
        self.lbl_status.setText("running" if self.running else "paused")
        self.lbl_sky.setText(self.sky_status)
        self.plotter.render()

    def show(self) -> None:
        self.window.resize(1500, 900)
        self.window.show()
        self.app.exec_()


def main() -> None:
    precompute = _resolve_precompute_path(Path("data") / "earliest_angles_precompute_10rs.npz")
    sky_vertices = _resolve_sky_vertices_path(Path("data") / "sky_vertices_precompute_100rs_sub2.npz")
    sky_interp = _resolve_sky_interp_path(Path("data") / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz")
    sky_image = _resolve_sky_image(Path("sky_projections") / "checkerboard_equirect.png")
    ui = InertialViewerPyVista(
        precompute_npz=precompute,
        sky_vertices_npz=sky_vertices,
        sky_interp_npz=sky_interp,
        sky_image_path=sky_image,
    )
    ui.show()


if __name__ == "__main__":
    main()
