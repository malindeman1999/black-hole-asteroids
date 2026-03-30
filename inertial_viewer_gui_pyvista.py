from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from earliest_visible_interpolated_session import SampledTrajectory3D
from inertial_objects import InertialTetrahedron


def _resolve_sky_vertices_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "sky_vertices_precompute_100rs_sub2.npz",
        Path("sky_vertices_precompute_100rs_sub2.npz"),
    ]
    return next((p for p in fallbacks if p.exists()), None)


def _resolve_ray_local_interp_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("ray_tracing") / "multi_sequences_interp_table.npz",
        Path("data") / "multi_sequences_interp_table.npz",
        Path("multi_sequences_interp_table.npz"),
    ]
    return next((p for p in fallbacks if p.exists()), None)


def _resolve_ray_sky_interp_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    fallbacks = [
        Path("ray_tracing") / "sky_interp_table.npz",
        Path("data") / "sky_interp_table.npz",
        Path("sky_interp_table.npz"),
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
        ray_local_interp_npz: Optional[Path],
        ray_sky_interp_npz: Optional[Path],
        sky_vertices_npz: Optional[Path],
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
        self.rs = 1.0
        self.ray_local_enabled = False
        self.ray_sky_enabled = False
        self.ray_local_status = "ray local: disabled"
        self.ray_sky_status = "ray sky: disabled"
        self.ray_local_meta = {}
        self.ray_sky_meta = {}
        self.ray_local_b = np.zeros((0,), dtype=float)
        self.ray_local_r = np.zeros((0,), dtype=float)
        self.ray_local_th = np.zeros((0,), dtype=float)
        self.ray_local_look = np.zeros((0, 0, 0, 2), dtype=float)
        self.ray_local_dt = np.zeros((0, 0, 0), dtype=float)
        self.ray_local_valid = np.zeros((0, 0, 0), dtype=bool)
        self.ray_sky_b = np.zeros((0,), dtype=float)
        self.ray_sky_sky = np.zeros((0, 0, 2), dtype=float)
        self.ray_sky_look = np.zeros((0, 0, 2), dtype=float)
        self.ray_sky_dt = np.zeros((0, 0), dtype=float)
        self.ray_sky_valid = np.zeros((0, 0), dtype=bool)
        self._sky_cache = None

        self._load_ray_tracing_tables(ray_local_interp_npz, ray_sky_interp_npz)
        if not self.ray_local_enabled or not self.ray_sky_enabled:
            raise RuntimeError("Ray-tracing local/sky interpolation tables are required. Build them in ray_tracing GUI first.")

        rs_guess = None
        src_local = self.ray_local_meta.get("source_metadata", {}) if isinstance(self.ray_local_meta, dict) else {}
        src_sky = self.ray_sky_meta.get("source_metadata", {}) if isinstance(self.ray_sky_meta, dict) else {}
        if isinstance(src_local, dict) and "rs_m" in src_local:
            rs_guess = float(src_local["rs_m"])
        elif isinstance(src_sky, dict) and "rs_m" in src_sky:
            rs_guess = float(src_sky["rs_m"])
        self.rs = float(rs_guess) if (rs_guess is not None and np.isfinite(rs_guess) and rs_guess > 0.0) else 1.0

        traj = _build_center_trajectory(tmin=-200.0, tmax=320.0, samples=4097, radius_m=5.0 * self.rs)
        self.tetra = InertialTetrahedron(
            sampled_trajectory=traj,
            size_light_seconds=0.1,
            rotation_angles_deg=(15.0, 25.0, 10.0),
        )
        self.corner_traj, self.face_traj = self._build_tetra_point_trajectories()

        self.observer_b = np.asarray([9.0 * self.rs, 0.0, 0.0], dtype=float)
        self.t = 0.0
        self.running = False

        self.sky_faces = np.zeros((0, 3), dtype=np.int32)
        self.sky_vertices = np.zeros((0, 3), dtype=float)
        self.sky_radius_m = 0.0
        self.sky_uv = np.zeros((0, 2), dtype=np.float32)
        self.sky_status = "sky: disabled"
        self.sky_texture = None
        self._load_sky_data(sky_vertices_npz, sky_image_path)

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
        self.lbl_ray = self.QtWidgets.QLabel(f"{self.ray_local_status} | {self.ray_sky_status}")
        panel_layout.addWidget(self.lbl_ray)
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

    def _load_ray_tracing_tables(self, local_npz: Optional[Path], sky_npz: Optional[Path]) -> None:
        self.ray_local_enabled = False
        self.ray_sky_enabled = False
        self.ray_local_status = "ray local: missing"
        self.ray_sky_status = "ray sky: missing"
        if local_npz is not None and local_npz.exists():
            try:
                data = np.load(local_npz, allow_pickle=True)
                self.ray_local_b = np.asarray(data["b_values_rs"], dtype=float)
                self.ray_local_r = np.asarray(data["r_values_rs"], dtype=float)
                self.ray_local_th = np.asarray(data["theta_values_deg"], dtype=float)
                self.ray_local_look = np.asarray(data["lookback_unit_xy"], dtype=float)
                self.ray_local_dt = np.asarray(data["back_time_s"], dtype=float)
                self.ray_local_valid = np.asarray(data["valid"], dtype=bool)
                meta_raw = data.get("metadata_json", np.asarray("", dtype=object))
                self.ray_local_meta = json.loads(str(meta_raw.item())) if np.size(meta_raw) > 0 else {}
                self.ray_local_enabled = (
                    self.ray_local_b.size > 0
                    and self.ray_local_r.size > 0
                    and self.ray_local_th.size > 0
                    and self.ray_local_look.ndim == 4
                    and self.ray_local_dt.ndim == 3
                    and self.ray_local_valid.ndim == 3
                )
                self.ray_local_status = f"ray local: loaded ({self.ray_local_b.size} B rows)"
            except Exception:
                self.ray_local_enabled = False
                self.ray_local_status = "ray local: load failed"
        if sky_npz is not None and sky_npz.exists():
            try:
                data = np.load(sky_npz, allow_pickle=True)
                self.ray_sky_b = np.asarray(data["b_values_rs"], dtype=float)
                self.ray_sky_sky = np.asarray(data["sky_unit_xy"], dtype=float)
                self.ray_sky_look = np.asarray(data["lookback_unit_xy"], dtype=float)
                self.ray_sky_dt = np.asarray(data["back_time_s"], dtype=float)
                self.ray_sky_valid = np.asarray(data["valid"], dtype=bool)
                meta_raw = data.get("metadata_json", np.asarray("", dtype=object))
                self.ray_sky_meta = json.loads(str(meta_raw.item())) if np.size(meta_raw) > 0 else {}
                self.ray_sky_enabled = (
                    self.ray_sky_b.size > 0
                    and self.ray_sky_sky.ndim == 3
                    and self.ray_sky_look.ndim == 3
                    and self.ray_sky_dt.ndim == 2
                    and self.ray_sky_valid.ndim == 2
                )
                self.ray_sky_status = f"ray sky: loaded ({self.ray_sky_b.size} B rows)"
            except Exception:
                self.ray_sky_enabled = False
                self.ray_sky_status = "ray sky: load failed"

    def _load_sky_data(
        self,
        sky_vertices_npz: Optional[Path],
        sky_image_path: Optional[Path],
    ) -> None:
        if sky_vertices_npz is None or not sky_vertices_npz.exists():
            self.sky_status = "sky: mesh missing"
            return
        try:
            data = np.load(sky_vertices_npz, allow_pickle=False)
            self.sky_faces = np.asarray(data["faces"], dtype=np.int32)
            self.sky_vertices = np.asarray(data["vertices_m"], dtype=float)
            if self.sky_vertices.size > 0:
                self.sky_radius_m = float(np.nanmedian(np.linalg.norm(self.sky_vertices, axis=1)))
            self.sky_uv = np.asarray(data["uv"], dtype=np.float32)
        except Exception:
            self.sky_status = "sky: mesh load failed"
            return

        if sky_image_path is None or not sky_image_path.exists():
            self.sky_status = "sky: image missing"
            return
        try:
            self.sky_texture = self.pv.read_texture(str(sky_image_path))
            try:
                self.sky_texture.repeat = True
            except Exception:
                pass
        except Exception:
            self.sky_status = "sky: image load failed"
            return
        self.sky_status = f"sky: loaded ({self.sky_faces.shape[0]} tris, ray-table mapping)"

    @staticmethod
    def _normalize_xy(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        n = np.linalg.norm(arr, axis=-1, keepdims=True)
        return np.where(n > 1e-12, arr / np.maximum(n, 1e-12), np.nan)

    @staticmethod
    def _nearest_b_rows(b_values: np.ndarray, b_query: float) -> tuple[int, int, float]:
        b = np.asarray(b_values, dtype=float)
        if b.size == 1:
            return 0, 0, 0.0
        if b_query <= float(b[0]):
            return 0, 0, 0.0
        if b_query >= float(b[-1]):
            n = int(b.size - 1)
            return n, n, 0.0
        hi = int(np.searchsorted(b, b_query, side="left"))
        lo = max(0, hi - 1)
        b0 = float(b[lo])
        b1 = float(b[hi])
        wb = 0.0 if abs(b1 - b0) <= 1e-12 else (float(b_query) - b0) / (b1 - b0)
        return lo, hi, float(np.clip(wb, 0.0, 1.0))

    def _lookup_local_row(self, bi: int, x_rs: float, y_rs: float) -> tuple[bool, np.ndarray, float]:
        r = float(np.hypot(x_rs, y_rs))
        th = float(np.rad2deg(np.arctan2(y_rs, x_rs)))
        if th < 0.0:
            th += 360.0

        r_grid = np.asarray(self.ray_local_r, dtype=float)
        th_grid = np.asarray(self.ray_local_th, dtype=float)
        if r_grid.size < 1 or th_grid.size < 1:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")
        if r < float(r_grid[0]) or r > float(r_grid[-1]) or th < float(th_grid[0]) or th > float(th_grid[-1]):
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")

        ir1 = int(np.searchsorted(r_grid, r, side="left"))
        it1 = int(np.searchsorted(th_grid, th, side="left"))
        ir0 = max(0, ir1 - 1)
        it0 = max(0, it1 - 1)
        ir1 = min(ir1, int(r_grid.size - 1))
        it1 = min(it1, int(th_grid.size - 1))

        r0 = float(r_grid[ir0]); r1 = float(r_grid[ir1])
        t0 = float(th_grid[it0]); t1 = float(th_grid[it1])
        wr = 0.0 if abs(r1 - r0) <= 1e-12 else (r - r0) / (r1 - r0)
        wt = 0.0 if abs(t1 - t0) <= 1e-12 else (th - t0) / (t1 - t0)
        wr = float(np.clip(wr, 0.0, 1.0)); wt = float(np.clip(wt, 0.0, 1.0))

        corners = [
            (ir0, it0, (1.0 - wr) * (1.0 - wt)),
            (ir1, it0, wr * (1.0 - wt)),
            (ir0, it1, (1.0 - wr) * wt),
            (ir1, it1, wr * wt),
        ]
        look = np.zeros(2, dtype=float)
        bt = 0.0
        wsum = 0.0
        for ii, jj, ww in corners:
            if ww <= 0.0:
                continue
            if not bool(self.ray_local_valid[bi, ii, jj]):
                continue
            lv = np.asarray(self.ray_local_look[bi, ii, jj, :], dtype=float)
            tv = float(self.ray_local_dt[bi, ii, jj])
            if not np.all(np.isfinite(lv)) or not np.isfinite(tv):
                continue
            look += ww * lv
            bt += ww * tv
            wsum += ww
        if wsum <= 1e-12:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")
        look /= wsum
        bt /= wsum
        n = float(np.hypot(look[0], look[1]))
        if n <= 1e-12:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")
        look /= n
        return bool(np.all(np.isfinite(look)) and np.isfinite(bt)), look, float(bt)

    def _lookup_local_two_family(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(points_xyz, dtype=float)
        n = int(p.shape[0])
        arr_p = np.full((n, 3), np.nan, dtype=float)
        arr_m = np.full((n, 3), np.nan, dtype=float)
        dt_p = np.full((n,), np.nan, dtype=float)
        dt_m = np.full((n,), np.nan, dtype=float)
        ok_p = np.zeros((n,), dtype=bool)
        ok_m = np.zeros((n,), dtype=bool)
        if not self.ray_local_enabled or n == 0:
            return arr_p, arr_m, dt_p, dt_m, ok_p, ok_m

        b_query = float(np.hypot(self.observer_b[0], self.observer_b[1]) / max(self.rs, 1e-12))
        b0, b1, wb = self._nearest_b_rows(self.ray_local_b, b_query)

        def lookup_blend(x_rs: float, y_rs: float) -> tuple[bool, np.ndarray, float]:
            ok0, l0, t0 = self._lookup_local_row(b0, x_rs, y_rs)
            if b1 == b0:
                return ok0, l0, t0
            ok1, l1, t1 = self._lookup_local_row(b1, x_rs, y_rs)
            if ok0 and ok1:
                l = (1.0 - wb) * l0 + wb * l1
                nrm = float(np.hypot(l[0], l[1]))
                if nrm > 1e-12:
                    l = l / nrm
                return True, l, float((1.0 - wb) * t0 + wb * t1)
            if ok0:
                return ok0, l0, t0
            return ok1, l1, t1

        for i in range(n):
            x_rs = float(p[i, 0] / max(self.rs, 1e-12))
            y_rs = float(p[i, 1] / max(self.rs, 1e-12))
            ok, look, bt = lookup_blend(x_rs, y_rs)
            if ok:
                arr_p[i, :2] = -look
                arr_p[i, 2] = 0.0
                dt_p[i] = bt
                ok_p[i] = True

            # Mirrored family: flip input across x-axis, evaluate, then flip output back.
            okf, lookf, btf = lookup_blend(x_rs, -y_rs)
            if okf:
                look_m = np.asarray([lookf[0], -lookf[1]], dtype=float)
                arr_m[i, :2] = -look_m
                arr_m[i, 2] = 0.0
                dt_m[i] = btf
                ok_m[i] = True
        return arr_p, arr_m, dt_p, dt_m, ok_p, ok_m

    def _lookup_sky_two_family(self, sky_query_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self._normalize_xy(np.asarray(sky_query_xy, dtype=float))
        n = int(q.shape[0])
        arr_p = np.full((n, 3), np.nan, dtype=float)
        arr_m = np.full((n, 3), np.nan, dtype=float)
        dt_p = np.full((n,), np.nan, dtype=float)
        dt_m = np.full((n,), np.nan, dtype=float)
        ok_p = np.zeros((n,), dtype=bool)
        ok_m = np.zeros((n,), dtype=bool)
        if not self.ray_sky_enabled or n == 0:
            return arr_p, arr_m, dt_p, dt_m, ok_p, ok_m

        b_query = float(np.hypot(self.observer_b[0], self.observer_b[1]) / max(self.rs, 1e-12))
        b0, b1, wb = self._nearest_b_rows(self.ray_sky_b, b_query)

        def solve_row(bi: int, qxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            ok = np.asarray(self.ray_sky_valid[bi], dtype=bool)
            s = np.asarray(self.ray_sky_sky[bi], dtype=float)[ok]
            l = np.asarray(self.ray_sky_look[bi], dtype=float)[ok]
            t = np.asarray(self.ray_sky_dt[bi], dtype=float)[ok]
            out_ok = np.zeros((qxy.shape[0],), dtype=bool)
            out_l = np.full((qxy.shape[0], 2), np.nan, dtype=float)
            out_t = np.full((qxy.shape[0],), np.nan, dtype=float)
            if s.shape[0] < 1:
                return out_ok, out_l, out_t

            sang = np.mod(np.rad2deg(np.arctan2(s[:, 1], s[:, 0])) + 360.0, 360.0)
            order = np.argsort(sang)
            sang = sang[order]; s = s[order]; l = l[order]; t = t[order]
            qang = np.mod(np.rad2deg(np.arctan2(qxy[:, 1], qxy[:, 0])) + 360.0, 360.0)
            in_rng = (qang >= float(sang[0]) - 1e-6) & (qang <= float(sang[-1]) + 1e-6)
            idx = np.where(in_rng)[0]
            if idx.size == 0:
                return out_ok, out_l, out_t
            qq = qang[idx]
            hi = np.searchsorted(sang, qq, side="left")
            hi = np.clip(hi, 0, sang.size - 1)
            lo = np.clip(hi - 1, 0, sang.size - 1)
            d0 = np.abs(qq - sang[lo])
            d1 = np.abs(sang[hi] - qq)
            same = (lo == hi)
            w0 = np.where(same, 1.0, 1.0 / np.maximum(d0, 1e-9))
            w1 = np.where(same, 0.0, 1.0 / np.maximum(d1, 1e-9))
            wsum = np.maximum(w0 + w1, 1e-12)
            w0 /= wsum; w1 /= wsum
            lv = w0[:, None] * l[lo] + w1[:, None] * l[hi]
            nv = np.linalg.norm(lv, axis=1, keepdims=True)
            lv = np.where(nv > 1e-12, lv / np.maximum(nv, 1e-12), np.nan)
            tv = w0 * t[lo] + w1 * t[hi]
            good = np.all(np.isfinite(lv), axis=1) & np.isfinite(tv)
            out_ok[idx] = good
            out_l[idx] = lv
            out_t[idx] = tv
            return out_ok, out_l, out_t

        # Plus family.
        ok0, l0, t0 = solve_row(b0, q)
        if b1 == b0:
            ok_p = ok0
            look_p = l0
            dt_p = t0
        else:
            ok1, l1, t1 = solve_row(b1, q)
            both = ok0 & ok1
            only0 = ok0 & (~ok1)
            only1 = ok1 & (~ok0)
            look_p = np.full_like(l0, np.nan)
            dtp = np.full_like(t0, np.nan)
            if np.any(both):
                mix = (1.0 - wb) * l0[both] + wb * l1[both]
                nrm = np.linalg.norm(mix, axis=1, keepdims=True)
                mix = np.where(nrm > 1e-12, mix / np.maximum(nrm, 1e-12), np.nan)
                look_p[both] = mix
                dtp[both] = (1.0 - wb) * t0[both] + wb * t1[both]
            look_p[only0] = l0[only0]; dtp[only0] = t0[only0]
            look_p[only1] = l1[only1]; dtp[only1] = t1[only1]
            ok_p = both | only0 | only1
            dt_p = dtp
        arr_p[:, :2] = -look_p
        arr_p[:, 2] = 0.0
        arr_p[~ok_p] = np.nan

        # Mirrored family from flipped input, then flip output back.
        qf = np.asarray(q, dtype=float)
        qf[:, 1] *= -1.0
        ok0, l0, t0 = solve_row(b0, qf)
        if b1 == b0:
            okm = ok0
            look_m = l0
            dtm = t0
        else:
            ok1, l1, t1 = solve_row(b1, qf)
            both = ok0 & ok1
            only0 = ok0 & (~ok1)
            only1 = ok1 & (~ok0)
            look_m = np.full_like(l0, np.nan)
            dtm = np.full_like(t0, np.nan)
            if np.any(both):
                mix = (1.0 - wb) * l0[both] + wb * l1[both]
                nrm = np.linalg.norm(mix, axis=1, keepdims=True)
                mix = np.where(nrm > 1e-12, mix / np.maximum(nrm, 1e-12), np.nan)
                look_m[both] = mix
                dtm[both] = (1.0 - wb) * t0[both] + wb * t1[both]
            look_m[only0] = l0[only0]; dtm[only0] = t0[only0]
            look_m[only1] = l1[only1]; dtm[only1] = t1[only1]
            okm = both | only0 | only1
        look_m[:, 1] *= -1.0
        ok_m = okm
        dt_m = dtm
        arr_m[:, :2] = -look_m
        arr_m[:, 2] = 0.0
        arr_m[~ok_m] = np.nan
        return arr_p, arr_m, dt_p, dt_m, ok_p, ok_m

    def _compute_sky_visibility_interp(self):
        if self.sky_vertices.size == 0:
            n_v = 0
            z3 = np.full((n_v, 3), np.nan, dtype=float)
            z1 = np.full((n_v,), np.nan, dtype=float)
            zb = np.zeros((n_v,), dtype=bool)
            return z3, z3, z1, z1, zb, zb
        cache_key = (
            float(self.observer_b[0]),
            float(self.observer_b[1]),
            float(self.observer_b[2]),
            int(self.sky_vertices.shape[0]),
        )
        if isinstance(self._sky_cache, tuple) and len(self._sky_cache) == 2 and self._sky_cache[0] == cache_key:
            return self._sky_cache[1]
        to_sky = np.asarray(self.sky_vertices, dtype=float) - self.observer_b.reshape(1, 3)
        qxy = self._normalize_xy(to_sky[:, :2])
        out = self._lookup_sky_two_family(qxy)
        self._sky_cache = (cache_key, out)
        return out

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

    def _solve_local_branch_for_trajectory(
        self,
        sampled: SampledTrajectory3D,
        t0: float,
        side: str,
        tmin: float,
        tmax: float,
        scan_samples: int,
    ) -> Optional[dict]:
        t_hi = min(float(tmax), float(t0))
        t_lo = float(tmin)
        if t_hi <= t_lo:
            return None
        ts = np.linspace(t_lo, t_hi, max(3, int(scan_samples)), dtype=float)
        pts = sampled.eval_points(ts)
        ap, am, dp, dm, op, om = self._lookup_local_two_family(pts)
        if side == "plus":
            ok = op
            dt = dp
            arr = ap
        else:
            ok = om
            dt = dm
            arr = am
        ok = np.asarray(ok, dtype=bool) & np.isfinite(dt)
        if np.count_nonzero(ok) < 2:
            return None
        f = ts + dt - float(t0)
        idx = np.where(ok[:-1] & ok[1:])[0]
        roots = []
        for i in idx:
            f0 = float(f[i]); f1 = float(f[i + 1])
            if (f0 == 0.0) or (f1 == 0.0) or (f0 * f1 <= 0.0):
                denom = (f1 - f0)
                if abs(denom) <= 1e-12:
                    te = float(ts[i])
                else:
                    te = float(ts[i] - f0 * (ts[i + 1] - ts[i]) / denom)
                roots.append(float(np.clip(te, ts[i], ts[i + 1])))
        if roots:
            te = float(min(roots))
        else:
            cand = np.where(ok)[0]
            if cand.size == 0:
                return None
            te = float(ts[int(cand[np.argmin(np.abs(f[cand]))])])
        p = sampled.eval_points(np.asarray([te], dtype=float))[0:1]
        ap1, am1, dp1, dm1, op1, om1 = self._lookup_local_two_family(p)
        if side == "plus":
            if not bool(op1[0]) or (not np.isfinite(dp1[0])) or (not np.all(np.isfinite(ap1[0]))):
                return None
            return {
                "emission_time_s": float(te),
                "delta_t_s": float(dp1[0]),
                "arrival_dir_xyz": tuple(float(v) for v in ap1[0].tolist()),
                "point_a_xyz": tuple(float(v) for v in p[0].tolist()),
            }
        if not bool(om1[0]) or (not np.isfinite(dm1[0])) or (not np.all(np.isfinite(am1[0]))):
            return None
        return {
            "emission_time_s": float(te),
            "delta_t_s": float(dm1[0]),
            "arrival_dir_xyz": tuple(float(v) for v in am1[0].tolist()),
            "point_a_xyz": tuple(float(v) for v in p[0].tolist()),
        }

    def _solve_local_batch_for_t0(self, sampled_trajectories: list[SampledTrajectory3D], t0: float, tmin: float, tmax: float, scan_samples: int) -> list[dict]:
        out = []
        for sampled in sampled_trajectories:
            plus = self._solve_local_branch_for_trajectory(sampled, t0=t0, side="plus", tmin=tmin, tmax=tmax, scan_samples=scan_samples)
            minus = self._solve_local_branch_for_trajectory(sampled, t0=t0, side="minus", tmin=tmin, tmax=tmax, scan_samples=scan_samples)
            earliest = None
            if isinstance(plus, dict) and isinstance(minus, dict):
                earliest = plus if float(plus["emission_time_s"]) <= float(minus["emission_time_s"]) else minus
            elif isinstance(plus, dict):
                earliest = plus
            elif isinstance(minus, dict):
                earliest = minus
            out.append({"plus": plus, "minus": minus, "earliest": earliest})
        return out

    def _draw_frame(self) -> None:
        tmin = float(np.min(np.asarray(self.tetra.sampled_trajectory.ts, dtype=float)))
        tmax = float(np.max(np.asarray(self.tetra.sampled_trajectory.ts, dtype=float)))
        scan_samples_now = 81
        corner_batch = self._solve_local_batch_for_t0(
            sampled_trajectories=self.corner_traj,
            t0=float(self.t),
            tmin=tmin,
            tmax=tmax,
            scan_samples=int(scan_samples_now),
        )

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
        self.lbl_ray.setText(f"{self.ray_local_status} | {self.ray_sky_status}")
        self.plotter.render()

    def show(self) -> None:
        self.window.resize(1500, 900)
        self.window.show()
        self.app.exec_()


def main() -> None:
    ray_local_interp = _resolve_ray_local_interp_path(Path("ray_tracing") / "multi_sequences_interp_table.npz")
    ray_sky_interp = _resolve_ray_sky_interp_path(Path("ray_tracing") / "sky_interp_table.npz")
    sky_vertices = _resolve_sky_vertices_path(Path("data") / "sky_vertices_precompute_100rs_sub2.npz")
    sky_image = _resolve_sky_image(Path("sky_projections") / "checkerboard_equirect.png")
    ui = InertialViewerPyVista(
        ray_local_interp_npz=ray_local_interp,
        ray_sky_interp_npz=ray_sky_interp,
        sky_vertices_npz=sky_vertices,
        sky_image_path=sky_image,
    )
    ui.show()


if __name__ == "__main__":
    main()
