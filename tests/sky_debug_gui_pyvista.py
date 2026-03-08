from __future__ import annotations

import argparse
from math import pi, sqrt
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    fallbacks = [
        Path("data") / "sky_vertices_precompute_100rs_sub2.npz",
        Path("sky_vertices_precompute_100rs_sub2.npz"),
    ]
    found = next((p for p in fallbacks if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Could not find sky vertex precompute file: {path}")
    return found


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedded PyVista GUI for sky debug with redraw button.")
    p.add_argument("--input", type=Path, default=Path("data") / "sky_vertices_precompute_100rs_sub2.npz")
    p.add_argument("--image", type=Path, default=Path("sky_projections") / "checkerboard_equirect.png")
    p.add_argument("--observer-rs", type=float, default=9.0)
    p.add_argument("--quality", choices=["fast", "medium", "high"], default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sky-opacity", type=float, default=0.50)
    p.add_argument("--show-edges", action="store_true", default=True)
    return p.parse_args()


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
    bh: SchwarzschildBlackHole, r_start: float, r_end: float, impact_b: float, target_phi: float, branch: str, n: int = 360
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
    return np.asarray(r_samples, dtype=float), np.asarray(phi_samples, dtype=float)


def _local_basis_for_pair(a: np.ndarray, b: np.ndarray):
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


def _arrival_dir_from_path(a: np.ndarray, b: np.ndarray, rs: float, path) -> np.ndarray:
    er, ephi, short_sign = _local_basis_for_pair(a, b)
    r_a = float(np.linalg.norm(a))
    r_b = float(np.linalg.norm(b))
    s = float(path.impact_parameter_m) * np.sqrt(max(0.0, 1.0 - rs / r_b)) / r_b
    s = max(-1.0, min(1.0, s))
    gamma = float(np.arcsin(s))
    radial_sign = (+1.0 if r_a > r_b else -1.0) if str(path.branch) == "turning" else (-1.0 if r_a > r_b else +1.0)
    orient_sign = short_sign if int(path.direction) == +1 else -short_sign
    vv = radial_sign * np.cos(gamma) * er + orient_sign * np.sin(gamma) * ephi
    vv = vv / max(float(np.linalg.norm(vv)), 1e-12)
    return vv


def _interp_row(data: np.lib.npyio.NpzFile, r_b: float):
    b_r = np.asarray(data["b_r_m"], dtype=float)
    arr_p = np.asarray(data["arrival_dir_plus_xyz"], dtype=float)
    arr_m = np.asarray(data["arrival_dir_minus_xyz"], dtype=float)
    ok_p = np.asarray(data["ok_plus"], dtype=bool)
    ok_m = np.asarray(data["ok_minus"], dtype=bool)
    if b_r.size == 1:
        return arr_p[0], arr_m[0], ok_p[0], ok_m[0]
    rb = float(np.clip(r_b, float(b_r[0]), float(b_r[-1])))
    i0 = int(np.searchsorted(b_r, rb, side="right") - 1)
    i0 = max(0, min(i0, b_r.size - 2))
    i1 = i0 + 1
    d = float(b_r[i1] - b_r[i0])
    w = 0.0 if d <= 0.0 else (rb - float(b_r[i0])) / d
    ap = (1.0 - w) * arr_p[i0] + w * arr_p[i1]
    am = (1.0 - w) * arr_m[i0] + w * arr_m[i1]
    op = ok_p[i0] & ok_p[i1] & np.all(np.isfinite(ap), axis=1)
    om = ok_m[i0] & ok_m[i1] & np.all(np.isfinite(am), axis=1)
    return ap, am, op, om


def _path_xyz_for_branch(a: np.ndarray, b: np.ndarray, bh: SchwarzschildBlackHole, path):
    n = np.cross(a, b)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    n = n / nn
    e1 = a / max(float(np.linalg.norm(a)), 1e-12)
    e2 = np.cross(n, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-12)
    target = float(path.target_azimuth_rad)
    orient = +1.0 if int(path.direction) == +1 else -1.0
    r_s, phi_s = _build_path_profile(
        bh=bh,
        r_start=float(np.linalg.norm(a)),
        r_end=float(np.linalg.norm(b)),
        impact_b=float(path.impact_parameter_m),
        target_phi=target,
        branch=str(path.branch),
        n=360,
    )
    th = orient * phi_s
    xyz = (r_s[:, None] * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]))
    return xyz


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


class SkyDebugWindow:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from PyQt5 import QtWidgets
            from pyvistaqt import QtInteractor
            import pyvista as pv
        except Exception as exc:
            raise RuntimeError("Install GUI deps: pip install pyvista pyvistaqt PyQt5") from exc

        self.QtWidgets = QtWidgets
        self.QtInteractor = QtInteractor
        self.pv = pv
        self.args = args

        npz_path = _resolve_input(Path(args.input))
        self.data = np.load(npz_path, allow_pickle=False)
        self.rs = float(np.asarray(self.data["rs_m"], dtype=float))
        self.verts = np.asarray(self.data["vertices_m"], dtype=float)
        self.faces = np.asarray(self.data["faces"], dtype=np.int32)
        self.uv = np.asarray(self.data["uv"], dtype=np.float32)
        self.observer = np.asarray([float(args.observer_rs) * self.rs, 0.0, 0.0], dtype=float)
        self.bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * self.rs) / C).with_quality(str(args.quality))
        self.arr_p, self.arr_m, self.ok_p, self.ok_m = _interp_row(self.data, r_b=float(np.linalg.norm(self.observer)))
        self.rng = np.random.default_rng(int(args.seed))
        self.dynamic_names: list[str] = []

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Sky Debug GUI (PyVista)")
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.plotter = self.QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 1)

        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel)
        self.btn_next = QtWidgets.QPushButton("Next Vertex")
        self.btn_next.clicked.connect(self._on_next_vertex)
        panel.addWidget(self.btn_next)
        self.lbl = QtWidgets.QLabel("vertex: -")
        panel.addWidget(self.lbl)
        panel.addStretch(1)

        self._build_static_scene()
        self._draw_random_vertex()

    def _build_static_scene(self) -> None:
        tri_pts, tri_faces, tri_uv = _expand_faces_with_seam_fix(self.verts, self.faces, self.uv)
        face_cells = np.hstack([np.full((tri_faces.shape[0], 1), 3, dtype=np.int32), tri_faces]).ravel()
        mesh = self.pv.PolyData(tri_pts, face_cells.astype(np.int32))
        mesh.active_texture_coordinates = tri_uv
        tex = self.pv.read_texture(str(self.args.image))
        try:
            tex.repeat = True
        except Exception:
            pass

        self.plotter.set_background("black")
        self.plotter.add_mesh(
            mesh,
            texture=tex,
            opacity=float(np.clip(self.args.sky_opacity, 0.0, 1.0)),
            show_edges=bool(self.args.show_edges),
            edge_color="red",
            line_width=3.0,
            smooth_shading=False,
            lighting=False,
            name="sky_mesh",
        )
        self.plotter.add_mesh(
            self.pv.Sphere(radius=float(self.rs), center=(0.0, 0.0, 0.0)),
            style="wireframe",
            color="#808080",
            opacity=0.5,
            name="horizon",
        )
        self.plotter.add_mesh(
            self.pv.Sphere(radius=float(1.5 * self.rs), center=(0.0, 0.0, 0.0)),
            style="wireframe",
            color="#b0c4de",
            opacity=0.4,
            name="photon",
        )
        self.plotter.add_points(self.observer.reshape(1, 3), color="#ff44cc", point_size=16, render_points_as_spheres=True, name="observer")
        self.plotter.show_grid(color="gray")
        self.plotter.camera_position = "iso"

    def _clear_dynamic(self) -> None:
        for nm in self.dynamic_names:
            try:
                self.plotter.remove_actor(nm)
            except Exception:
                pass
        self.dynamic_names.clear()

    def _draw_random_vertex(self) -> None:
        self._clear_dynamic()
        vidx = int(self.rng.integers(0, self.verts.shape[0]))
        a = np.asarray(self.verts[vidx], dtype=float)
        rr = self.bh.find_two_shortest_geodesics(a, self.observer, a_before_b=True, use_gpu=False)
        self.plotter.add_points(a.reshape(1, 3), color="#6fb8ff", point_size=14, render_points_as_spheres=True, name="vtx")
        self.dynamic_names.append("vtx")

        vec_scale = 0.08 * float(np.linalg.norm(a))
        text_lines = [f"vertex={vidx}", f"observer={self.args.observer_rs:.2f} Rs"]
        for direction, color in ((+1, "#4cc9ff"), (-1, "#ffb17a")):
            path = next((pp for pp in rr.paths if int(pp.direction) == int(direction)), None)
            if path is None:
                continue
            xyz = _path_xyz_for_branch(a=a, b=self.observer, bh=self.bh, path=path)
            if xyz is not None:
                spl = self.pv.Spline(xyz.astype(np.float32), n_points=xyz.shape[0])
                nm = f"path_{direction:+d}"
                self.plotter.add_mesh(spl, color=color, line_width=3, name=nm)
                self.dynamic_names.append(nm)

            solve_arr = _arrival_dir_from_path(a=a, b=self.observer, rs=self.rs, path=path)
            nm_s = f"arr_solve_{direction:+d}"
            self.plotter.add_arrows(self.observer.reshape(1, 3), (-solve_arr).reshape(1, 3), mag=vec_scale, color="#b455ff", name=nm_s)
            self.dynamic_names.append(nm_s)

            pre_arr = self.arr_p[vidx] if direction == +1 else self.arr_m[vidx]
            ok = bool(self.ok_p[vidx]) if direction == +1 else bool(self.ok_m[vidx])
            if ok and np.all(np.isfinite(pre_arr)):
                nm_p = f"arr_pre_{direction:+d}"
                self.plotter.add_arrows(
                    self.observer.reshape(1, 3),
                    (-np.asarray(pre_arr)).reshape(1, 3),
                    mag=0.8 * vec_scale,
                    color="#d5a6ff",
                    name=nm_p,
                )
                self.dynamic_names.append(nm_p)
                diff = float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                float(
                                    np.dot(
                                        solve_arr / max(np.linalg.norm(solve_arr), 1e-12),
                                        np.asarray(pre_arr, dtype=float) / max(np.linalg.norm(pre_arr), 1e-12),
                                    )
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                text_lines.append(f"dir {direction:+d} diff={diff:.4f} deg")

        nm_t = "overlay_text"
        self.plotter.add_text("\n".join(text_lines), font_size=10, color="white", name=nm_t)
        self.dynamic_names.append(nm_t)
        self.lbl.setText(f"vertex: {vidx}")
        self.plotter.render()

    def _on_next_vertex(self) -> None:
        self._draw_random_vertex()

    def show(self) -> None:
        self.window.resize(1400, 900)
        self.window.show()
        self.app.exec_()


def main() -> None:
    args = _parse_args()
    ui = SkyDebugWindow(args)
    ui.show()


if __name__ == "__main__":
    main()
