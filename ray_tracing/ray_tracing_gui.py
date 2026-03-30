from __future__ import annotations

import json
import sys
import tkinter as tk
from math import ceil, sqrt
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from matplotlib import cm, colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C
from ray_tracing import NullGeodesicSolverRS


class RayTracingGUI:
    DEFAULTS_PATH = Path(__file__).resolve().parent / "gui_defaults.json"
    MULTI_SEQS_DEFAULT_PATH = Path(__file__).resolve().parent / "multi_sequences_cache.npz"
    INTERP_TABLE_DEFAULT_PATH = Path(__file__).resolve().parent / "multi_sequences_interp_table.npz"
    SKY_INTERP_TABLE_DEFAULT_PATH = Path(__file__).resolve().parent / "sky_interp_table.npz"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Schwarzschild Ray Backtrace")
        self.root.geometry("1200x820")

        self.solver = NullGeodesicSolverRS()

        self._build_controls()
        self._build_plot()
        self._draw_empty()

    def _build_controls(self) -> None:
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(side=tk.TOP, fill=tk.X)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.angle_deg_var = tk.StringVar(value="90")
        self.rs_ls_var = tk.StringVar(value="1")
        self.back_time_s_var = tk.StringVar(value="10")
        self.bx_rs_var = tk.StringVar(value="5")
        self.endpoint_thresh_rs_var = tk.StringVar(value="1")
        self.max_recursion_var = tk.StringVar(value="3")
        self.b_range_min_rs_var = tk.StringVar(value="1.6")
        self.b_range_max_rs_var = tk.StringVar(value="10")
        self.b_range_count_var = tk.StringVar(value="5")
        self.multi_file_path_var = tk.StringVar(value=str(self.MULTI_SEQS_DEFAULT_PATH))
        self.interp_r_min_rs_var = tk.StringVar(value="1.6")
        self.interp_r_max_rs_var = tk.StringVar(value="10")
        self.interp_r_count_var = tk.StringVar(value="5")
        self.interp_theta_min_deg_var = tk.StringVar(value="0")
        self.interp_theta_max_deg_var = tk.StringVar(value="360")
        self.interp_theta_count_var = tk.StringVar(value="36")
        self.heatmap_res_mult_var = tk.StringVar(value="10")
        self.interp_vec_res_mult_var = tk.StringVar(value="1")
        self.interp_file_path_var = tk.StringVar(value=str(self.INTERP_TABLE_DEFAULT_PATH))
        self.sky_interp_file_path_var = tk.StringVar(value=str(self.SKY_INTERP_TABLE_DEFAULT_PATH))
        self.show_ticks_var = tk.BooleanVar(value=True)
        self.show_lookback_vec_var = tk.BooleanVar(value=False)
        self.show_final_light_vec_var = tk.BooleanVar(value=False)
        self.show_tick_light_vec_var = tk.BooleanVar(value=False)

        common = ttk.LabelFrame(frame, text="Common")
        common.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        for col in (1, 3, 5, 7):
            common.columnconfigure(col, weight=1)
        ttk.Label(common, text="Rs (ls)").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(common, textvariable=self.rs_ls_var, width=8).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ttk.Label(common, text="Back Time (s)").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(common, textvariable=self.back_time_s_var, width=8).grid(row=0, column=3, sticky="we", padx=4, pady=2)
        ttk.Label(common, text="B x (Rs)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(common, textvariable=self.bx_rs_var, width=8).grid(row=1, column=1, sticky="we", padx=4, pady=2)
        ttk.Label(common, text="Endpt Seperation Thresh (Rs)").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(common, textvariable=self.endpoint_thresh_rs_var, width=8).grid(row=1, column=3, sticky="we", padx=4, pady=2)
        ttk.Button(common, text="Save Defaults", command=self._save_defaults).grid(row=0, column=4, rowspan=2, sticky="nsw", padx=8, pady=2)

        toggles = ttk.Frame(common)
        toggles.grid(row=2, column=0, columnspan=5, sticky="w", padx=2, pady=(2, 4))
        ttk.Checkbutton(toggles, text="Ticks", variable=self.show_ticks_var).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Checkbutton(toggles, text="Look-Back Vec", variable=self.show_lookback_vec_var).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Checkbutton(toggles, text="Final Light Vec", variable=self.show_final_light_vec_var).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Checkbutton(toggles, text="Tick Light Vec", variable=self.show_tick_light_vec_var).grid(row=0, column=3, sticky="w", padx=4)

        single = ttk.LabelFrame(frame, text="Single Trajectory")
        single.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        ttk.Label(single, text="Look Angle (deg)").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(single, textvariable=self.angle_deg_var, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(single, text="Solve + Plot Single", command=self._solve_and_plot_single).grid(row=0, column=2, sticky="w", padx=8, pady=2)

        multi = ttk.LabelFrame(frame, text="Multiple Trajectories")
        multi.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 4))
        multi.columnconfigure(9, weight=1)
        ttk.Label(multi, text="Max Rec").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(multi, textvariable=self.max_recursion_var, width=7).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(multi, text="B min").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(multi, textvariable=self.b_range_min_rs_var, width=7).grid(row=0, column=3, sticky="w", padx=4, pady=2)
        ttk.Label(multi, text="B max").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        ttk.Entry(multi, textvariable=self.b_range_max_rs_var, width=7).grid(row=0, column=5, sticky="w", padx=4, pady=2)
        ttk.Label(multi, text="B n").grid(row=0, column=6, sticky="w", padx=4, pady=2)
        ttk.Entry(multi, textvariable=self.b_range_count_var, width=7).grid(row=0, column=7, sticky="w", padx=4, pady=2)
        ttk.Button(multi, text="Solve+Plot 1 Seq", command=self._solve_and_plot_multi).grid(row=0, column=8, sticky="w", padx=(8, 4), pady=2)
        ttk.Button(multi, text="Solve Seqs -> File", command=self._solve_sequences_to_file).grid(row=0, column=9, sticky="w", padx=4, pady=2)
        ttk.Button(multi, text="Plot From File", command=self._plot_sequences_from_file).grid(row=0, column=10, sticky="w", padx=4, pady=2)
        ttk.Label(multi, text="Seq File").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(multi, textvariable=self.multi_file_path_var).grid(row=1, column=1, columnspan=10, sticky="we", padx=4, pady=2)

        interp = ttk.LabelFrame(frame, text="Interpolation Tables")
        interp.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 2))
        interp.columnconfigure(1, weight=1)
        interp.columnconfigure(3, weight=1)
        interp.columnconfigure(5, weight=1)
        interp.columnconfigure(7, weight=1)

        ranges = ttk.Frame(interp)
        ranges.grid(row=0, column=0, columnspan=2, sticky="we", padx=2, pady=(2, 4))
        for col in (1, 3, 5, 7, 9, 11, 13, 15):
            ranges.columnconfigure(col, weight=1)
        ttk.Label(ranges, text="r min").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_r_min_rs_var, width=7).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="r max").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_r_max_rs_var, width=7).grid(row=0, column=3, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="r n").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_r_count_var, width=7).grid(row=0, column=5, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="th min").grid(row=0, column=6, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_theta_min_deg_var, width=7).grid(row=0, column=7, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="th max").grid(row=0, column=8, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_theta_max_deg_var, width=7).grid(row=0, column=9, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="th n").grid(row=0, column=10, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_theta_count_var, width=7).grid(row=0, column=11, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="Heat x").grid(row=0, column=12, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.heatmap_res_mult_var, width=7).grid(row=0, column=13, sticky="we", padx=4, pady=2)
        ttk.Label(ranges, text="Vec x").grid(row=0, column=14, sticky="w", padx=4, pady=2)
        ttk.Entry(ranges, textvariable=self.interp_vec_res_mult_var, width=7).grid(row=0, column=15, sticky="we", padx=4, pady=2)

        local_ops = ttk.LabelFrame(interp, text="Local Table")
        local_ops.grid(row=1, column=0, sticky="nsew", padx=(2, 4), pady=2)
        local_ops.columnconfigure(1, weight=1)
        ttk.Label(local_ops, text="File").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(local_ops, textvariable=self.interp_file_path_var).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ttk.Button(local_ops, text="Build Local Table", command=self._build_interp_table_from_saved_sequences).grid(row=1, column=0, sticky="w", padx=4, pady=(2, 4))
        ttk.Button(local_ops, text="Back-Time Heatmap", command=self._plot_back_time_heatmap_from_interp).grid(row=1, column=1, sticky="w", padx=4, pady=(2, 4))
        ttk.Button(local_ops, text="Propagation Vectors", command=self._plot_propagation_vectors_from_interp).grid(row=1, column=2, sticky="w", padx=4, pady=(2, 4))

        sky_ops = ttk.LabelFrame(interp, text="Sky Table")
        sky_ops.grid(row=1, column=1, sticky="nsew", padx=(4, 2), pady=2)
        sky_ops.columnconfigure(1, weight=1)
        ttk.Label(sky_ops, text="File").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(sky_ops, textvariable=self.sky_interp_file_path_var).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ttk.Button(sky_ops, text="Build Sky Table", command=self._build_sky_interp_table_from_saved_sequences).grid(row=1, column=0, sticky="w", padx=4, pady=(2, 4))
        ttk.Button(sky_ops, text="Sky->Look Mapping", command=self._plot_sky_to_look_mapping).grid(row=1, column=1, sticky="w", padx=4, pady=(2, 4))
        ttk.Button(sky_ops, text="Plot Table Pairs", command=self._plot_sky_table_pairs).grid(row=1, column=2, sticky="w", padx=4, pady=(2, 4))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, sticky="we", padx=4, pady=(6, 0))

        self._load_defaults()

    def _build_plot(self) -> None:
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

    def _use_single_axis(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def _draw_empty(self) -> None:
        self._use_single_axis()
        self.ax.clear()
        self.ax.set_title("Back-traced null geodesic (Rs units)")
        self.ax.set_xlabel("x / Rs")
        self.ax.set_ylabel("y / Rs")
        self.ax.set_aspect("equal", "box")
        self.ax.grid(alpha=0.25)
        self.canvas.draw_idle()

    def _solve_and_plot_single(self) -> None:
        parsed = self._parse_common_inputs(require_angle=True)
        if parsed is None:
            return
        (
            angle_deg,
            rs_m,
            back_time_s,
            bx_rs,
            _threshold_rs,
            show_ticks,
            show_lookback_vec,
            show_final_light_vec,
            show_tick_light_vec,
        ) = parsed

        solved = self._solve_trajectory(angle_rad=np.deg2rad(angle_deg), bx_rs=bx_rs, back_time_s=back_time_s, rs_m=rs_m)
        if solved is None:
            return
        xs, ys, ts_s, out = solved

        self._use_single_axis()
        self.ax.clear()
        self.ax.plot(xs, ys, color="tab:blue", lw=1.5, label="Back-traced ray")
        self.ax.scatter([bx_rs], [0.0], s=40, color="tab:red", label="Observer B")
        self.ax.scatter([xs[-1]], [ys[-1]], s=30, color="tab:green", label="Last sample")
        self._plot_reference_geometry(self.ax)

        self.ax.set_title("Back-traced null geodesic (Rs units)")
        self.ax.set_xlabel("x / Rs")
        self.ax.set_ylabel("y / Rs")
        self.ax.set_aspect("equal", "box")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="best")
        if show_ticks:
            self._draw_time_ticks(self.ax, xs=xs, ys=ys, ts_s=ts_s, color="black", lw=0.9)
        if show_lookback_vec:
            ang = np.deg2rad(angle_deg)
            self._draw_unit_vector(self.ax, x=bx_rs, y=0.0, vx=float(np.cos(ang)), vy=float(np.sin(ang)), color="tab:red")
        if show_final_light_vec:
            self._draw_final_light_vector(self.ax, xs=xs, ys=ys, ts_s=ts_s, color="tab:green")
        if show_tick_light_vec:
            self._draw_tick_light_vectors(self.ax, xs=xs, ys=ys, ts_s=ts_s, color="tab:gray")

        back_time_reached_s = -out.samples[-1].t * rs_m / C
        self.status_var.set(
            f"single: status={out.status}, reached_target={out.reached_target}, rounds={out.rounds}, "
            f"samples={len(out.samples)}, reached_back_time_s={back_time_reached_s:.6g}"
        )
        self.canvas.draw_idle()

    def _solve_and_plot_multi(self) -> None:
        parsed = self._parse_common_inputs(require_angle=False)
        if parsed is None:
            return
        (
            _angle_deg_unused,
            rs_m,
            back_time_s,
            bx_rs,
            threshold_rs,
            show_ticks,
            show_lookback_vec,
            show_final_light_vec,
            show_tick_light_vec,
        ) = parsed

        try:
            max_recursion = int(self.max_recursion_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Max recursion must be an integer.")
            return
        if max_recursion < 0:
            messagebox.showerror("Input Error", "Max recursion must be >= 0.")
            return

        try:
            sorted_angles, cache = self._build_multi_sequence(
                bx_rs=bx_rs,
                back_time_s=back_time_s,
                rs_m=rs_m,
                threshold_rs=threshold_rs,
                max_recursion=max_recursion,
            )
        except RuntimeError as exc:
            messagebox.showerror("Solve Error", str(exc))
            return

        self._use_single_axis()
        self._plot_multi_sequence_on_axis(
            self.ax,
            bx_rs=bx_rs,
            sorted_angles=sorted_angles,
            cache=cache,
            show_ticks=show_ticks,
            show_lookback_vec=show_lookback_vec,
            show_final_light_vec=show_final_light_vec,
            show_tick_light_vec=show_tick_light_vec,
            title="Back-traced null geodesics (multi, recursive endpoint refinement)",
            show_legend=True,
        )
        n_paths = len(sorted_angles)
        self.status_var.set(
            f"multi: paths={n_paths}, threshold_rs={threshold_rs:.6g}, max_recursion={max_recursion}, "
            f"angle_range=[0, 180] deg"
        )
        self.canvas.draw_idle()

    def _solve_and_plot_multi_panels(self) -> None:
        # Backward-compatible helper: use file-driven flow.
        self._solve_sequences_to_file()
        self._plot_sequences_from_file()

    def _solve_sequences_to_file(self) -> None:
        parsed = self._parse_common_inputs(require_angle=False)
        if parsed is None:
            return
        (
            _angle_deg_unused,
            rs_m,
            back_time_s,
            _bx_unused,
            threshold_rs,
            show_ticks,
            show_lookback_vec,
            show_final_light_vec,
            show_tick_light_vec,
        ) = parsed

        try:
            max_recursion = int(self.max_recursion_var.get().strip())
            b_min = float(self.b_range_min_rs_var.get().strip())
            b_max = float(self.b_range_max_rs_var.get().strip())
            b_count = int(self.b_range_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Panel controls must be valid numbers.")
            return

        if max_recursion < 0:
            messagebox.showerror("Input Error", "Max recursion must be >= 0.")
            return
        if b_min <= 1.0:
            messagebox.showerror("Input Error", "B range min must be > 1 Rs.")
            return
        if b_max <= b_min:
            messagebox.showerror("Input Error", "B range max must be greater than min.")
            return
        if b_count < 1:
            messagebox.showerror("Input Error", "B count must be >= 1.")
            return
        seq_file = Path(self.multi_file_path_var.get().strip())
        if not str(seq_file):
            messagebox.showerror("Input Error", "Seq file path cannot be empty.")
            return

        b_values = np.linspace(b_min, b_max, b_count, dtype=float)
        n = int(b_values.size)
        path_counts: list[int] = []
        angles_obj: list[np.ndarray] = []
        xs_obj: list[object] = []
        ys_obj: list[object] = []
        ts_obj: list[object] = []

        self.status_var.set(f"solve->file: calculating 0/{n} B positions...")
        self.root.update_idletasks()

        for i, bx_rs in enumerate(b_values):
            try:
                sorted_angles, cache = self._build_multi_sequence(
                    bx_rs=float(bx_rs),
                    back_time_s=back_time_s,
                    rs_m=rs_m,
                    threshold_rs=threshold_rs,
                    max_recursion=max_recursion,
                )
            except RuntimeError as exc:
                messagebox.showerror("Solve Error", str(exc))
                return
            path_counts.append(len(sorted_angles))
            angles_arr = np.asarray(sorted_angles, dtype=float)
            row_xs: list[np.ndarray] = []
            row_ys: list[np.ndarray] = []
            row_ts: list[np.ndarray] = []
            for a in sorted_angles:
                xs, ys, ts_s, _endpoint, _out = cache[a]
                row_xs.append(np.asarray(xs, dtype=float))
                row_ys.append(np.asarray(ys, dtype=float))
                row_ts.append(np.asarray(ts_s, dtype=float))
            angles_obj.append(angles_arr)
            xs_obj.append(np.asarray(row_xs, dtype=object))
            ys_obj.append(np.asarray(row_ys, dtype=object))
            ts_obj.append(np.asarray(row_ts, dtype=object))

            self.status_var.set(f"solve->file: calculating {i + 1}/{n} B positions...")
            self.root.update_idletasks()

        seq_file.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "kind": "ray_tracing_multi_sequences_v1",
            "rs_m": float(rs_m),
            "back_time_s": float(back_time_s),
            "threshold_rs": float(threshold_rs),
            "max_recursion": int(max_recursion),
            "b_min_rs": float(b_min),
            "b_max_rs": float(b_max),
            "b_count": int(b_count),
        }
        np.savez_compressed(
            seq_file,
            metadata_json=np.asarray(json.dumps(metadata), dtype=object),
            b_values_rs=np.asarray(b_values, dtype=float),
            angles_deg=np.asarray(angles_obj, dtype=object),
            xs=np.asarray(xs_obj, dtype=object),
            ys=np.asarray(ys_obj, dtype=object),
            ts_s=np.asarray(ts_obj, dtype=object),
        )
        self.status_var.set(
            f"solve->file complete: {seq_file} | B count={b_count}, "
            f"paths(min/avg/max)={min(path_counts)}/{(sum(path_counts)/len(path_counts)):.2f}/{max(path_counts)}"
        )

    def _plot_sequences_from_file(self) -> None:
        parsed = self._parse_common_inputs(require_angle=False)
        if parsed is None:
            return
        (
            _angle_deg_unused,
            _rs_m_unused,
            _back_time_s_unused,
            _bx_unused,
            _threshold_unused,
            show_ticks,
            show_lookback_vec,
            show_final_light_vec,
            show_tick_light_vec,
        ) = parsed

        seq_file = Path(self.multi_file_path_var.get().strip())
        if not seq_file.exists():
            messagebox.showerror("Load Error", f"Sequence file not found:\n{seq_file}")
            return

        try:
            data = np.load(seq_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            angles_rows = np.asarray(data["angles_deg"], dtype=object)
            xs_rows = np.asarray(data["xs"], dtype=object)
            ys_rows = np.asarray(data["ys"], dtype=object)
            ts_rows = np.asarray(data["ts_s"], dtype=object)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load sequence file:\n{exc}")
            return

        n = int(b_values.size)
        if n == 0:
            messagebox.showerror("Load Error", "Sequence file contains no B positions.")
            return
        ncols = int(ceil(sqrt(n)))
        nrows = int(ceil(n / ncols))
        self.fig.clear()
        axes = self.fig.subplots(nrows, ncols, squeeze=False)

        path_counts: list[int] = []
        for i in range(n):
            ax = axes[i // ncols][i % ncols]
            bx_rs = float(b_values[i])
            sorted_angles = [float(v) for v in np.asarray(angles_rows[i], dtype=float).tolist()]
            cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], object]] = {}
            row_xs = np.asarray(xs_rows[i], dtype=object)
            row_ys = np.asarray(ys_rows[i], dtype=object)
            row_ts = np.asarray(ts_rows[i], dtype=object)
            for j, a in enumerate(sorted_angles):
                xs = np.asarray(row_xs[j], dtype=float)
                ys = np.asarray(row_ys[j], dtype=float)
                ts_s = np.asarray(row_ts[j], dtype=float)
                endpoint = (float(xs[-1]), float(ys[-1]))
                cache[float(a)] = (xs, ys, ts_s, endpoint, None)
            path_counts.append(len(sorted_angles))
            self._plot_multi_sequence_on_axis(
                ax,
                bx_rs=bx_rs,
                sorted_angles=sorted_angles,
                cache=cache,
                show_ticks=show_ticks,
                show_lookback_vec=show_lookback_vec,
                show_final_light_vec=show_final_light_vec,
                show_tick_light_vec=show_tick_light_vec,
                title=f"B={bx_rs:.3g} Rs | paths={len(sorted_angles)}",
                show_legend=False,
            )
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        self.ax = axes[0][0]
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(
            f"plot-from-file: {seq_file} | B count={n}, "
            f"paths(min/avg/max)={min(path_counts)}/{(sum(path_counts)/len(path_counts)):.2f}/{max(path_counts)}"
        )

    def _build_interp_table_from_saved_sequences(self) -> None:
        seq_file = Path(self.multi_file_path_var.get().strip())
        if not seq_file.exists():
            messagebox.showerror("Load Error", f"Sequence file not found:\n{seq_file}")
            return
        out_file = Path(self.interp_file_path_var.get().strip())
        if not str(out_file):
            messagebox.showerror("Input Error", "Interp file path cannot be empty.")
            return

        try:
            r_min = float(self.interp_r_min_rs_var.get().strip())
            r_max = float(self.interp_r_max_rs_var.get().strip())
            r_count = int(self.interp_r_count_var.get().strip())
            th_min = float(self.interp_theta_min_deg_var.get().strip())
            th_max = float(self.interp_theta_max_deg_var.get().strip())
            th_count = int(self.interp_theta_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Interpolation controls must be valid numbers.")
            return
        if r_min <= 1.0:
            messagebox.showerror("Input Error", "r min must be > 1 Rs.")
            return
        if r_max <= r_min:
            messagebox.showerror("Input Error", "r max must be greater than r min.")
            return
        if r_count < 1:
            messagebox.showerror("Input Error", "r count must be >= 1.")
            return
        if th_count < 1:
            messagebox.showerror("Input Error", "theta count must be >= 1.")
            return

        try:
            data = np.load(seq_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            angles_rows = np.asarray(data["angles_deg"], dtype=object)
            xs_rows = np.asarray(data["xs"], dtype=object)
            ys_rows = np.asarray(data["ys"], dtype=object)
            ts_rows = np.asarray(data["ts_s"], dtype=object)
            metadata_raw = data.get("metadata_json", np.asarray("", dtype=object))
            metadata_in = json.loads(str(metadata_raw.item())) if np.size(metadata_raw) > 0 else {}
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load sequence file:\n{exc}")
            return

        n_b = int(b_values.size)
        if n_b < 1:
            messagebox.showerror("Load Error", "Sequence file contains no B positions.")
            return

        r_grid = np.linspace(r_min, r_max, r_count, dtype=float)
        # Keep theta non-periodic so paths that wind around the hole are not blended
        # with non-winding paths at a 0/360 seam.
        theta_periodic = False
        th_grid_deg = np.linspace(th_min, th_max, th_count, endpoint=True, dtype=float)
        th_grid_rad = np.deg2rad(th_grid_deg)

        lookback_grid = np.full((n_b, r_count, th_count, 2), np.nan, dtype=float)
        back_time_grid = np.full((n_b, r_count, th_count), np.nan, dtype=float)
        prop_dir_grid = np.full((n_b, r_count, th_count, 2), np.nan, dtype=float)
        valid_grid = np.zeros((n_b, r_count, th_count), dtype=bool)

        traj_lookback_rows: list[np.ndarray] = []
        traj_prop_rows: list[np.ndarray] = []

        total_interp = n_b * r_count * th_count
        interp_done = 0

        self.status_var.set(f"interp-table: preparing trajectory vectors for {n_b} B positions...")
        self.root.update_idletasks()

        for bi in range(n_b):
            angles = np.asarray(angles_rows[bi], dtype=float)
            row_xs = np.asarray(xs_rows[bi], dtype=object)
            row_ys = np.asarray(ys_rows[bi], dtype=object)
            row_ts = np.asarray(ts_rows[bi], dtype=object)

            n_traj = int(angles.size)
            if n_traj < 1:
                traj_lookback_rows.append(np.zeros((0, 2), dtype=float))
                traj_prop_rows.append(np.asarray([], dtype=object))
                continue

            lb_row = np.zeros((n_traj, 2), dtype=float)
            lb_valid_list: list[np.ndarray] = []
            prop_row_list: list[np.ndarray] = []
            curves: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

            for ti in range(n_traj):
                xs = np.asarray(row_xs[ti], dtype=float)
                ys = np.asarray(row_ys[ti], dtype=float)
                ts = np.asarray(row_ts[ti], dtype=float)
                if xs.size < 2 or ys.size < 2 or ts.size < 2:
                    continue
                ang = float(np.deg2rad(angles[ti]))
                lb_row[ti, 0] = np.cos(ang)
                lb_row[ti, 1] = np.sin(ang)
                prop_xy = self._compute_prop_dirs_along_curve(xs, ys)
                r_curve = np.hypot(xs, ys)
                theta_curve_deg = np.rad2deg(np.unwrap(np.arctan2(ys, xs)))
                prop_row_list.append(prop_xy)
                lb_valid_list.append(np.asarray([lb_row[ti, 0], lb_row[ti, 1]], dtype=float))
                curves.append((xs, ys, ts, prop_xy, r_curve, theta_curve_deg))

            lb_valid = np.asarray(lb_valid_list, dtype=float) if lb_valid_list else np.zeros((0, 2), dtype=float)
            traj_lookback_rows.append(lb_valid)
            traj_prop_rows.append(np.asarray(prop_row_list, dtype=object))

            self.status_var.set(f"interp-table: interpolating B {bi + 1}/{n_b}...")
            self.root.update_idletasks()

            for ri, rr in enumerate(r_grid):
                for thi in range(th_count):
                    q_theta_deg = float(th_grid_deg[thi])
                    ok, lb_xy, bt_s, pd_xy = self._interpolate_from_curves_at_point(curves, lb_valid, float(rr), q_theta_deg)
                    if ok:
                        lookback_grid[bi, ri, thi, :] = lb_xy
                        back_time_grid[bi, ri, thi] = bt_s
                        prop_dir_grid[bi, ri, thi, :] = pd_xy
                        valid_grid[bi, ri, thi] = True
                    interp_done += 1
                    if interp_done % 200 == 0 or interp_done == total_interp:
                        self.status_var.set(
                            f"interp-table: progress {interp_done}/{total_interp} grid points "
                            f"({100.0 * interp_done / max(1, total_interp):.1f}%)"
                        )
                        self.root.update_idletasks()

        meta_out = {
            "kind": "ray_tracing_interp_table_v1",
            "source_sequence_file": str(seq_file),
            "source_metadata": metadata_in,
            "r_min_rs": float(r_min),
            "r_max_rs": float(r_max),
            "r_count": int(r_count),
            "theta_min_deg": float(th_min),
            "theta_max_deg": float(th_max),
            "theta_count": int(th_count),
            "theta_periodic": bool(theta_periodic),
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_file,
            metadata_json=np.asarray(json.dumps(meta_out), dtype=object),
            b_values_rs=np.asarray(b_values, dtype=float),
            r_values_rs=np.asarray(r_grid, dtype=float),
            theta_values_deg=np.asarray(th_grid_deg, dtype=float),
            lookback_unit_xy=np.asarray(lookback_grid, dtype=float),
            back_time_s=np.asarray(back_time_grid, dtype=float),
            propagation_unit_xy=np.asarray(prop_dir_grid, dtype=float),
            valid=np.asarray(valid_grid, dtype=bool),
            traj_lookback_unit_xy=np.asarray(traj_lookback_rows, dtype=object),
            traj_prop_dirs_xy=np.asarray(traj_prop_rows, dtype=object),
        )
        valid_count = int(np.count_nonzero(valid_grid))
        self.status_var.set(
            f"interp-table saved: {out_file} | valid={valid_count}/{valid_grid.size} "
            f"({100.0 * valid_count / max(1, valid_grid.size):.1f}%) | "
            f"theta_range=[{th_min:.3g}, {th_max:.3g}] deg"
        )

    def _build_sky_interp_table_from_saved_sequences(self) -> None:
        seq_file = Path(self.multi_file_path_var.get().strip())
        if not seq_file.exists():
            messagebox.showerror("Load Error", f"Sequence file not found:\n{seq_file}")
            return
        out_file = Path(self.sky_interp_file_path_var.get().strip())
        if not str(out_file):
            messagebox.showerror("Input Error", "Sky interp file path cannot be empty.")
            return

        try:
            th_min = float(self.interp_theta_min_deg_var.get().strip())
            th_max = float(self.interp_theta_max_deg_var.get().strip())
            th_count = int(self.interp_theta_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Interpolation controls must be valid numbers.")
            return
        if th_count < 1:
            messagebox.showerror("Input Error", "theta count must be >= 1.")
            return

        try:
            data = np.load(seq_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            angles_rows = np.asarray(data["angles_deg"], dtype=object)
            xs_rows = np.asarray(data["xs"], dtype=object)
            ys_rows = np.asarray(data["ys"], dtype=object)
            ts_rows = np.asarray(data["ts_s"], dtype=object)
            metadata_raw = data.get("metadata_json", np.asarray("", dtype=object))
            metadata_in = json.loads(str(metadata_raw.item())) if np.size(metadata_raw) > 0 else {}
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load sequence file:\n{exc}")
            return

        n_b = int(b_values.size)
        if n_b < 1:
            messagebox.showerror("Load Error", "Sequence file contains no B positions.")
            return

        theta_periodic = False
        th_grid_deg = np.linspace(th_min, th_max, th_count, endpoint=True, dtype=float)
        n_th = int(th_grid_deg.size)

        sky_unit = np.full((n_b, n_th, 2), np.nan, dtype=float)
        look_unit = np.full((n_b, n_th, 2), np.nan, dtype=float)
        back_time_s = np.full((n_b, n_th), np.nan, dtype=float)
        valid = np.zeros((n_b, n_th), dtype=bool)

        total_interp = n_b * n_th
        interp_done = 0
        total_truncated = 0

        self.status_var.set(f"sky-table: preparing trajectories for {n_b} B positions...")
        self.root.update_idletasks()

        for bi in range(n_b):
            angles = np.asarray(angles_rows[bi], dtype=float)
            row_xs = np.asarray(xs_rows[bi], dtype=object)
            row_ys = np.asarray(ys_rows[bi], dtype=object)
            row_ts = np.asarray(ts_rows[bi], dtype=object)

            n_traj = int(angles.size)
            theta_end_list: list[float] = []
            sky_end_list: list[np.ndarray] = []
            look_end_list: list[np.ndarray] = []
            bt_end_list: list[float] = []

            for ti in range(n_traj):
                xs = np.asarray(row_xs[ti], dtype=float)
                ys = np.asarray(row_ys[ti], dtype=float)
                ts = np.asarray(row_ts[ti], dtype=float)
                if xs.size < 2 or ys.size < 2 or ts.size < 2:
                    continue
                ang = float(np.deg2rad(angles[ti]))
                look_vec = np.asarray([np.cos(ang), np.sin(ang)], dtype=float)
                prop_xy = self._compute_prop_dirs_along_curve(xs, ys)
                sky_vec = -np.asarray(prop_xy[-1, :], dtype=float)
                theta_curve_deg = np.rad2deg(np.unwrap(np.arctan2(ys, xs)))
                theta_end = float(theta_curve_deg[-1])
                nsky = float(np.hypot(sky_vec[0], sky_vec[1]))
                if nsky <= 1e-12:
                    continue
                sky_vec = sky_vec / nsky
                theta_end_list.append(theta_end)
                sky_end_list.append(sky_vec)
                look_end_list.append(look_vec)
                bt_end_list.append(float(-ts[-1]))

            self.status_var.set(f"sky-table: interpolating B {bi + 1}/{n_b}...")
            self.root.update_idletasks()

            theta_end_arr = np.asarray(theta_end_list, dtype=float)
            sky_end_arr = np.asarray(sky_end_list, dtype=float) if sky_end_list else np.zeros((0, 2), dtype=float)
            look_end_arr = np.asarray(look_end_list, dtype=float) if look_end_list else np.zeros((0, 2), dtype=float)
            bt_end_arr = np.asarray(bt_end_list, dtype=float) if bt_end_list else np.zeros((0,), dtype=float)

            for thi, th_deg in enumerate(th_grid_deg):
                ok, lb_xy, bt_s, sky_xy = self._interpolate_from_theta_endpoints(
                    theta_end_deg=theta_end_arr,
                    look_dirs=look_end_arr,
                    sky_dirs=sky_end_arr,
                    back_times=bt_end_arr,
                    query_theta_deg=float(th_deg),
                )
                if ok:
                    look_unit[bi, thi, :] = lb_xy
                    sky_unit[bi, thi, :] = sky_xy
                    back_time_s[bi, thi] = bt_s
                    valid[bi, thi] = True
                interp_done += 1
                if interp_done % 200 == 0 or interp_done == total_interp:
                    self.status_var.set(
                        f"sky-table: progress {interp_done}/{total_interp} "
                        f"({100.0 * interp_done / max(1, total_interp):.1f}%)"
                    )
                    self.root.update_idletasks()

            total_truncated += self._truncate_sky_row_after_wrap(
                sky_row=sky_unit[bi],
                look_row=look_unit[bi],
                bt_row=back_time_s[bi],
                valid_row=valid[bi],
            )

        meta_out = {
            "kind": "ray_tracing_sky_interp_table_v1",
            "source_sequence_file": str(seq_file),
            "source_metadata": metadata_in,
            "r_used_rs": "threshold_endpoint",
            "theta_min_deg": float(th_min),
            "theta_max_deg": float(th_max),
            "theta_count": int(th_count),
            "theta_periodic": bool(theta_periodic),
            "sky_angle_monotonic_until_wrap": True,
            "semantics": "Given (B radius, sky unit vector) infer lookback unit vector at B by interpolation.",
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_file,
            metadata_json=np.asarray(json.dumps(meta_out), dtype=object),
            b_values_rs=np.asarray(b_values, dtype=float),
            theta_values_deg=np.asarray(th_grid_deg, dtype=float),
            r_used_rs=np.asarray(-1.0, dtype=float),
            sky_unit_xy=np.asarray(sky_unit, dtype=float),
            lookback_unit_xy=np.asarray(look_unit, dtype=float),
            back_time_s=np.asarray(back_time_s, dtype=float),
            valid=np.asarray(valid, dtype=bool),
        )
        valid_count = int(np.count_nonzero(valid))
        self.status_var.set(
            f"sky-table saved: {out_file} | valid={valid_count}/{valid.size} "
            f"({100.0 * valid_count / max(1, valid.size):.1f}%) | truncated_after_wrap={total_truncated}"
        )

    @staticmethod
    def _interpolate_from_theta_endpoints(
        *,
        theta_end_deg: np.ndarray,
        look_dirs: np.ndarray,
        sky_dirs: np.ndarray,
        back_times: np.ndarray,
        query_theta_deg: float,
    ) -> tuple[bool, np.ndarray, float, np.ndarray]:
        if theta_end_deg.size == 0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan"), np.asarray([np.nan, np.nan], dtype=float)

        d = np.abs(np.asarray(theta_end_deg, dtype=float) - float(query_theta_deg))
        order = np.argsort(d)
        k = 2 if order.size >= 2 else 1
        sel = order[:k]
        eps = 1e-9
        w = 1.0 / (d[sel] + eps)
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan"), np.asarray([np.nan, np.nan], dtype=float)
        w = w / wsum

        look = np.zeros(2, dtype=float)
        sky = np.zeros(2, dtype=float)
        bt = 0.0
        for ww, ii in zip(w, sel):
            i = int(ii)
            look += float(ww) * np.asarray(look_dirs[i], dtype=float)
            sky += float(ww) * np.asarray(sky_dirs[i], dtype=float)
            bt += float(ww) * float(back_times[i])

        ln = float(np.hypot(look[0], look[1]))
        sn = float(np.hypot(sky[0], sky[1]))
        if ln > 1e-12:
            look /= ln
        if sn > 1e-12:
            sky /= sn
        ok = bool(np.all(np.isfinite(look)) and np.all(np.isfinite(sky)) and np.isfinite(bt))
        return ok, look, bt, sky

    @staticmethod
    def _truncate_sky_row_after_wrap(
        *,
        sky_row: np.ndarray,
        look_row: np.ndarray,
        bt_row: np.ndarray,
        valid_row: np.ndarray,
    ) -> int:
        idx = np.flatnonzero(np.asarray(valid_row, dtype=bool))
        if idx.size < 2:
            return 0
        sky = np.asarray(sky_row[idx], dtype=float)
        ang = np.rad2deg(np.arctan2(sky[:, 1], sky[:, 0]))
        ang = np.mod(ang + 360.0, 360.0)

        cutoff_pos = None
        prev = float(ang[0])
        for j in range(1, ang.size):
            cur = float(ang[j])
            wrapped = (prev >= 300.0 and cur <= 60.0) or ((prev - cur) > 180.0)
            non_mono = cur < (prev - 1e-6)
            if wrapped or non_mono:
                cutoff_pos = j
                break
            prev = cur

        if cutoff_pos is None:
            return 0

        drop_idx = idx[cutoff_pos:]
        valid_row[drop_idx] = False
        sky_row[drop_idx, :] = np.nan
        look_row[drop_idx, :] = np.nan
        bt_row[drop_idx] = np.nan
        return int(drop_idx.size)

    def _plot_back_time_heatmap_from_interp(self) -> None:
        interp_file = Path(self.interp_file_path_var.get().strip())
        if not interp_file.exists():
            messagebox.showerror("Load Error", f"Interpolation file not found:\n{interp_file}")
            return
        try:
            bx_rs = float(self.bx_rs_var.get().strip())
            heatmap_res_mult = int(self.heatmap_res_mult_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Observer B and Heatmap Res x must be valid numbers.")
            return
        if bx_rs <= 1.0:
            messagebox.showerror("Input Error", "Observer B x-position must be > 1 Rs.")
            return
        if heatmap_res_mult < 1:
            messagebox.showerror("Input Error", "Heatmap Res x must be >= 1.")
            return

        try:
            data = np.load(interp_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            r_values = np.asarray(data["r_values_rs"], dtype=float)
            theta_values_deg = np.asarray(data["theta_values_deg"], dtype=float)
            back_time = np.asarray(data["back_time_s"], dtype=float)
            valid = np.asarray(data["valid"], dtype=bool)
            metadata_raw = data.get("metadata_json", np.asarray("", dtype=object))
            metadata_in = json.loads(str(metadata_raw.item())) if np.size(metadata_raw) > 0 else {}
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load interpolation file:\n{exc}")
            return

        if b_values.size < 1:
            messagebox.showerror("Load Error", "Interpolation file has no B samples.")
            return
        bi = int(np.argmin(np.abs(b_values - bx_rs)))
        b_used = float(b_values[bi])

        z = np.asarray(back_time[bi], dtype=float)
        ok = np.asarray(valid[bi], dtype=bool)
        z_plot = np.where(ok, z, np.nan)

        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="polar")
        self.ax = ax

        r_dense = np.linspace(float(r_values[0]), float(r_values[-1]), max(2, int(r_values.size) * heatmap_res_mult), dtype=float)
        theta_periodic = bool(metadata_in.get("theta_periodic", False))
        th_dense_deg = np.linspace(
            float(theta_values_deg[0]),
            float(theta_values_deg[-1]),
            max(2, int(theta_values_deg.size) * heatmap_res_mult),
            endpoint=not theta_periodic,
            dtype=float,
        )
        z_dense = self._resample_masked_grid(
            z=z_plot,
            valid=ok,
            r_old=np.asarray(r_values, dtype=float),
            th_old=np.asarray(theta_values_deg, dtype=float),
            r_new=r_dense,
            th_new=th_dense_deg,
            periodic_theta=theta_periodic,
        )

        theta_rad = np.deg2rad(th_dense_deg)
        mesh = ax.pcolormesh(theta_rad, r_dense, z_dense, shading="auto", cmap="viridis")
        cbar = self.fig.colorbar(mesh, ax=ax)
        cbar.set_label("Back Time (s)")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(120)
        th_lo = float(theta_values_deg[0])
        th_hi = float(theta_values_deg[-1])
        span = th_hi - th_lo
        if span >= 360.0 - 1e-9:
            ax.set_thetamin(0.0)
            ax.set_thetamax(360.0)
        else:
            ax.set_thetamin(float(th_lo % 360.0))
            ax.set_thetamax(float(th_hi % 360.0) if th_hi >= 0 else float((th_hi % 360.0) + 360.0))
        ax.set_xlabel("theta")
        ax.set_ylabel("r")
        ax.set_title(
            f"Back-Time Heatmap | requested B={bx_rs:.3g} Rs, using table B={b_used:.3g} Rs\n"
            f"theta table range=[{th_lo:.3g}, {th_hi:.3g}] deg"
        )
        ax.grid(alpha=0.2)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(
            f"heatmap plotted from {interp_file} | requested B={bx_rs:.6g} Rs, "
            f"using B[{bi}]={b_used:.6g} Rs | theta_range=[{th_lo:.6g}, {th_hi:.6g}] deg | res x{heatmap_res_mult}"
        )

    def _plot_propagation_vectors_from_interp(self) -> None:
        interp_file = Path(self.interp_file_path_var.get().strip())
        if not interp_file.exists():
            messagebox.showerror("Load Error", f"Interpolation file not found:\n{interp_file}")
            return
        try:
            bx_rs = float(self.bx_rs_var.get().strip())
            vec_res_mult = int(self.interp_vec_res_mult_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Observer B x-position and Vector Res x must be valid numbers.")
            return
        if bx_rs <= 1.0:
            messagebox.showerror("Input Error", "Observer B x-position must be > 1 Rs.")
            return
        if vec_res_mult < 1:
            messagebox.showerror("Input Error", "Vector Res x must be >= 1.")
            return

        try:
            data = np.load(interp_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            r_values = np.asarray(data["r_values_rs"], dtype=float)
            theta_values_deg = np.asarray(data["theta_values_deg"], dtype=float)
            prop_dir = np.asarray(data["propagation_unit_xy"], dtype=float)
            valid = np.asarray(data["valid"], dtype=bool)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load interpolation file:\n{exc}")
            return

        if b_values.size < 1:
            messagebox.showerror("Load Error", "Interpolation file has no B samples.")
            return
        bi = int(np.argmin(np.abs(b_values - bx_rs)))
        b_used = float(b_values[bi])

        u = np.asarray(prop_dir[bi, :, :, 0], dtype=float)
        v = np.asarray(prop_dir[bi, :, :, 1], dtype=float)
        ok = np.asarray(valid[bi], dtype=bool) & np.isfinite(u) & np.isfinite(v)

        rr = np.asarray(r_values, dtype=float)
        tt = np.asarray(theta_values_deg, dtype=float)
        if vec_res_mult > 1:
            rr_dense = np.linspace(float(rr[0]), float(rr[-1]), max(2, int(rr.size) * vec_res_mult), dtype=float)
            tt_dense = np.linspace(float(tt[0]), float(tt[-1]), max(2, int(tt.size) * vec_res_mult), dtype=float)
            u = self._resample_masked_grid(
                z=u,
                valid=ok,
                r_old=rr,
                th_old=tt,
                r_new=rr_dense,
                th_new=tt_dense,
                periodic_theta=False,
            )
            v = self._resample_masked_grid(
                z=v,
                valid=ok,
                r_old=rr,
                th_old=tt,
                r_new=rr_dense,
                th_new=tt_dense,
                periodic_theta=False,
            )
            mag = np.hypot(u, v)
            ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(mag) & (mag > 1e-9)
            u = np.where(ok, u / np.where(ok, mag, 1.0), np.nan)
            v = np.where(ok, v / np.where(ok, mag, 1.0), np.nan)
            rr = rr_dense
            tt = tt_dense

        th_rad = np.deg2rad(tt)
        th_mesh, r_mesh = np.meshgrid(th_rad, rr, indexing="xy")
        x = r_mesh * np.cos(th_mesh)
        y = r_mesh * np.sin(th_mesh)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.ax = ax
        self._plot_reference_geometry(ax)
        ax.scatter([bx_rs], [0.0], s=35, color="tab:red", label="Observer B")

        if np.any(ok):
            cvals = np.asarray(np.rad2deg(th_mesh), dtype=float)
            cmax = max(360.0, float(np.nanmax(cvals[ok])) if np.any(ok) else 360.0)
            rb_nowhite = colors.LinearSegmentedColormap.from_list(
                "rb_nowhite",
                ["#d7191c", "#7b3294", "#2c7bb6"],
            )
            q = ax.quiver(
                x[ok],
                y[ok],
                u[ok],
                v[ok],
                cvals[ok],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                cmap=rb_nowhite,
                norm=colors.Normalize(vmin=0.0, vmax=cmax),
                width=0.003,
                alpha=0.9,
            )
            cbar = self.fig.colorbar(q, ax=ax)
            cbar.set_label("theta (deg, unwrapped)")

        ax.set_title(f"Propagation Unit Vectors | requested B={bx_rs:.3g} Rs, using table B={b_used:.3g} Rs")
        ax.set_xlabel("x / Rs")
        ax.set_ylabel("y / Rs")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        valid_count = int(np.count_nonzero(ok))
        total = int(ok.size)
        self.status_var.set(
            f"prop vectors plotted from {interp_file} | requested B={bx_rs:.6g} Rs, "
            f"using B[{bi}]={b_used:.6g} Rs | valid={valid_count}/{total} | res x{vec_res_mult}"
        )

    def _plot_sky_to_look_mapping(self) -> None:
        sky_file = Path(self.sky_interp_file_path_var.get().strip())
        if not sky_file.exists():
            messagebox.showerror("Load Error", f"Sky interpolation file not found:\n{sky_file}")
            return
        try:
            bx_rs = float(self.bx_rs_var.get().strip())
            th_min = float(self.interp_theta_min_deg_var.get().strip())
            th_max = float(self.interp_theta_max_deg_var.get().strip())
            th_count = int(self.interp_theta_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "B and theta controls must be valid numbers.")
            return
        if bx_rs <= 1.0:
            messagebox.showerror("Input Error", "Observer B x-position must be > 1 Rs.")
            return
        if th_count < 2:
            messagebox.showerror("Input Error", "theta count must be >= 2.")
            return

        try:
            data = np.load(sky_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            sky_unit = np.asarray(data["sky_unit_xy"], dtype=float)       # [b, th, 2]
            look_unit = np.asarray(data["lookback_unit_xy"], dtype=float) # [b, th, 2]
            back_time = np.asarray(data["back_time_s"], dtype=float)      # [b, th]
            valid = np.asarray(data["valid"], dtype=bool)                 # [b, th]
            theta_tab = np.asarray(data["theta_values_deg"], dtype=float)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load sky table:\n{exc}")
            return

        if b_values.size < 1:
            messagebox.showerror("Load Error", "Sky table contains no B samples.")
            return
        bi = int(np.argmin(np.abs(b_values - bx_rs)))
        b_used = float(b_values[bi])
        sky_row = np.asarray(sky_unit[bi], dtype=float)
        look_row = np.asarray(look_unit[bi], dtype=float)
        bt_row = np.asarray(back_time[bi], dtype=float)
        ok_row = np.asarray(valid[bi], dtype=bool)

        theta_q = np.linspace(th_min, th_max, th_count, endpoint=True, dtype=float)
        sky_q = np.column_stack([np.cos(np.deg2rad(theta_q)), np.sin(np.deg2rad(theta_q))]).astype(float)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.ax = ax
        ax.axhline(0.0, color="#bbbbbb", lw=0.8, alpha=0.7)
        ax.axvline(0.0, color="#bbbbbb", lw=0.8, alpha=0.7)
        ax.scatter([0.0], [0.0], s=18, color="black")

        ok_count = 0
        cmap_vals = np.linspace(0.05, 0.95, theta_q.size)
        for i in range(theta_q.size):
            qsky = np.asarray(sky_q[i], dtype=float)
            ok, qlook, _bt = self._lookup_look_from_sky_vectors(
                sky_row=sky_row,
                look_row=look_row,
                bt_row=bt_row,
                valid_row=ok_row,
                qsky=qsky,
            )
            if not ok:
                continue
            ok_count += 1
            col = cm.plasma(cmap_vals[i])
            # Look vector from origin.
            ax.arrow(
                0.0,
                0.0,
                float(qlook[0]),
                float(qlook[1]),
                color=col,
                width=0.006,
                head_width=0.07,
                head_length=0.09,
                length_includes_head=True,
                alpha=0.95,
                zorder=5,
            )
            # Sky vector from head of look vector.
            ax.arrow(
                float(qlook[0]),
                float(qlook[1]),
                float(qsky[0]),
                float(qsky[1]),
                color=col,
                width=0.006,
                head_width=0.07,
                head_length=0.09,
                length_includes_head=True,
                alpha=0.95,
                zorder=5,
            )

        circ_t = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(np.cos(circ_t), np.sin(circ_t), "k--", lw=0.8, alpha=0.7, label="unit circle")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Sky->Look Mapping | requested B={bx_rs:.3g} Rs, using table B={b_used:.3g} Rs\n"
            f"sampled theta=[{th_min:.3g}, {th_max:.3g}] deg, count={th_count}"
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.status_var.set(
            f"sky->look mapping plotted from {sky_file} | requested B={bx_rs:.6g} Rs, "
            f"using B[{bi}]={b_used:.6g} Rs | valid={ok_count}/{theta_q.size} | "
            f"table_theta=[{theta_tab[0]:.6g}, {theta_tab[-1]:.6g}]"
        )

    def _plot_sky_table_pairs(self) -> None:
        sky_file = Path(self.sky_interp_file_path_var.get().strip())
        if not sky_file.exists():
            messagebox.showerror("Load Error", f"Sky interpolation file not found:\n{sky_file}")
            return
        try:
            bx_rs = float(self.bx_rs_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Observer B x-position must be a valid number.")
            return
        if bx_rs <= 1.0:
            messagebox.showerror("Input Error", "Observer B x-position must be > 1 Rs.")
            return

        try:
            data = np.load(sky_file, allow_pickle=True)
            b_values = np.asarray(data["b_values_rs"], dtype=float)
            sky_unit = np.asarray(data["sky_unit_xy"], dtype=float)
            look_unit = np.asarray(data["lookback_unit_xy"], dtype=float)
            valid = np.asarray(data["valid"], dtype=bool)
            theta_tab = np.asarray(data["theta_values_deg"], dtype=float)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load sky table:\n{exc}")
            return

        if b_values.size < 1:
            messagebox.showerror("Load Error", "Sky table contains no B samples.")
            return
        lo_candidates = np.where(b_values <= bx_rs)[0]
        hi_candidates = np.where(b_values >= bx_rs)[0]
        lo_idx = int(lo_candidates[-1]) if lo_candidates.size > 0 else 0
        hi_idx = int(hi_candidates[0]) if hi_candidates.size > 0 else int(b_values.size - 1)
        idx_list = [lo_idx] if lo_idx == hi_idx else [lo_idx, hi_idx]

        self.fig.clear()
        axes = self.fig.subplots(len(idx_list), 1, squeeze=False).ravel()
        self.ax = axes[0]

        # Visual spacing: within-pair tail separation is half of between-pair separation.
        pair_sep_x = 1.2
        between_pair_x = 2.4
        pair_stride = pair_sep_x + between_pair_x
        pair_order_label = "(look, sky)"
        th_lo = float(theta_tab[0]) if theta_tab.size else float("nan")
        th_hi = float(theta_tab[-1]) if theta_tab.size else float("nan")
        plotted_counts: list[int] = []
        plotted_b: list[float] = []

        for ax, bi in zip(axes, idx_list):
            b_used = float(b_values[bi])
            sky_row = np.asarray(sky_unit[bi], dtype=float)
            look_row = np.asarray(look_unit[bi], dtype=float)
            ok_row = np.asarray(valid[bi], dtype=bool)
            valid_idx = np.flatnonzero(ok_row)
            plotted_counts.append(int(valid_idx.size))
            plotted_b.append(b_used)

            ax.axhline(0.0, color="#bbbbbb", lw=0.8, alpha=0.8)
            if valid_idx.size > 0:
                cmap_vals = np.linspace(0.05, 0.95, valid_idx.size)
                for pi, idx in enumerate(valid_idx):
                    x0 = float(pi) * pair_stride
                    look = np.asarray(look_row[int(idx)], dtype=float)
                    sky = np.asarray(sky_row[int(idx)], dtype=float)
                    col = cm.plasma(cmap_vals[pi])

                    ax.arrow(
                        x0,
                        0.0,
                        float(look[0]),
                        float(look[1]),
                        color=col,
                        width=0.012,
                        head_width=0.12,
                        head_length=0.14,
                        length_includes_head=True,
                        alpha=0.95,
                        zorder=5,
                    )
                    ax.arrow(
                        x0 + pair_sep_x,
                        0.0,
                        float(sky[0]),
                        float(sky[1]),
                        color=col,
                        width=0.012,
                        head_width=0.12,
                        head_length=0.14,
                        length_includes_head=True,
                        alpha=0.95,
                        zorder=5,
                    )
                    ax.plot([x0, x0 + pair_sep_x], [0.0, 0.0], color="#999999", lw=0.5, alpha=0.6)

                # Black separators between neighboring pairs.
                for pi in range(valid_idx.size - 1):
                    x_pair0 = float(pi) * pair_stride
                    x_pair1 = float(pi + 1) * pair_stride
                    x_sep = 0.5 * ((x_pair0 + pair_sep_x) + x_pair1)
                    ax.axvline(x_sep, color="black", lw=0.9, alpha=0.9, zorder=1)

                x_max = float((valid_idx.size - 1) * pair_stride + pair_sep_x + 1.3)
                ax.set_xlim(-0.7, x_max)
            else:
                ax.set_xlim(-0.7, 2.3)

            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal", "box")
            ax.set_ylabel("vector y")
            ax.set_title(
                f"B row: {b_used:.3g} Rs ({'<= query' if b_used <= bx_rs else '>= query'}) {pair_order_label} | "
                f"valid pairs={valid_idx.size}"
            )
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("table order (pair tails separated by +1 x)")
        self.fig.suptitle(
            f"Sky Table Pairs {pair_order_label} | requested B={bx_rs:.3g} Rs | theta=[{th_lo:.3g}, {th_hi:.3g}] deg",
            y=0.995,
        )
        self.fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        self.canvas.draw_idle()

        if len(idx_list) == 1:
            self.status_var.set(
                f"sky table pairs plotted from {sky_file} | requested B={bx_rs:.6g} Rs | "
                f"using exact/only B[{idx_list[0]}]={plotted_b[0]:.6g} Rs | valid pairs={plotted_counts[0]}"
            )
        else:
            self.status_var.set(
                f"sky table pairs plotted from {sky_file} | requested B={bx_rs:.6g} Rs | "
                f"B<= uses [{idx_list[0]}]={plotted_b[0]:.6g} (pairs={plotted_counts[0]}), "
                f"B>= uses [{idx_list[1]}]={plotted_b[1]:.6g} (pairs={plotted_counts[1]})"
            )

    @staticmethod
    def _lookup_look_from_sky_vectors(
        *,
        sky_row: np.ndarray,
        look_row: np.ndarray,
        bt_row: np.ndarray,
        valid_row: np.ndarray,
        qsky: np.ndarray,
    ) -> tuple[bool, np.ndarray, float]:
        ok = np.asarray(valid_row, dtype=bool)
        if np.count_nonzero(ok) == 0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")

        s = np.asarray(sky_row[ok], dtype=float)
        l = np.asarray(look_row[ok], dtype=float)
        t = np.asarray(bt_row[ok], dtype=float)
        q = np.asarray(qsky, dtype=float)
        nq = float(np.hypot(q[0], q[1]))
        if nq <= 1e-12:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")
        q = q / nq

        # Enforce interpolation domain: query sky direction must lie within
        # the valid angular span in this row (no extrapolation outside table range).
        s_ang = np.mod(np.rad2deg(np.arctan2(s[:, 1], s[:, 0])) + 360.0, 360.0)
        q_ang = float(np.mod(np.rad2deg(np.arctan2(q[1], q[0])) + 360.0, 360.0))
        sort_idx = np.argsort(s_ang)
        s_ang = s_ang[sort_idx]
        s = s[sort_idx]
        l = l[sort_idx]
        t = t[sort_idx]
        eps_deg = 1e-6
        if q_ang < float(s_ang[0]) - eps_deg or q_ang > float(s_ang[-1]) + eps_deg:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")

        d = np.abs(s_ang - q_ang)
        order = np.argsort(d)
        k = 2 if order.size >= 2 else 1
        sel = order[:k]
        eps = 1e-9
        w = 1.0 / (d[sel] + eps)
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan")
        w /= wsum

        look = np.zeros(2, dtype=float)
        bt = 0.0
        for ww, ii in zip(w, sel):
            i = int(ii)
            look += float(ww) * l[i]
            bt += float(ww) * float(t[i])
        nlook = float(np.hypot(look[0], look[1]))
        if nlook > 1e-12:
            look /= nlook
        ok_out = bool(np.all(np.isfinite(look)) and np.isfinite(bt))
        return ok_out, look, bt

    @staticmethod
    def _resample_masked_grid(
        *,
        z: np.ndarray,
        valid: np.ndarray,
        r_old: np.ndarray,
        th_old: np.ndarray,
        r_new: np.ndarray,
        th_new: np.ndarray,
        periodic_theta: bool = False,
    ) -> np.ndarray:
        z0 = np.where(valid, z, 0.0).astype(float)
        w0 = valid.astype(float)

        def interp_axis_theta(arr2d: np.ndarray) -> np.ndarray:
            out = np.empty((arr2d.shape[0], th_new.size), dtype=float)
            for i in range(arr2d.shape[0]):
                if periodic_theta:
                    out[i, :] = RayTracingGUI._interp_periodic_1d(th_old, arr2d[i, :], th_new, period=360.0)
                else:
                    out[i, :] = np.interp(th_new, th_old, arr2d[i, :])
            return out

        z_th = interp_axis_theta(z0)
        w_th = interp_axis_theta(w0)

        z_out = np.empty((r_new.size, th_new.size), dtype=float)
        w_out = np.empty((r_new.size, th_new.size), dtype=float)
        for j in range(th_new.size):
            z_out[:, j] = np.interp(r_new, r_old, z_th[:, j])
            w_out[:, j] = np.interp(r_new, r_old, w_th[:, j])

        with np.errstate(invalid="ignore", divide="ignore"):
            out = z_out / w_out
        out[w_out < 1e-6] = np.nan
        return out

    @staticmethod
    def _is_full_circle_theta_range(theta_min_deg: float, theta_max_deg: float) -> bool:
        span = float(theta_max_deg - theta_min_deg)
        if span <= 0.0:
            return False
        k = span / 360.0
        return abs(k - round(k)) < 1e-9

    @staticmethod
    def _interp_periodic_1d(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray, period: float = 360.0) -> np.ndarray:
        if x_old.size < 2:
            return np.full(x_new.shape, float(y_old[0]) if y_old.size else np.nan, dtype=float)

        x0 = float(x_old[0])
        xo = np.asarray(x_old, dtype=float)
        yo = np.asarray(y_old, dtype=float)

        # Drop duplicated endpoint if present (e.g., 0 and 360 equivalent sample).
        if abs((xo[-1] - xo[0]) - period) < 1e-9:
            xo = xo[:-1]
            yo = yo[:-1]

        xo_ext = np.concatenate([xo, [xo[0] + period]])
        yo_ext = np.concatenate([yo, [yo[0]]])

        xn = ((np.asarray(x_new, dtype=float) - x0) % period) + x0
        return np.interp(xn, xo_ext, yo_ext)

    @staticmethod
    def _compute_prop_dirs_along_curve(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        n = int(xs.size)
        out = np.zeros((n, 2), dtype=float)
        if n < 2:
            return out
        for i in range(n):
            if i == 0:
                tx = float(xs[1] - xs[0])
                ty = float(ys[1] - ys[0])
            elif i == n - 1:
                tx = float(xs[n - 1] - xs[n - 2])
                ty = float(ys[n - 1] - ys[n - 2])
            else:
                tx = float(xs[i + 1] - xs[i - 1])
                ty = float(ys[i + 1] - ys[i - 1])
            # Reverse tangent because stored trajectory is back-trace direction.
            vx = -tx
            vy = -ty
            nn = float(np.hypot(vx, vy))
            if nn > 1e-12:
                out[i, 0] = vx / nn
                out[i, 1] = vy / nn
        return out

    @staticmethod
    def _interpolate_from_curves_at_point(
        curves: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        lookbacks: np.ndarray,
        q_r: float,
        q_theta_deg: float,
    ) -> tuple[bool, np.ndarray, float, np.ndarray]:
        if len(curves) == 0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan"), np.asarray([np.nan, np.nan], dtype=float)

        d_list: list[float] = []
        idx_list: list[int] = []
        for (_xs, _ys, _ts, _prop, r_curve, theta_curve_deg) in curves:
            dr = r_curve - q_r
            dtheta_rad = np.deg2rad(theta_curve_deg - q_theta_deg)
            d_arc = q_r * dtheta_rad
            d2 = dr * dr + d_arc * d_arc
            ii = int(np.argmin(d2))
            d_list.append(float(np.sqrt(float(d2[ii]))))
            idx_list.append(ii)

        order = np.argsort(np.asarray(d_list, dtype=float))
        k = 2 if len(order) >= 2 else 1
        sel = order[:k]

        eps = 1e-9
        w = np.asarray([1.0 / (d_list[int(i)] + eps) for i in sel], dtype=float)
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return False, np.asarray([np.nan, np.nan], dtype=float), float("nan"), np.asarray([np.nan, np.nan], dtype=float)
        w /= wsum

        lb = np.zeros(2, dtype=float)
        pd = np.zeros(2, dtype=float)
        bt = 0.0
        for ww, oi in zip(w, sel):
            i = int(oi)
            _xs, _ys, ts, prop, _rr, _tt = curves[i]
            ii = int(idx_list[i])
            lb += float(ww) * np.asarray(lookbacks[i, :], dtype=float)
            pd += float(ww) * np.asarray(prop[ii, :], dtype=float)
            bt += float(ww) * float(-ts[ii])

        lb_n = float(np.hypot(lb[0], lb[1]))
        pd_n = float(np.hypot(pd[0], pd[1]))
        if lb_n > 1e-12:
            lb /= lb_n
        if pd_n > 1e-12:
            pd /= pd_n
        ok = bool(np.isfinite(bt) and np.all(np.isfinite(lb)) and np.all(np.isfinite(pd)))
        return ok, lb, bt, pd

    def _build_multi_sequence(
        self,
        *,
        bx_rs: float,
        back_time_s: float,
        rs_m: float,
        threshold_rs: float,
        max_recursion: int,
    ) -> tuple[list[float], dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], object]]]:
        cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], object]] = {}

        def get_solution(angle_deg: float):
            key = float(angle_deg)
            if key in cache:
                return cache[key]
            solved = self._solve_trajectory(
                angle_rad=np.deg2rad(key),
                bx_rs=bx_rs,
                back_time_s=back_time_s,
                rs_m=rs_m,
            )
            if solved is None:
                raise RuntimeError(f"Failed to solve trajectory at {key:.6g} deg, B={bx_rs:.6g} Rs")
            xs, ys, ts_s, out = solved
            endpoint = (float(xs[-1]), float(ys[-1]))
            cache[key] = (xs, ys, ts_s, endpoint, out)
            return cache[key]

        intervals: list[tuple[float, float, int]] = [(0.0, 180.0, 0)]
        angles_kept: set[float] = {0.0, 180.0}
        get_solution(0.0)
        get_solution(180.0)

        while intervals:
            a0, a1, level = intervals.pop(0)
            _, _, _, p0, _ = get_solution(a0)
            _, _, _, p1, _ = get_solution(a1)
            dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            if dist > threshold_rs and level < max_recursion:
                amid = 0.5 * (a0 + a1)
                angles_kept.add(amid)
                get_solution(amid)
                intervals.append((a0, amid, level + 1))
                intervals.append((amid, a1, level + 1))

        return sorted(angles_kept), cache

    def _plot_multi_sequence_on_axis(
        self,
        ax,
        *,
        bx_rs: float,
        sorted_angles: list[float],
        cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], object]],
        show_ticks: bool,
        show_lookback_vec: bool,
        show_final_light_vec: bool,
        show_tick_light_vec: bool,
        title: str,
        show_legend: bool,
    ) -> None:
        ax.clear()
        self._plot_reference_geometry(ax)
        ax.scatter([bx_rs], [0.0], s=30, color="tab:red", label="Observer B")
        colors = np.linspace(0.05, 0.95, len(sorted_angles))
        for i, a in enumerate(sorted_angles):
            xs, ys, ts_s, _endpoint, _out = cache[a]
            col = cm.viridis(colors[i])
            ax.plot(xs, ys, color=col, lw=1.2, alpha=0.95)
            ax.scatter([xs[-1]], [ys[-1]], s=10, color=col)
            if show_ticks:
                self._draw_time_ticks(ax, xs=xs, ys=ys, ts_s=ts_s, color=col, lw=1.0)
            if show_lookback_vec:
                ang = np.deg2rad(float(a))
                self._draw_unit_vector(ax, x=bx_rs, y=0.0, vx=float(np.cos(ang)), vy=float(np.sin(ang)), color=col)
            if show_final_light_vec:
                self._draw_final_light_vector(ax, xs=xs, ys=ys, ts_s=ts_s, color=col)
            if show_tick_light_vec:
                self._draw_tick_light_vectors(ax, xs=xs, ys=ys, ts_s=ts_s, color=col)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / Rs")
        ax.set_ylabel("y / Rs")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.25)
        if show_legend:
            ax.legend(loc="best")

    def _parse_common_inputs(
        self, require_angle: bool
    ) -> tuple[float, float, float, float, float, bool, bool, bool, bool] | None:
        try:
            angle_deg = float(self.angle_deg_var.get().strip()) if require_angle else 0.0
            rs_ls = float(self.rs_ls_var.get().strip())
            back_time_s = float(self.back_time_s_var.get().strip())
            bx_rs = float(self.bx_rs_var.get().strip())
            threshold_rs = float(self.endpoint_thresh_rs_var.get().strip())
            show_ticks = bool(self.show_ticks_var.get())
            show_lookback_vec = bool(self.show_lookback_vec_var.get())
            show_final_light_vec = bool(self.show_final_light_vec_var.get())
            show_tick_light_vec = bool(self.show_tick_light_vec_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Shared controls must be valid numbers.")
            return None
        if rs_ls <= 0.0:
            messagebox.showerror("Input Error", "Rs must be > 0 light-seconds.")
            return None
        if back_time_s <= 0.0:
            messagebox.showerror("Input Error", "Back time threshold must be > 0 seconds.")
            return None
        if bx_rs <= 1.0:
            messagebox.showerror("Input Error", "Observer B x-position must be > 1 Rs.")
            return None
        if threshold_rs <= 0.0:
            messagebox.showerror("Input Error", "Endpoint threshold must be > 0 Rs.")
            return None
        return (
            angle_deg,
            rs_ls * C,
            back_time_s,
            bx_rs,
            threshold_rs,
            show_ticks,
            show_lookback_vec,
            show_final_light_vec,
            show_tick_light_vec,
        )

    def _solve_trajectory(
        self,
        *,
        angle_rad: float,
        bx_rs: float,
        back_time_s: float,
        rs_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, object] | None:
        try:
            state0 = self.solver.initial_state_from_observer(
                observer_b_xyz_rs=(bx_rs, 0.0, 0.0),
                incoming_angle_at_b_rad=float(angle_rad),
                t_b_rsc=0.0,
            )
            out = self.solver.trace_until_back_time(
                state0,
                back_time_s=back_time_s,
                rs_m=rs_m,
                initial_lambda_span=25.0,
                max_spatial_step_rs=0.1,
            )
        except Exception as exc:
            messagebox.showerror("Solve Error", str(exc))
            return None
        samples = out.samples
        if len(samples) < 2:
            messagebox.showerror("Solve Error", "Not enough samples were produced.")
            return None
        xs = np.asarray([s.r * np.cos(s.phi) for s in samples], dtype=float)
        ys = np.asarray([s.r * np.sin(s.phi) for s in samples], dtype=float)
        ts_s = np.asarray([s.t * rs_m / C for s in samples], dtype=float)
        return xs, ys, ts_s, out

    def _plot_reference_geometry(self, ax) -> None:
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(np.cos(th), np.sin(th), "k-", lw=1.0, label="Horizon (r=1 Rs)")
        ax.plot(1.5 * np.cos(th), 1.5 * np.sin(th), "k--", lw=0.9, label="Photon sphere (r=1.5 Rs)")

    def _draw_time_ticks(self, ax, xs: np.ndarray, ys: np.ndarray, ts_s: np.ndarray, color: str = "black", lw: float = 0.9) -> None:
        if xs.size < 3:
            return
        t0 = float(ts_s[0])
        t1 = float(ts_s[-1])
        if not np.isfinite(t0) or not np.isfinite(t1):
            return
        if t1 >= t0:
            return

        elapsed = t0 - ts_s
        if np.any(~np.isfinite(elapsed)):
            return
        order = np.argsort(elapsed)
        elapsed_s = elapsed[order]
        xs_s = xs[order]
        ys_s = ys[order]

        total_back_s = float(elapsed_s[-1])
        max_whole = int(np.floor(total_back_s + 1e-12))
        if max_whole < 1:
            return
        tick_elapsed = np.arange(1, max_whole + 1, dtype=float)

        path_span = max(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)), 1e-6)
        tick_len = 0.02 * path_span

        for e in tick_elapsed:
            x = float(np.interp(e, elapsed_s, xs_s))
            y = float(np.interp(e, elapsed_s, ys_s))

            d = 0.05
            e0 = max(float(elapsed_s[0]), e - d)
            e1 = min(float(elapsed_s[-1]), e + d)
            if e1 <= e0:
                continue
            x0i = float(np.interp(e0, elapsed_s, xs_s))
            y0i = float(np.interp(e0, elapsed_s, ys_s))
            x1i = float(np.interp(e1, elapsed_s, xs_s))
            y1i = float(np.interp(e1, elapsed_s, ys_s))

            tx = x1i - x0i
            ty = y1i - y0i
            nrm = float(np.hypot(tx, ty))
            if nrm <= 1e-12:
                continue

            nx = -ty / nrm
            ny = tx / nrm

            x0 = x - 0.5 * tick_len * nx
            y0 = y - 0.5 * tick_len * ny
            x1 = x + 0.5 * tick_len * nx
            y1 = y + 0.5 * tick_len * ny
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.95, zorder=5)

    def _draw_unit_vector(self, ax, *, x: float, y: float, vx: float, vy: float, color: str, vector_len_rs: float = 5.0) -> None:
        n = float(np.hypot(vx, vy))
        if n <= 1e-12:
            return
        ux, uy = vx / n, vy / n
        dx = float(vector_len_rs) * ux
        dy = float(vector_len_rs) * uy
        ax.arrow(
            x,
            y,
            dx,
            dy,
            color=color,
            width=0.03,
            head_width=0.35,
            head_length=0.50,
            length_includes_head=True,
            alpha=0.95,
            zorder=6,
        )

    def _curve_param_arrays(self, xs: np.ndarray, ys: np.ndarray, ts_s: np.ndarray):
        t0 = float(ts_s[0])
        elapsed = t0 - ts_s
        order = np.argsort(elapsed)
        return elapsed[order], xs[order], ys[order]

    def _interp_xy_tangent(self, elapsed_s: np.ndarray, xs_s: np.ndarray, ys_s: np.ndarray, e: float, d: float = 0.05):
        x = float(np.interp(e, elapsed_s, xs_s))
        y = float(np.interp(e, elapsed_s, ys_s))
        e0 = max(float(elapsed_s[0]), e - d)
        e1 = min(float(elapsed_s[-1]), e + d)
        if e1 <= e0:
            return x, y, 0.0, 0.0
        x0 = float(np.interp(e0, elapsed_s, xs_s))
        y0 = float(np.interp(e0, elapsed_s, ys_s))
        x1 = float(np.interp(e1, elapsed_s, xs_s))
        y1 = float(np.interp(e1, elapsed_s, ys_s))
        return x, y, (x1 - x0), (y1 - y0)

    def _draw_final_light_vector(self, ax, *, xs: np.ndarray, ys: np.ndarray, ts_s: np.ndarray, color: str) -> None:
        if xs.size < 3:
            return
        elapsed_s, xs_s, ys_s = self._curve_param_arrays(xs, ys, ts_s)
        e_final = float(elapsed_s[-1])
        x, y, tx, ty = self._interp_xy_tangent(elapsed_s, xs_s, ys_s, e_final)
        # Reverse tangent: plotted direction is back-tracing; physical light direction is toward B.
        self._draw_unit_vector(ax, x=x, y=y, vx=-tx, vy=-ty, color=color)

    def _draw_tick_light_vectors(self, ax, *, xs: np.ndarray, ys: np.ndarray, ts_s: np.ndarray, color: str) -> None:
        if xs.size < 3:
            return
        elapsed_s, xs_s, ys_s = self._curve_param_arrays(xs, ys, ts_s)
        total_back_s = float(elapsed_s[-1])
        max_whole = int(np.floor(total_back_s + 1e-12))
        if max_whole < 2:
            return
        tick_elapsed = np.arange(1, max_whole + 1, dtype=float)
        for e in tick_elapsed:
            # Exclude endpoints.
            if e <= 0.0 or e >= total_back_s:
                continue
            x, y, tx, ty = self._interp_xy_tangent(elapsed_s, xs_s, ys_s, float(e))
            self._draw_unit_vector(ax, x=x, y=y, vx=-tx, vy=-ty, color=color)

    def _save_defaults(self) -> None:
        payload = {
            "look_angle_deg": self.angle_deg_var.get().strip(),
            "rs_light_seconds": self.rs_ls_var.get().strip(),
            "back_time_s": self.back_time_s_var.get().strip(),
            "observer_bx_rs": self.bx_rs_var.get().strip(),
            "endpoint_threshold_rs": self.endpoint_thresh_rs_var.get().strip(),
            "max_recursion": self.max_recursion_var.get().strip(),
            "b_range_min_rs": self.b_range_min_rs_var.get().strip(),
            "b_range_max_rs": self.b_range_max_rs_var.get().strip(),
            "b_range_count": self.b_range_count_var.get().strip(),
            "multi_file_path": self.multi_file_path_var.get().strip(),
            "interp_r_min_rs": self.interp_r_min_rs_var.get().strip(),
            "interp_r_max_rs": self.interp_r_max_rs_var.get().strip(),
            "interp_r_count": self.interp_r_count_var.get().strip(),
            "interp_theta_min_deg": self.interp_theta_min_deg_var.get().strip(),
            "interp_theta_max_deg": self.interp_theta_max_deg_var.get().strip(),
            "interp_theta_count": self.interp_theta_count_var.get().strip(),
            "heatmap_res_mult": self.heatmap_res_mult_var.get().strip(),
            "interp_vec_res_mult": self.interp_vec_res_mult_var.get().strip(),
            "interp_file_path": self.interp_file_path_var.get().strip(),
            "show_ticks": bool(self.show_ticks_var.get()),
            "show_lookback_vec": bool(self.show_lookback_vec_var.get()),
            "show_final_light_vec": bool(self.show_final_light_vec_var.get()),
            "show_tick_light_vec": bool(self.show_tick_light_vec_var.get()),
        }
        try:
            self.DEFAULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save Defaults Error", f"Failed to save defaults:\n{exc}")
            return
        self.status_var.set(f"Defaults saved: {self.DEFAULTS_PATH}")

    def _load_defaults(self) -> None:
        path = self.DEFAULTS_PATH
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        mapping = [
            ("look_angle_deg", self.angle_deg_var),
            ("rs_light_seconds", self.rs_ls_var),
            ("back_time_s", self.back_time_s_var),
            ("observer_bx_rs", self.bx_rs_var),
            ("endpoint_threshold_rs", self.endpoint_thresh_rs_var),
            ("max_recursion", self.max_recursion_var),
            ("b_range_min_rs", self.b_range_min_rs_var),
            ("b_range_max_rs", self.b_range_max_rs_var),
            ("b_range_count", self.b_range_count_var),
            ("multi_file_path", self.multi_file_path_var),
            ("interp_r_min_rs", self.interp_r_min_rs_var),
            ("interp_r_max_rs", self.interp_r_max_rs_var),
            ("interp_r_count", self.interp_r_count_var),
            ("interp_theta_min_deg", self.interp_theta_min_deg_var),
            ("interp_theta_max_deg", self.interp_theta_max_deg_var),
            ("interp_theta_count", self.interp_theta_count_var),
            ("heatmap_res_mult", self.heatmap_res_mult_var),
            ("interp_vec_res_mult", self.interp_vec_res_mult_var),
            ("interp_file_path", self.interp_file_path_var),
        ]
        for key, var in mapping:
            if key in payload and payload[key] is not None:
                var.set(str(payload[key]))
        if "show_ticks" in payload:
            self.show_ticks_var.set(bool(payload["show_ticks"]))
        if "show_lookback_vec" in payload:
            self.show_lookback_vec_var.set(bool(payload["show_lookback_vec"]))
        if "show_final_light_vec" in payload:
            self.show_final_light_vec_var.set(bool(payload["show_final_light_vec"]))
        if "show_tick_light_vec" in payload:
            self.show_tick_light_vec_var.set(bool(payload["show_tick_light_vec"]))


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    RayTracingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
