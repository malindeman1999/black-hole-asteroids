from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import C, SchwarzschildBlackHole

LEGACY_RS_M = 1.0


class SegmentedNullGeodesicGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Segmented Null Geodesic Sweep GUI")
        self.proc: subprocess.Popen[str] | None = None
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.run_lock = threading.Lock()

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.python_var = tk.StringVar(value=sys.executable)
        # Legacy fixed-scale mode: keep Rs at 1 meter.
        self.rs_m_var = tk.StringVar(value=f"{LEGACY_RS_M:.1f}")
        self.a_radius_rs_var = tk.StringVar(value="100.0")
        self.b_r_min_rs_var = tk.StringVar(value="1.6")
        self.b_r_max_rs_var = tk.StringVar(value="10.0")
        self.b_r_count_var = tk.StringVar(value="12")
        self.b_spacing_var = tk.StringVar(value="r3")
        self.a_phi_count_var = tk.StringVar(value="97")
        self.node_count_var = tk.StringVar(value="20")
        self.node_spacing_var = tk.StringVar(value="r3")
        self.optimizer_var = tk.StringVar(value="scipy")
        self.max_iter_var = tk.StringVar(value="250")
        self.opt_ftol_var = tk.StringVar(value="1e-9")
        self.opt_gtol_var = tk.StringVar(value="1e-6")
        self.output_npz_var = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "segmented_null_geodesic_two_family_sweep.npz")
        )
        self.scale_var = tk.StringVar(value="")

        row = 0
        for label, var in [
            ("Python", self.python_var),
            ("Rs (meters)", self.rs_m_var),
            ("A Radius (Rs)", self.a_radius_rs_var),
            ("B r min (Rs)", self.b_r_min_rs_var),
            ("B r max (Rs)", self.b_r_max_rs_var),
            ("B r count", self.b_r_count_var),
            ("B spacing (r3|linear)", self.b_spacing_var),
            ("A Phi Count", self.a_phi_count_var),
            ("Node Count", self.node_count_var),
            ("Optimizer (scipy|coord)", self.optimizer_var),
            ("Max Iter", self.max_iter_var),
            ("Opt Ftol", self.opt_ftol_var),
            ("Opt Gtol", self.opt_gtol_var),
            ("Output NPZ", self.output_npz_var),
        ]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(frame, textvariable=var, width=96).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1

        ttk.Label(frame, text="Node Spacing").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        node_spacing_row = ttk.Frame(frame)
        node_spacing_row.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Radiobutton(node_spacing_row, text="log", value="log", variable=self.node_spacing_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Radiobutton(node_spacing_row, text="r3", value="r3", variable=self.node_spacing_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Radiobutton(node_spacing_row, text="linear", value="linear", variable=self.node_spacing_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        row += 1

        ttk.Label(frame, text="Scale (from BH object)").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(frame, textvariable=self.scale_var).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=row, column=1, sticky="w", pady=(2, 8))
        self.btn_run = ttk.Button(btn_row, text="Run Sweep", command=self.run_sweep)
        self.btn_run_debug = ttk.Button(btn_row, text="Run Sweep (Debug)", command=self.run_sweep_debug)
        self.btn_run_first_two = ttk.Button(
            btn_row,
            text="Run First 30 Paths",
            command=self.run_first_thirty_paths,
        )
        self.btn_plot_times = ttk.Button(btn_row, text="Plot Times", command=self.plot_times)
        self.btn_stop = ttk.Button(btn_row, text="Stop", command=self.stop_current)
        for b in [self.btn_run, self.btn_run_debug, self.btn_run_first_two, self.btn_plot_times, self.btn_stop]:
            b.pack(side=tk.LEFT, padx=(0, 6))
        row += 1

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frame, textvariable=self.status_var).grid(row=row, column=1, sticky="w", pady=(0, 6))
        row += 1

        self.log = tk.Text(frame, height=22, width=130, wrap=tk.NONE)
        self.log.grid(row=row, column=0, columnspan=2, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log.yview)
        yscroll.grid(row=row, column=2, sticky="ns")
        self.log.configure(yscrollcommand=yscroll.set)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)
        self.rs_m_var.trace_add("write", self._on_rs_changed)
        self._refresh_scale_summary()
        self.root.after(80, self._drain_log_queue)
        self._set_running(False, "Idle")

    def _script(self, rel_path: str) -> str:
        return str(PROJECT_ROOT / rel_path)

    def _append_log(self, msg: str) -> None:
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._append_log(msg)
        self.root.after(80, self._drain_log_queue)

    def _set_running(self, running: bool, status: str) -> None:
        self.status_var.set(status)
        state = tk.DISABLED if running else tk.NORMAL
        for b in [self.btn_run, self.btn_run_debug, self.btn_run_first_two]:
            b.configure(state=state)
        self.btn_stop.configure(state=(tk.NORMAL if running else tk.DISABLED))

    def _run_subprocess(self, cmd: list[str], label: str) -> None:
        with self.run_lock:
            if self.proc is not None:
                self.log_q.put("Another process is already running.\n")
                return
            self._set_running(True, f"Running: {label}")
            self.log_q.put(f"\n[{label}] {' '.join(cmd)}\n")

            def _worker() -> None:
                try:
                    self.proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert self.proc.stdout is not None
                    for line in self.proc.stdout:
                        self.log_q.put(line)
                    rc = self.proc.wait()
                    self.log_q.put(f"[{label}] exit code: {rc}\n")
                    self.status_var.set("Idle" if rc == 0 else f"Failed: {label} (rc={rc})")
                except Exception as exc:
                    self.log_q.put(f"[{label}] ERROR: {exc}\n")
                    self.status_var.set(f"Error: {label}")
                finally:
                    with self.run_lock:
                        self.proc = None
                    self.root.after(0, lambda: self._set_running(False, self.status_var.get()))

            threading.Thread(target=_worker, daemon=True).start()

    def _try_build_black_hole(self) -> tuple[SchwarzschildBlackHole | None, str | None]:
        # Reverted legacy behavior: always solve at Rs=1 meter.
        rs_m = float(LEGACY_RS_M)
        try:
            bh = SchwarzschildBlackHole.from_diameter_light_seconds((2.0 * rs_m) / C)
        except Exception as exc:
            return None, str(exc)
        return bh, None

    def _refresh_scale_summary(self) -> None:
        bh, err = self._try_build_black_hole()
        if bh is None:
            self.scale_var.set(f"Invalid: {err}")
            return
        self.scale_var.set(
            f"Rs={bh.schwarzschild_radius_m:.6e} m | "
            f"D={bh.diameter_light_seconds:.6g} light-seconds | "
            f"M={bh.mass_kg:.6e} kg"
        )

    def _on_rs_changed(self, *_: object) -> None:
        # Keep the UI field pinned to legacy fixed value.
        if self.rs_m_var.get().strip() != f"{LEGACY_RS_M:.1f}":
            self.rs_m_var.set(f"{LEGACY_RS_M:.1f}")
        self._refresh_scale_summary()

    def _base_cmd(self) -> list[str] | None:
        bh, err = self._try_build_black_hole()
        if bh is None:
            self.log_q.put(f"[Run] {err}\n")
            return None

        self.log_q.put(
            "[Run] Using BH object scale: "
            f"D={bh.diameter_light_seconds:.6g} light-seconds, "
            f"Rs={bh.schwarzschild_radius_m:.6e} m, "
            f"M={bh.mass_kg:.6e} kg\n"
        )

        cmd = [
            self.python_var.get(),
            "-u",
            self._script("geodesics/segmented_null_geodesic_two_family_sweep.py"),
            "--rs-m",
            f"{bh.schwarzschild_radius_m:.17g}",
            "--a-radius-rs",
            self.a_radius_rs_var.get(),
            "--b-r-min-rs",
            self.b_r_min_rs_var.get(),
            "--b-r-max-rs",
            self.b_r_max_rs_var.get(),
            "--b-r-count",
            self.b_r_count_var.get(),
            "--b-spacing",
            self.b_spacing_var.get(),
            "--a-phi-count",
            self.a_phi_count_var.get(),
            "--node-count",
            self.node_count_var.get(),
            "--node-spacing",
            self.node_spacing_var.get(),
            "--optimizer",
            self.optimizer_var.get(),
            "--max-iter",
            self.max_iter_var.get(),
            "--opt-ftol",
            self.opt_ftol_var.get(),
            "--opt-gtol",
            self.opt_gtol_var.get(),
            "--output",
            self.output_npz_var.get(),
        ]
        return cmd

    def run_sweep(self) -> None:
        cmd = self._base_cmd()
        if cmd is None:
            return
        cmd.append("--no-debug-show-rings")
        cmd.append("--no-debug-pause-rings")
        self._run_subprocess(cmd, "Segmented Sweep")

    def run_sweep_debug(self) -> None:
        cmd = self._base_cmd()
        if cmd is None:
            return
        cmd.append("--debug-show-rings")
        cmd.append("--debug-pause-rings")
        self._run_subprocess(cmd, "Segmented Sweep (Debug)")

    def run_first_thirty_paths(self) -> None:
        cmd = self._base_cmd()
        if cmd is None:
            return
        try:
            nominal_count = int(self.a_phi_count_var.get().strip())
        except Exception:
            nominal_count = 97
        if nominal_count < 2:
            nominal_count = 97
        dphi = (2.0 * float(np.pi)) / float(nominal_count)
        # Fast debug mode: solve the seed path and the next 29 continuation
        # paths on the first B ring (b-r-min-rs), using small angular steps.
        cmd.extend(
            [
                "--a-phi-count",
                "30",
                "--a-phi-step-rad",
                f"{dphi:.17g}",
                "--b-r-count",
                "1",
            ]
        )
        cmd.append("--debug-show-rings")
        cmd.append("--debug-pause-rings")
        self._run_subprocess(cmd, "Segmented Sweep (First 30 Paths)")

    def stop_current(self) -> None:
        with self.run_lock:
            if self.proc is None:
                return
            try:
                self.proc.terminate()
                self.log_q.put("[Stop] terminate signal sent.\n")
            except Exception as exc:
                self.log_q.put(f"[Stop] ERROR: {exc}\n")

    def plot_times(self) -> None:
        npz_path = Path(self.output_npz_var.get()).expanduser()
        if not npz_path.exists():
            self.log_q.put(f"[Plot Times] NPZ not found: {npz_path}\n")
            return
        try:
            with np.load(npz_path) as data:
                required = ["time_plus_s", "time_minus_s", "proper_time_plus_s", "proper_time_minus_s"]
                missing = [k for k in required if k not in data]
                if missing:
                    self.log_q.put(
                        "[Plot Times] Missing key(s) in NPZ: "
                        + ", ".join(missing)
                        + ". Re-run sweep with updated solver.\n"
                    )
                    return
                t_plus = np.asarray(data["time_plus_s"], dtype=float).reshape(-1)
                t_minus = np.asarray(data["time_minus_s"], dtype=float).reshape(-1)
                tau_plus = np.asarray(data["proper_time_plus_s"], dtype=float).reshape(-1)
                tau_minus = np.asarray(data["proper_time_minus_s"], dtype=float).reshape(-1)
                bh_diam_ls = None
                if "bh_diameter_light_seconds" in data:
                    bh_diam_ls = float(np.asarray(data["bh_diameter_light_seconds"], dtype=float).reshape(()))
        except Exception as exc:
            self.log_q.put(f"[Plot Times] Failed to load NPZ: {exc}\n")
            return

        finite_t = np.concatenate([t_plus[np.isfinite(t_plus)], t_minus[np.isfinite(t_minus)]])
        if finite_t.size and float(np.nanmax(np.abs(finite_t))) < 1e-3:
            if bh_diam_ls is not None and bh_diam_ls < 1e-6:
                self.log_q.put(
                    "[Plot Times] Note: durations are microseconds because this NPZ was generated "
                    f"with a tiny black-hole scale (diameter={bh_diam_ls:.3e} light-seconds, likely rs_m=1 meter).\n"
                )
                self.log_q.put(
                    "[Plot Times] For legacy tiny-scale timing, run with Rs (meters)=1.0.\n"
                )

        try:
            import matplotlib.pyplot as plt

            n = int(max(t_plus.size, t_minus.size))
            idx = np.arange(n, dtype=int)
            fig, ax = plt.subplots(1, 1, figsize=(11, 5.2))
            if t_plus.size:
                ax.plot(idx[: t_plus.size], t_plus, color="tab:blue", lw=1.2, label="coordinate time (+)")
            if t_minus.size:
                ax.plot(idx[: t_minus.size], t_minus, color="tab:orange", lw=1.2, label="coordinate time (-)")
            if tau_plus.size:
                ax.plot(idx[: tau_plus.size], tau_plus, color="tab:green", lw=1.2, label="proper time (+)")
            if tau_minus.size:
                ax.plot(idx[: tau_minus.size], tau_minus, color="tab:red", lw=1.2, label="proper time (-)")
            ax.set_xlabel("Path number (flattened index)")
            ax.set_ylabel("Time (s)")
            ax.set_title("Coordinate and Proper Time vs Path Number")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            self.log_q.put(f"[Plot Times] Showing time plot from {npz_path}\n")
            plt.show()
        except Exception as exc:
            self.log_q.put(f"[Plot Times] Plot failed: {exc}\n")


def main() -> None:
    root = tk.Tk()
    SegmentedNullGeodesicGUI(root)
    root.geometry("1320x780")
    root.minsize(1080, 620)
    root.mainloop()


if __name__ == "__main__":
    main()
