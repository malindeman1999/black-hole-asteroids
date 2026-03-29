from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SkyPrecomputePlotGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sky Precompute + Plot GUI")
        self.proc: subprocess.Popen[str] | None = None
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.run_lock = threading.Lock()

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.python_var = tk.StringVar(value=sys.executable)
        self.quality_var = tk.StringVar(value="high")
        self.use_gpu_var = tk.BooleanVar(value=False)
        self.sky_radius_var = tk.StringVar(value="100.0")
        self.a_phi_count_var = tk.StringVar(value="97")
        self.true_path_samples_var = tk.StringVar(value="7000")
        self.max_step_rs_var = tk.StringVar(value="0.01")
        self.output_npz_var = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "earliest_angles_sky_100rs_fixed_ar_two_family_solver.npz")
        )
        self.plot_png_var = tk.StringVar(
            value=str(PROJECT_ROOT / "sky_projections" / "precomputed_sky_a_angle_outermost_plus_true_paths.png")
        )
        self.max_paths_var = tk.StringVar(value="20")
        self.plot_show_var = tk.BooleanVar(value=True)

        row = 0
        for label, var in [
            ("Python", self.python_var),
            ("Quality", self.quality_var),
            ("Sky Radius (rs)", self.sky_radius_var),
            ("A Phi Count", self.a_phi_count_var),
            ("True Path Samples", self.true_path_samples_var),
            ("Max Step (Rs)", self.max_step_rs_var),
            ("Output NPZ", self.output_npz_var),
            ("Plot PNG", self.plot_png_var),
            ("Plot Max Paths", self.max_paths_var),
        ]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(frame, textvariable=var, width=96).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1

        opt_row = ttk.Frame(frame)
        opt_row.grid(row=row, column=1, sticky="w", pady=(2, 8))
        ttk.Checkbutton(opt_row, text="Use GPU", variable=self.use_gpu_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(opt_row, text="Show Plot Window", variable=self.plot_show_var).pack(side=tk.LEFT, padx=(0, 14))
        row += 1

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=row, column=1, sticky="w", pady=(2, 8))
        self.btn_save = ttk.Button(btn_row, text="Run Saver", command=self.run_saver)
        self.btn_save_debug = ttk.Button(btn_row, text="Run Saver (Debug Show)", command=self.run_saver_debug)
        self.btn_plot = ttk.Button(btn_row, text="Plot Saved Data", command=self.run_plotter)
        self.btn_both = ttk.Button(btn_row, text="Run Saver + Plot", command=self.run_both)
        self.btn_stop = ttk.Button(btn_row, text="Stop", command=self.stop_current)
        for b in [self.btn_save, self.btn_save_debug, self.btn_plot, self.btn_both, self.btn_stop]:
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
        self.root.after(80, self._drain_log_queue)
        self._set_running(False, "Idle")

    def _script(self, rel_path: str) -> str:
        return str(PROJECT_ROOT / rel_path)

    def _gpu_flag(self) -> str:
        return "--use-gpu" if self.use_gpu_var.get() else "--no-use-gpu"

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
        for b in [self.btn_save, self.btn_save_debug, self.btn_plot, self.btn_both]:
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

    def _run_sequential(self, steps: list[tuple[str, list[str]]]) -> None:
        with self.run_lock:
            if self.proc is not None:
                self.log_q.put("Another process is already running.\n")
                return
            self._set_running(True, "Running sequence")

            def _worker() -> None:
                final_status = "Idle"
                try:
                    for label, cmd in steps:
                        self.log_q.put(f"\n[{label}] {' '.join(cmd)}\n")
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
                        if rc != 0:
                            final_status = f"Failed: {label} (rc={rc})"
                            return
                    final_status = "Idle"
                except Exception as exc:
                    self.log_q.put(f"[sequence] ERROR: {exc}\n")
                    final_status = "Error"
                finally:
                    with self.run_lock:
                        self.proc = None
                    self.root.after(0, lambda: self._set_running(False, final_status))

            threading.Thread(target=_worker, daemon=True).start()

    def run_saver(self) -> None:
        cmd = [
            self.python_var.get(),
            "-u",
            self._script("solver/precompute_sky_fixed_radius_table_two_family_solver.py"),
            "--quality",
            self.quality_var.get(),
            "--sky-radius-rs",
            self.sky_radius_var.get(),
            "--a-phi-count",
            self.a_phi_count_var.get(),
            "--true-path-samples",
            self.true_path_samples_var.get(),
            "--max-spatial-step-rs",
            self.max_step_rs_var.get(),
            self._gpu_flag(),
            "--output",
            self.output_npz_var.get(),
        ]
        self._run_subprocess(cmd, "Saver")

    def run_saver_debug(self) -> None:
        cmd = [
            self.python_var.get(),
            "-u",
            self._script("solver/precompute_sky_fixed_radius_table_two_family_solver.py"),
            "--quality",
            self.quality_var.get(),
            "--sky-radius-rs",
            self.sky_radius_var.get(),
            "--a-phi-count",
            self.a_phi_count_var.get(),
            "--true-path-samples",
            self.true_path_samples_var.get(),
            "--max-spatial-step-rs",
            self.max_step_rs_var.get(),
            self._gpu_flag(),
            "--debug-show-rings",
            "--debug-pause-rings",
            "--output",
            self.output_npz_var.get(),
        ]
        self._run_subprocess(cmd, "Saver (Debug Show)")

    def run_plotter(self) -> None:
        cmd = [
            self.python_var.get(),
            "-u",
            self._script("sky_projections/plot_precomputed_sky_a_angles_outer_plus_true_paths.py"),
            "--input",
            self.output_npz_var.get(),
            "--max-paths",
            self.max_paths_var.get(),
            "--output",
            self.plot_png_var.get(),
            "--show" if self.plot_show_var.get() else "--no-show",
        ]
        self._run_subprocess(cmd, "Plotter")

    def run_both(self) -> None:
        saver_cmd = [
            self.python_var.get(),
            "-u",
            self._script("solver/precompute_sky_fixed_radius_table_two_family_solver.py"),
            "--quality",
            self.quality_var.get(),
            "--sky-radius-rs",
            self.sky_radius_var.get(),
            "--a-phi-count",
            self.a_phi_count_var.get(),
            "--true-path-samples",
            self.true_path_samples_var.get(),
            "--max-spatial-step-rs",
            self.max_step_rs_var.get(),
            self._gpu_flag(),
            "--output",
            self.output_npz_var.get(),
        ]
        plot_cmd = [
            self.python_var.get(),
            "-u",
            self._script("sky_projections/plot_precomputed_sky_a_angles_outer_plus_true_paths.py"),
            "--input",
            self.output_npz_var.get(),
            "--max-paths",
            self.max_paths_var.get(),
            "--output",
            self.plot_png_var.get(),
            "--show" if self.plot_show_var.get() else "--no-show",
        ]
        self._run_sequential([("Saver", saver_cmd), ("Plotter", plot_cmd)])

    def stop_current(self) -> None:
        with self.run_lock:
            if self.proc is None:
                return
            try:
                self.proc.terminate()
                self.log_q.put("[Stop] terminate signal sent.\n")
            except Exception as exc:
                self.log_q.put(f"[Stop] ERROR: {exc}\n")


def main() -> None:
    root = tk.Tk()
    SkyPrecomputePlotGUI(root)
    root.geometry("1300x760")
    root.minsize(1050, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
