from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from blackhole_geodesics import GeodesicSolution, SchwarzschildBlackHole, TimelikeTrajectoryResult


@dataclass(frozen=True)
class SampledCoordinateTrajectory3D:
    ts: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray

    def eval_point(self, t: float) -> Tuple[float, float, float]:
        tt = float(np.clip(t, float(self.ts[0]), float(self.ts[-1])))
        x = float(np.interp(tt, self.ts, self.xs))
        y = float(np.interp(tt, self.ts, self.ys))
        z = float(np.interp(tt, self.ts, self.zs))
        return (x, y, z)


def _path_for_direction(paths: Sequence[GeodesicSolution], direction: int) -> Optional[GeodesicSolution]:
    for p in paths:
        if int(p.direction) == int(direction):
            return p
    return None


class TimelikeVisibilitySession:
    """
    Earliest-visibility solver from an initial (position, velocity) state.

    The body trajectory is integrated once around the initial-condition epoch t=0
    and then sampled by interpolation during root solves.
    """

    def __init__(
        self,
        bh: SchwarzschildBlackHole,
        initial_position_m: Sequence[float],
        initial_velocity_m_s: Sequence[float],
        tmin: float,
        tmax: float,
        proper_time_step_s: float = 1e-3,
        integrator: str = "symplectic",
        max_steps: int = 400_000,
        max_proper_time_s: Optional[float] = None,
        escape_radius_m: Optional[float] = None,
        use_gpu: bool = False,
    ) -> None:
        self.bh = bh
        self.initial_position_m = tuple(float(x) for x in initial_position_m)
        self.initial_velocity_m_s = tuple(float(v) for v in initial_velocity_m_s)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.max_proper_time_s = None if max_proper_time_s is None else float(max_proper_time_s)
        if self.max_proper_time_s is not None and self.max_proper_time_s <= 0.0:
            raise ValueError("max_proper_time_s must be > 0 when provided.")
        self.escape_radius_m = None if escape_radius_m is None else float(escape_radius_m)
        self.use_gpu = bool(use_gpu)
        self._prev_solution_by_side: Dict[str, Dict[str, float]] = {}
        self._printed_backward_warning = False
        self.trajectory = self._build_coordinate_trajectory(
            proper_time_step_s=float(proper_time_step_s),
            integrator=str(integrator),
            max_steps=int(max_steps),
        )

    def _integrate_for_window(
        self,
        velocity_m_s: Tuple[float, float, float],
        target_coord_time_s: float,
        proper_time_step_s: float,
        integrator: str,
        max_steps: int,
    ) -> TimelikeTrajectoryResult:
        tau_guess = max(1e-3, 1.5 * abs(float(target_coord_time_s)))
        tau_limit = max(10.0, tau_guess)
        if self.max_proper_time_s is not None:
            tau_limit = min(tau_limit, float(self.max_proper_time_s))
        kwargs = {}
        if self.escape_radius_m is not None:
            kwargs["escape_radius_m"] = float(self.escape_radius_m)
        last = self.bh.integrate_timelike_trajectory(
            initial_position_m=self.initial_position_m,
            initial_velocity_m_s=velocity_m_s,
            proper_time_step_s=float(proper_time_step_s),
            max_proper_time_s=float(tau_limit),
            max_steps=int(max_steps),
            integrator=integrator,  # type: ignore[arg-type]
            **kwargs,
        )
        for _ in range(5):
            if not last.samples:
                return last
            t_end = float(last.samples[-1].coordinate_time_s)
            if t_end >= float(target_coord_time_s):
                return last
            if self.max_proper_time_s is not None and tau_limit >= float(self.max_proper_time_s) - 1e-15:
                return last
            tau_limit *= 2.0
            if self.max_proper_time_s is not None:
                tau_limit = min(tau_limit, float(self.max_proper_time_s))
            last = self.bh.integrate_timelike_trajectory(
                initial_position_m=self.initial_position_m,
                initial_velocity_m_s=velocity_m_s,
                proper_time_step_s=float(proper_time_step_s),
                max_proper_time_s=float(tau_limit),
                max_steps=int(max_steps),
                integrator=integrator,  # type: ignore[arg-type]
                **kwargs,
            )
        return last

    def _build_coordinate_trajectory(
        self,
        proper_time_step_s: float,
        integrator: str,
        max_steps: int,
    ) -> SampledCoordinateTrajectory3D:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        ts: list[float] = []

        # Backward segment from time-reversed initial velocity.
        if self.tmin < 0.0:
            v_back = tuple(-v for v in self.initial_velocity_m_s)
            back = self._integrate_for_window(
                velocity_m_s=v_back,
                target_coord_time_s=abs(float(self.tmin)),
                proper_time_step_s=proper_time_step_s,
                integrator=integrator,
                max_steps=max_steps,
            )
            back_points = []
            for s in back.samples:
                tt = -float(s.coordinate_time_s)
                back_points.append((tt, float(s.position_xyz_m[0]), float(s.position_xyz_m[1]), float(s.position_xyz_m[2])))
            back_points.sort(key=lambda p: p[0])
            for t, x, y, z in back_points:
                if t < self.tmin - 1e-12 or t > 0.0 + 1e-12:
                    continue
                ts.append(t)
                xs.append(x)
                ys.append(y)
                zs.append(z)

        # Ensure t=0 is present exactly once.
        if not ts or abs(ts[-1]) > 1e-12:
            ts.append(0.0)
            xs.append(self.initial_position_m[0])
            ys.append(self.initial_position_m[1])
            zs.append(self.initial_position_m[2])

        # Forward segment from original velocity.
        if self.tmax > 0.0:
            fwd = self._integrate_for_window(
                velocity_m_s=self.initial_velocity_m_s,
                target_coord_time_s=abs(float(self.tmax)),
                proper_time_step_s=proper_time_step_s,
                integrator=integrator,
                max_steps=max_steps,
            )
            for s in fwd.samples:
                tt = float(s.coordinate_time_s)
                if tt <= 0.0 + 1e-12:
                    continue
                if tt > self.tmax + 1e-12:
                    break
                ts.append(tt)
                xs.append(float(s.position_xyz_m[0]))
                ys.append(float(s.position_xyz_m[1]))
                zs.append(float(s.position_xyz_m[2]))

        t_arr = np.asarray(ts, dtype=float)
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        z_arr = np.asarray(zs, dtype=float)
        if t_arr.size < 2:
            raise RuntimeError("Trajectory sampling failed: need at least two time samples.")
        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        x_arr = x_arr[order]
        y_arr = y_arr[order]
        z_arr = z_arr[order]
        unique_t, idx = np.unique(t_arr, return_index=True)
        return SampledCoordinateTrajectory3D(
            ts=unique_t,
            xs=x_arr[idx],
            ys=y_arr[idx],
            zs=z_arr[idx],
        )

    def _f_side(
        self,
        side: str,
        te: float,
        observer_point_b: Tuple[float, float, float],
        observer_time_s: float,
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        direction = +1 if side == "plus" else -1
        point_a = self.trajectory.eval_point(float(te))
        result = self.bh.find_two_shortest_geodesics(
            point_a=point_a,
            point_b=observer_point_b,
            a_before_b=True,
            use_gpu=self.use_gpu,
        )
        path = _path_for_direction(result.paths, direction)
        if path is None:
            return float("inf"), None
        fval = float(te) + float(path.travel_time_s) - float(observer_time_s)
        observed = {
            "emission_time_s": float(te),
            "arrival_time_s": float(observer_time_s),
            "travel_time_s": float(path.travel_time_s),
            "gamma_at_b_rad": float(self.bh._arrival_angle_at_b(observer_point_b, path.impact_parameter_m)),
            "impact_parameter_m": float(path.impact_parameter_m),
            "direction": int(direction),
            "point_a_m": tuple(float(v) for v in point_a),
        }
        return fval, observed

    def _find_bracket(
        self,
        side: str,
        seed_te: float,
        observer_point_b: Tuple[float, float, float],
        observer_time_s: float,
        t_lo: float,
        t_hi: float,
        base_step_s: float = 0.05,
        expand_steps: int = 10,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Dict[str, float]]]:
        seed = float(np.clip(seed_te, t_lo, t_hi))
        f_seed, event_seed = self._f_side(side, seed, observer_point_b, observer_time_s)
        if np.isfinite(f_seed) and abs(f_seed) <= 1e-12:
            return None, event_seed
        step = max(1e-6, float(base_step_s))
        for _ in range(max(1, int(expand_steps))):
            lo = max(float(t_lo), seed - step)
            hi = min(float(t_hi), seed + step)
            f_lo, _ = self._f_side(side, lo, observer_point_b, observer_time_s)
            f_hi, _ = self._f_side(side, hi, observer_point_b, observer_time_s)
            if np.isfinite(f_lo) and abs(f_lo) <= 1e-12:
                _, ev = self._f_side(side, lo, observer_point_b, observer_time_s)
                return None, ev
            if np.isfinite(f_hi) and abs(f_hi) <= 1e-12:
                _, ev = self._f_side(side, hi, observer_point_b, observer_time_s)
                return None, ev
            if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi <= 0.0):
                return (float(lo), float(hi)), None
            step *= 2.0
        return None, None

    def _fallback_scan_bracket(
        self,
        side: str,
        observer_point_b: Tuple[float, float, float],
        observer_time_s: float,
        t_lo: float,
        t_hi: float,
        scan_samples: int = 129,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Dict[str, float]]]:
        ts = np.linspace(float(t_lo), float(t_hi), max(3, int(scan_samples)), dtype=float)
        vals: list[float] = []
        events: list[Optional[Dict[str, float]]] = []
        for t in ts:
            f, ev = self._f_side(side, float(t), observer_point_b, observer_time_s)
            vals.append(float(f))
            events.append(ev)
        for i, f in enumerate(vals):
            if np.isfinite(f) and abs(f) <= 1e-12:
                return None, events[i]
        for i in range(1, len(vals)):
            f0 = vals[i - 1]
            f1 = vals[i]
            if np.isfinite(f0) and np.isfinite(f1) and (f0 * f1 <= 0.0):
                return (float(ts[i - 1]), float(ts[i])), None
        return None, None

    def _solve_root_bracketed(
        self,
        side: str,
        bracket: Tuple[float, float],
        observer_point_b: Tuple[float, float, float],
        observer_time_s: float,
        root_max_iter: int,
        root_tol_time: float,
    ) -> float:
        a, b = float(bracket[0]), float(bracket[1])
        fa, _ = self._f_side(side, a, observer_point_b, observer_time_s)
        fb, _ = self._f_side(side, b, observer_point_b, observer_time_s)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return 0.5 * (a + b)
        if abs(fa) <= 1e-12:
            return a
        if abs(fb) <= 1e-12:
            return b
        for _ in range(max(1, int(root_max_iter))):
            # Safeguarded secant inside a valid bracket for faster convergence than pure bisection.
            denom = (fb - fa)
            if abs(denom) > 1e-15:
                x = b - fb * (b - a) / denom
            else:
                x = 0.5 * (a + b)
            if (x <= a) or (x >= b):
                x = 0.5 * (a + b)
            fx, _ = self._f_side(side, x, observer_point_b, observer_time_s)
            if (not np.isfinite(fx)) or abs(fx) <= 1e-12 or abs(b - a) <= float(root_tol_time):
                return float(x)
            if fa * fx <= 0.0:
                b, fb = x, fx
            else:
                a, fa = x, fx
        return 0.5 * (a + b)

    def solve(
        self,
        observer_point_b: Sequence[float],
        observer_time_s: float,
        root_max_iter: int = 12,
        root_tol_time: float = 1e-6,
        fallback_scan_samples: int = 65,
    ) -> Dict[str, object]:
        b = tuple(float(v) for v in observer_point_b)
        t0 = float(observer_time_s)
        if t0 < 0.0 and (not self._printed_backward_warning):
            print(
                "warning: observer_time_s is before initial conditions (t=0); "
                "solving on backward trajectory segment."
            )
            self._printed_backward_warning = True

        t_hi = min(float(self.tmax), float(t0))
        t_lo = float(self.tmin)
        if t_hi < t_lo:
            raise ValueError("Invalid solve window: min(tmax, observer_time_s) < tmin.")

        out: Dict[str, object] = {
            "observer_point_b": b,
            "observer_time_s": float(t0),
            "plus": None,
            "minus": None,
            "earliest": None,
        }

        for side in ("plus", "minus"):
            prev = self._prev_solution_by_side.get(side)
            if prev is not None:
                seed = float(prev["emission_time_s"]) + (float(t0) - float(prev["observer_time_s"]))
            else:
                p0 = np.asarray(self.initial_position_m, dtype=float)
                bb = np.asarray(b, dtype=float)
                seed = float(t0) - float(np.linalg.norm(bb - p0)) / 299_792_458.0
            seed = float(np.clip(seed, t_lo, t_hi))

            bracket, exact = self._find_bracket(
                side=side,
                seed_te=seed,
                observer_point_b=b,
                observer_time_s=float(t0),
                t_lo=t_lo,
                t_hi=t_hi,
                base_step_s=max(1e-3, 0.02 * max(1.0, abs(t0))),
                expand_steps=10,
            )
            if exact is not None:
                out[side] = exact
                self._prev_solution_by_side[side] = {
                    "emission_time_s": float(exact["emission_time_s"]),
                    "observer_time_s": float(t0),
                }
                continue

            if bracket is None:
                bracket, exact = self._fallback_scan_bracket(
                    side=side,
                    observer_point_b=b,
                    observer_time_s=float(t0),
                    t_lo=t_lo,
                    t_hi=t_hi,
                    scan_samples=max(5, int(fallback_scan_samples)),
                )
                if exact is not None:
                    out[side] = exact
                    self._prev_solution_by_side[side] = {
                        "emission_time_s": float(exact["emission_time_s"]),
                        "observer_time_s": float(t0),
                    }
                    continue
                if bracket is None:
                    continue

            te = self._solve_root_bracketed(
                side=side,
                bracket=bracket,
                observer_point_b=b,
                observer_time_s=float(t0),
                root_max_iter=int(root_max_iter),
                root_tol_time=float(root_tol_time),
            )
            _, ev = self._f_side(side, te, b, float(t0))
            out[side] = ev
            if ev is not None:
                self._prev_solution_by_side[side] = {
                    "emission_time_s": float(ev["emission_time_s"]),
                    "observer_time_s": float(t0),
                }

        cands = [out.get("plus"), out.get("minus")]
        cands = [c for c in cands if isinstance(c, dict)]
        if cands:
            out["earliest"] = min(cands, key=lambda c: float(c["emission_time_s"]))
        return out
