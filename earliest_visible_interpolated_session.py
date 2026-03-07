from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from precompute_earliest_grid import PrecomputedEarliestInterpolator


@dataclass
class SampledTrajectory3D:
    ts: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        ts: Sequence[float],
        xs: Sequence[float],
        ys: Sequence[float],
        zs: Sequence[float],
    ) -> "SampledTrajectory3D":
        t = np.asarray(ts, dtype=float).reshape(-1)
        x = np.asarray(xs, dtype=float).reshape(-1)
        y = np.asarray(ys, dtype=float).reshape(-1)
        z = np.asarray(zs, dtype=float).reshape(-1)
        if not (t.size == x.size == y.size == z.size):
            raise ValueError("ts/xs/ys/zs must have the same length.")
        if t.size < 2:
            raise ValueError("Trajectory must include at least 2 samples.")
        return cls(ts=t, xs=x, ys=y, zs=z)

    @classmethod
    def from_callable(
        cls,
        trajectory,
        tmin: float,
        tmax: float,
        samples: int = 2049,
    ) -> "SampledTrajectory3D":
        n = max(2, int(samples))
        ts = np.linspace(float(tmin), float(tmax), n, dtype=float)
        pts = np.asarray([trajectory(float(t)) for t in ts], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("trajectory(t) must return length-3 points.")
        return cls.from_arrays(ts, pts[:, 0], pts[:, 1], pts[:, 2])

    def eval_points(self, query_ts: np.ndarray) -> np.ndarray:
        qt = np.asarray(query_ts, dtype=float)
        x = np.interp(qt, self.ts, self.xs)
        y = np.interp(qt, self.ts, self.ys)
        z = np.interp(qt, self.ts, self.zs)
        return np.stack([x, y, z], axis=1)


class EarliestVisibleInterpolatedSession:
    """
    Reusable session for repeated earliest-visible queries with varying (B, t0).

    Designed to avoid repeated large table transfers:
    - precompute table is loaded once
    - backend arrays are materialized once via prepare_backend()
    - trajectory samples are held in memory once
    """

    def __init__(
        self,
        precompute_npz: Path | str,
        sampled_trajectory: SampledTrajectory3D,
        use_gpu: bool = True,
        batch_size: int = 5000,
        gpu_min_batch: int = 256,
    ) -> None:
        self.interp = PrecomputedEarliestInterpolator.from_npz(precompute_npz)
        self.sampled_trajectory = sampled_trajectory
        self.use_gpu = bool(use_gpu)
        self.batch_size = int(batch_size)
        self.gpu_min_batch = int(max(1, gpu_min_batch))
        self.interp.prepare_backend(use_gpu=self.use_gpu)

    @staticmethod
    def _find_first_bracket(ts: np.ndarray, fvals: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        for i in range(ts.size):
            fi = float(fvals[i])
            if np.isfinite(fi) and abs(fi) <= 1e-12:
                return None, float(ts[i])
        for i in range(1, ts.size):
            f0 = float(fvals[i - 1])
            f1 = float(fvals[i])
            if not (np.isfinite(f0) and np.isfinite(f1)):
                continue
            if f0 * f1 <= 0.0:
                return (float(ts[i - 1]), float(ts[i])), None
        return None, None

    def _eval_times(self, ts: np.ndarray, point_b: np.ndarray, t0: float):
        a_pts = self.sampled_trajectory.eval_points(ts)
        b_pts = np.repeat(np.asarray(point_b, dtype=float)[None, :], ts.size, axis=0)
        use_gpu_eval = bool(self.use_gpu and ts.size >= self.gpu_min_batch)
        out = self.interp.interpolate_pairs_3d(
            a_points_m=a_pts,
            b_points_m=b_pts,
            use_gpu=use_gpu_eval,
            batch_size=self.batch_size,
        )
        dt_plus = np.asarray(out["delta_t_plus_s"], dtype=float)
        dt_minus = np.asarray(out["delta_t_minus_s"], dtype=float)
        f_plus = ts + dt_plus - float(t0)
        f_minus = ts + dt_minus - float(t0)
        return out, f_plus, f_minus

    def _solve_root_bisection(
        self,
        side: str,
        bracket: Tuple[float, float],
        point_b: np.ndarray,
        t0: float,
        root_max_iter: int,
        root_tol_time: float,
    ) -> float:
        a, c = float(bracket[0]), float(bracket[1])
        for _ in range(int(root_max_iter)):
            m = 0.5 * (a + c)
            _, f_p, f_m = self._eval_times(np.asarray([a, m, c], dtype=float), point_b, t0=float(t0))
            fa = float(f_p[0] if side == "plus" else f_m[0])
            fm = float(f_p[1] if side == "plus" else f_m[1])
            fc = float(f_p[2] if side == "plus" else f_m[2])
            if abs(fm) <= 1e-12 or abs(c - a) <= float(root_tol_time):
                return m
            if np.isfinite(fa) and np.isfinite(fm) and (fa * fm <= 0.0):
                c = m
            elif np.isfinite(fm) and np.isfinite(fc):
                a = m
            else:
                return m
        return 0.5 * (a + c)

    def _build_observed(self, side: str, te: float, point_b: np.ndarray, t0: float) -> Optional[Dict[str, float]]:
        out, _, _ = self._eval_times(np.asarray([te], dtype=float), point_b, t0=float(t0))
        if side == "plus":
            ok = bool(np.asarray(out["ok_plus"], dtype=bool)[0])
            dt = float(np.asarray(out["delta_t_plus_s"], dtype=float)[0])
            gamma_b = float(np.asarray(out["gamma_at_b_plus_rad"], dtype=float)[0])
        else:
            ok = bool(np.asarray(out["ok_minus"], dtype=bool)[0])
            dt = float(np.asarray(out["delta_t_minus_s"], dtype=float)[0])
            gamma_b = float(np.asarray(out["gamma_at_b_minus_rad"], dtype=float)[0])
        if not ok or not np.isfinite(dt):
            return None
        return {
            "emission_time_s": float(te),
            "travel_time_s": float(dt),
            "arrival_time_s": float(te + dt),
            "gamma_at_b_rad": float(gamma_b),
        }

    def _find_local_bracket_from_seed(
        self,
        side: str,
        seed_te: float,
        point_b: np.ndarray,
        t0: float,
        t_lo: float,
        t_hi: float,
        base_step_s: float,
        expand_steps: int,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        seed = float(np.clip(seed_te, t_lo, t_hi))
        step = max(1e-6, float(base_step_s))
        for _ in range(max(1, int(expand_steps))):
            lo = max(t_lo, seed - step)
            hi = min(t_hi, seed + step)
            ts = np.asarray(sorted(set([lo, seed, hi])), dtype=float)
            _, f_p, f_m = self._eval_times(ts, point_b, t0=float(t0))
            fv = f_p if side == "plus" else f_m

            for i in range(ts.size):
                fi = float(fv[i])
                if np.isfinite(fi) and abs(fi) <= 1e-12:
                    return None, float(ts[i])
            for i in range(1, ts.size):
                f0 = float(fv[i - 1])
                f1 = float(fv[i])
                if not (np.isfinite(f0) and np.isfinite(f1)):
                    continue
                if f0 * f1 <= 0.0:
                    return (float(ts[i - 1]), float(ts[i])), None
            step *= 2.0
        return None, None

    def solve(
        self,
        point_b: Sequence[float],
        t0: float,
        tmin: float,
        tmax: float,
        scan_samples: int = 257,
        root_max_iter: int = 36,
        root_tol_time: float = 1e-6,
    ) -> Dict[str, object]:
        t_lo = float(tmin)
        t_hi = min(float(tmax), float(t0))
        if t_hi < t_lo:
            raise ValueError("Invalid time window: min(tmax, t0) must be >= tmin.")

        b = np.asarray(point_b, dtype=float).reshape(3)
        ts = np.linspace(t_lo, t_hi, max(3, int(scan_samples)), dtype=float)
        _, f_plus, f_minus = self._eval_times(ts, b, t0=float(t0))
        plus_bracket, plus_exact = self._find_first_bracket(ts, f_plus)
        minus_bracket, minus_exact = self._find_first_bracket(ts, f_minus)

        plus = None
        minus = None
        if plus_exact is not None:
            plus = self._build_observed("plus", plus_exact, b, float(t0))
        elif plus_bracket is not None:
            plus_te = self._solve_root_bisection("plus", plus_bracket, b, float(t0), int(root_max_iter), float(root_tol_time))
            plus = self._build_observed("plus", plus_te, b, float(t0))

        if minus_exact is not None:
            minus = self._build_observed("minus", minus_exact, b, float(t0))
        elif minus_bracket is not None:
            minus_te = self._solve_root_bisection(
                "minus", minus_bracket, b, float(t0), int(root_max_iter), float(root_tol_time)
            )
            minus = self._build_observed("minus", minus_te, b, float(t0))

        return {
            "observer_point_b": tuple(float(x) for x in b.tolist()),
            "observer_time_s": float(t0),
            "plus": plus,
            "minus": minus,
        }

    def solve_from_previous(
        self,
        point_b: Sequence[float],
        t0: float,
        tmin: float,
        tmax: float,
        previous_result: Optional[Dict[str, object]] = None,
        previous_t0: Optional[float] = None,
        scan_samples_fallback: int = 257,
        root_max_iter: int = 36,
        root_tol_time: float = 1e-6,
        seed_base_step_s: float = 0.1,
        seed_expand_steps: int = 8,
    ) -> Dict[str, object]:
        """
        Warm-start solver for sequential queries.
        Uses previous branch emission times as seeds and falls back to full scan when needed.
        """
        t_lo = float(tmin)
        t_hi = min(float(tmax), float(t0))
        if t_hi < t_lo:
            raise ValueError("Invalid time window: min(tmax, t0) must be >= tmin.")

        b = np.asarray(point_b, dtype=float).reshape(3)
        if previous_result is None:
            return self.solve(
                point_b=b,
                t0=float(t0),
                tmin=t_lo,
                tmax=t_hi,
                scan_samples=max(3, int(scan_samples_fallback)),
                root_max_iter=int(root_max_iter),
                root_tol_time=float(root_tol_time),
            )

        dt0 = float(t0 - previous_t0) if previous_t0 is not None else 0.0
        out: Dict[str, object] = {
            "observer_point_b": tuple(float(x) for x in b.tolist()),
            "observer_time_s": float(t0),
            "plus": None,
            "minus": None,
        }
        unresolved = []

        for side in ("plus", "minus"):
            prev_side = previous_result.get(side) if isinstance(previous_result, dict) else None
            prev_te = None
            if isinstance(prev_side, dict):
                prev_te = prev_side.get("emission_time_s")
            if prev_te is None or not np.isfinite(float(prev_te)):
                unresolved.append(side)
                continue

            seed = float(prev_te) + dt0
            bracket, exact = self._find_local_bracket_from_seed(
                side=side,
                seed_te=seed,
                point_b=b,
                t0=float(t0),
                t_lo=t_lo,
                t_hi=t_hi,
                base_step_s=max(float(seed_base_step_s), abs(dt0), float(root_tol_time) * 8.0),
                expand_steps=int(seed_expand_steps),
            )
            if exact is not None:
                out[side] = self._build_observed(side, exact, b, float(t0))
                if out[side] is None:
                    unresolved.append(side)
                continue
            if bracket is not None:
                te = self._solve_root_bisection(
                    side=side,
                    bracket=bracket,
                    point_b=b,
                    t0=float(t0),
                    root_max_iter=int(root_max_iter),
                    root_tol_time=float(root_tol_time),
                )
                out[side] = self._build_observed(side, te, b, float(t0))
                if out[side] is None:
                    unresolved.append(side)
                continue
            unresolved.append(side)

        if unresolved:
            full = self.solve(
                point_b=b,
                t0=float(t0),
                tmin=t_lo,
                tmax=t_hi,
                scan_samples=max(3, int(scan_samples_fallback)),
                root_max_iter=int(root_max_iter),
                root_tol_time=float(root_tol_time),
            )
            for side in unresolved:
                out[side] = full.get(side)

        return out
