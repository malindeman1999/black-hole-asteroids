from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GeodesicState:
    """
    Null geodesic state in Schwarzschild coordinates (equatorial plane).

    Units:
    - r is in Rs (Schwarzschild-radius units).
    - t is in Rs/c.
    - lambda_affine is arbitrary affine parameter.
    """

    t: float
    r: float
    phi: float
    p_r: float
    p_t: float
    p_phi: float


@dataclass(frozen=True)
class TraceSample:
    lambda_affine: float
    t: float
    r: float
    phi: float
    p_r: float
    null_residual: float


@dataclass(frozen=True)
class TraceResult:
    status: Literal["ok", "hit_horizon", "escaped", "max_steps", "integration_error"]
    samples: Tuple[TraceSample, ...]


@dataclass(frozen=True)
class BackTimeTraceResult:
    status: Literal["reached_back_time", "hit_horizon", "escaped", "max_steps", "integration_error", "max_rounds"]
    target_back_time_s: float
    target_t_rsc: float
    reached_target: bool
    lambda_at_target: Optional[float]
    rounds: int
    samples: Tuple[TraceSample, ...]


def _null_residual(r: float, p_r: float, p_t: float, p_phi: float) -> float:
    f = 1.0 - 1.0 / r
    return (-p_t * p_t / f) + (f * p_r * p_r) + (p_phi * p_phi / (r * r))


def _rhs(y: np.ndarray, p_t: float, p_phi: float) -> np.ndarray:
    t, r, phi, p_r = float(y[0]), float(y[1]), float(y[2]), float(y[3])
    if r <= 1.0:
        raise ValueError("r <= 1 encountered at/inside the horizon in Rs units.")
    f = 1.0 - 1.0 / r
    if f <= 0.0:
        raise ValueError("Non-positive lapse encountered.")

    dt_dlambda = -p_t / f
    dr_dlambda = f * p_r
    dphi_dlambda = p_phi / (r * r)
    dp_r_dlambda = -0.5 * ((p_t * p_t) / (r * r * f * f) + (p_r * p_r) / (r * r) - 2.0 * (p_phi * p_phi) / (r * r * r))
    return np.asarray([dt_dlambda, dr_dlambda, dphi_dlambda, dp_r_dlambda], dtype=float)


def _rk4_step(y: np.ndarray, h: float, p_t: float, p_phi: float) -> np.ndarray:
    k1 = _rhs(y, p_t=p_t, p_phi=p_phi)
    k2 = _rhs(y + 0.5 * h * k1, p_t=p_t, p_phi=p_phi)
    k3 = _rhs(y + 0.5 * h * k2, p_t=p_t, p_phi=p_phi)
    k4 = _rhs(y + h * k3, p_t=p_t, p_phi=p_phi)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _adaptive_rk4_step(
    y: np.ndarray,
    h: float,
    p_t: float,
    p_phi: float,
    rtol: float,
    atol: float,
) -> Tuple[bool, np.ndarray, float]:
    y_full = _rk4_step(y, h=h, p_t=p_t, p_phi=p_phi)
    y_half = _rk4_step(y, h=0.5 * h, p_t=p_t, p_phi=p_phi)
    y_half = _rk4_step(y_half, h=0.5 * h, p_t=p_t, p_phi=p_phi)

    scale = atol + rtol * np.maximum(np.abs(y_half), np.abs(y_full))
    err = np.sqrt(np.mean(((y_half - y_full) / scale) ** 2))
    if not np.isfinite(err):
        return False, y, 0.5 * h

    if err <= 1.0:
        if err == 0.0:
            h_next = 2.0 * h
        else:
            h_next = h * min(2.0, max(0.2, 0.9 * (1.0 / err) ** 0.2))
        return True, y_half, h_next

    h_next = h * max(0.1, 0.9 * (1.0 / err) ** 0.2)
    return False, y, h_next


def _spatial_step_limiter(y: np.ndarray, p_t: float, p_phi: float, max_spatial_step_rs: float) -> float:
    dy = _rhs(y, p_t=p_t, p_phi=p_phi)
    r = float(y[1])
    dr = float(dy[1])
    dphi = float(dy[2])
    speed_rs_per_lambda = sqrt(dr * dr + (r * dphi) * (r * dphi))
    if speed_rs_per_lambda <= 1e-15:
        return float("inf")
    return max_spatial_step_rs / speed_rs_per_lambda


class NullGeodesicSolverRS:
    """
    Adaptive affine-parameter null geodesic solver in Schwarzschild spacetime.

    All radial distances are normalized by Schwarzschild radius Rs.
    """

    def initial_state_from_local_angle(
        self,
        r0_rs: float,
        phi0_rad: float,
        alpha_rad: float,
        *,
        t0_rsc: float = 0.0,
        frequency_scale: float = 1.0,
        time_orientation: Literal["future", "past"] = "past",
    ) -> GeodesicState:
        """
        Build initial state from a static-observer local launch angle.

        alpha_rad is measured from +radial direction toward +phi direction
        in the local orthonormal frame.
        """

        r0 = float(r0_rs)
        if r0 <= 1.0:
            raise ValueError("r0_rs must be > 1.")
        f0 = 1.0 - 1.0 / r0
        s0 = sqrt(f0)

        k_t_hat = float(frequency_scale) if time_orientation == "future" else -float(frequency_scale)
        k_r_hat = float(frequency_scale) * cos(float(alpha_rad))
        k_phi_hat = float(frequency_scale) * sin(float(alpha_rad))

        p_t = -s0 * k_t_hat
        p_r = k_r_hat / s0
        p_phi = r0 * k_phi_hat

        return GeodesicState(
            t=float(t0_rsc),
            r=r0,
            phi=float(phi0_rad),
            p_r=float(p_r),
            p_t=float(p_t),
            p_phi=float(p_phi),
        )

    def initial_state_from_observer(
        self,
        observer_b_xyz_rs: Tuple[float, float, float],
        incoming_angle_at_b_rad: float,
        *,
        t_b_rsc: float = 0.0,
        frequency_scale: float = 1.0,
    ) -> GeodesicState:
        """
        Build a back-tracing initial state from observer position B and observer look angle.

        Parameters:
        - observer_b_xyz_rs: observer position in Rs units.
        - incoming_angle_at_b_rad: observer look angle at B, measured from +radial
          toward +phi in the static local frame at B.

        Notes:
        - The solver is equatorial (theta = pi/2), so z must be ~0.
        - This angle is interpreted as the back-trace launch direction (the look direction).
        """

        bx, by, bz = float(observer_b_xyz_rs[0]), float(observer_b_xyz_rs[1]), float(observer_b_xyz_rs[2])
        if abs(bz) > 1e-10:
            raise ValueError("observer_b_xyz_rs z-component must be ~0 for this equatorial solver.")
        r_b = sqrt(bx * bx + by * by)
        if r_b <= 1.0:
            raise ValueError("Observer B must satisfy r_B > 1 in Rs units.")
        phi_b = float(np.arctan2(by, bx))

        return self.initial_state_from_local_angle(
            r0_rs=r_b,
            phi0_rad=phi_b,
            alpha_rad=float(incoming_angle_at_b_rad),
            t0_rsc=float(t_b_rsc),
            frequency_scale=float(frequency_scale),
            time_orientation="past",
        )

    def trace(
        self,
        initial: GeodesicState,
        *,
        lambda_max: float = 200.0,
        h0: float = 1e-2,
        max_spatial_step_rs: float = 0.1,
        max_affine_step: float = 0.5,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        r_min: float = 1.0 + 1e-6,
        r_max: float = 1e4,
        max_steps: int = 200000,
    ) -> TraceResult:
        if lambda_max <= 0.0:
            raise ValueError("lambda_max must be > 0.")
        if h0 <= 0.0:
            raise ValueError("h0 must be > 0.")
        if max_spatial_step_rs <= 0.0:
            raise ValueError("max_spatial_step_rs must be > 0.")

        p_t = float(initial.p_t)
        p_phi = float(initial.p_phi)
        y0 = np.asarray([initial.t, initial.r, initial.phi, initial.p_r], dtype=float)
        samples: List[TraceSample] = [
            TraceSample(
                lambda_affine=0.0,
                t=float(y0[0]),
                r=float(y0[1]),
                phi=float(y0[2]),
                p_r=float(y0[3]),
                null_residual=float(_null_residual(float(y0[1]), float(y0[3]), p_t, p_phi)),
            )
        ]
        status, _y, _lam, _h, _steps_used = self._trace_segment(
            y_start=y0,
            lam_start=0.0,
            h_start=float(h0),
            lambda_stop=float(lambda_max),
            p_t=p_t,
            p_phi=p_phi,
            samples=samples,
            max_spatial_step_rs=float(max_spatial_step_rs),
            max_affine_step=float(max_affine_step),
            rtol=float(rtol),
            atol=float(atol),
            r_min=float(r_min),
            r_max=float(r_max),
            max_steps=int(max_steps),
        )
        return TraceResult(status=status, samples=tuple(samples))

    def _trace_segment(
        self,
        *,
        y_start: np.ndarray,
        lam_start: float,
        h_start: float,
        lambda_stop: float,
        p_t: float,
        p_phi: float,
        samples: List[TraceSample],
        max_spatial_step_rs: float,
        max_affine_step: float,
        rtol: float,
        atol: float,
        r_min: float,
        r_max: float,
        max_steps: int,
    ) -> Tuple[Literal["ok", "hit_horizon", "escaped", "max_steps", "integration_error"], np.ndarray, float, float, int]:
        y = np.asarray(y_start, dtype=float).copy()
        lam = float(lam_start)
        h = max(1e-10, float(h_start))

        status: Literal["ok", "hit_horizon", "escaped", "max_steps", "integration_error"] = "ok"
        steps_used = 0
        for _ in range(int(max_steps)):
            steps_used += 1
            r_now = float(y[1])
            if r_now <= r_min:
                status = "hit_horizon"
                break
            if r_now >= r_max:
                status = "escaped"
                break
            if lam >= lambda_stop:
                status = "ok"
                break

            h_cap_spatial = _spatial_step_limiter(y, p_t=p_t, p_phi=p_phi, max_spatial_step_rs=max_spatial_step_rs)
            h_try = min(float(h), float(max_affine_step), float(h_cap_spatial), float(lambda_stop - lam))
            h_try = max(h_try, 1e-10)

            try:
                accept, y_new, h_new = _adaptive_rk4_step(
                    y,
                    h=h_try,
                    p_t=p_t,
                    p_phi=p_phi,
                    rtol=float(rtol),
                    atol=float(atol),
                )
            except Exception:
                status = "integration_error"
                break

            if accept:
                lam += h_try
                y = y_new
                samples.append(
                    TraceSample(
                        lambda_affine=lam,
                        t=float(y[0]),
                        r=float(y[1]),
                        phi=float(y[2]),
                        p_r=float(y[3]),
                        null_residual=float(_null_residual(float(y[1]), float(y[3]), p_t, p_phi)),
                    )
                )
                h = max(1e-10, float(h_new))
            else:
                h = max(1e-10, float(h_new))
        else:
            status = "max_steps"

        return status, y, lam, h, steps_used

    def trace_until_back_time(
        self,
        initial: GeodesicState,
        *,
        back_time_s: float,
        rs_m: float,
        initial_lambda_span: float = 25.0,
        max_rounds: int = 20,
        h0: float = 1e-2,
        max_spatial_step_rs: float = 0.1,
        max_affine_step: float = 0.5,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        r_min: float = 1.0 + 1e-6,
        r_max: float = 1e4,
        max_steps_total: int = 400000,
    ) -> BackTimeTraceResult:
        """
        Iteratively extend integration until requested backward coordinate-time is reached.

        Strategy:
        - Integrate over affine span `initial_lambda_span`.
        - If back-time target not yet reached, double the additional affine span
          and continue from the last computed sample.
        """

        if back_time_s <= 0.0:
            raise ValueError("back_time_s must be > 0.")
        if rs_m <= 0.0:
            raise ValueError("rs_m must be > 0.")
        if initial_lambda_span <= 0.0:
            raise ValueError("initial_lambda_span must be > 0.")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        c_si = 299_792_458.0
        target_back_rsc = float(back_time_s) * c_si / float(rs_m)
        target_t_rsc = float(initial.t) - target_back_rsc

        p_t = float(initial.p_t)
        p_phi = float(initial.p_phi)
        y = np.asarray([initial.t, initial.r, initial.phi, initial.p_r], dtype=float)
        lam = 0.0
        h = float(h0)
        samples: List[TraceSample] = [
            TraceSample(
                lambda_affine=0.0,
                t=float(initial.t),
                r=float(initial.r),
                phi=float(initial.phi),
                p_r=float(initial.p_r),
                null_residual=float(_null_residual(float(initial.r), float(initial.p_r), p_t, p_phi)),
            )
        ]

        rounds = 0
        steps_used_total = 0
        span = float(initial_lambda_span)
        status: Literal["reached_back_time", "hit_horizon", "escaped", "max_steps", "integration_error", "max_rounds"] = "max_rounds"
        lambda_at_target: Optional[float] = None

        for rr in range(int(max_rounds)):
            rounds = rr + 1
            lambda_stop = lam + span
            steps_left = max(1, int(max_steps_total - steps_used_total))

            seg_status, y, lam, h, seg_steps = self._trace_segment(
                y_start=y,
                lam_start=lam,
                h_start=h,
                lambda_stop=lambda_stop,
                p_t=p_t,
                p_phi=p_phi,
                samples=samples,
                max_spatial_step_rs=float(max_spatial_step_rs),
                max_affine_step=float(max_affine_step),
                rtol=float(rtol),
                atol=float(atol),
                r_min=float(r_min),
                r_max=float(r_max),
                max_steps=steps_left,
            )
            steps_used_total += int(seg_steps)

            if samples[-1].t <= target_t_rsc:
                status = "reached_back_time"
                break
            if seg_status in ("hit_horizon", "escaped", "integration_error", "max_steps"):
                status = seg_status
                break
            if steps_used_total >= int(max_steps_total):
                status = "max_steps"
                break

            span *= 2.0

        if status == "reached_back_time":
            samples = self._truncate_samples_at_target_t(samples=samples, target_t_rsc=target_t_rsc)
            if len(samples) > 0:
                lambda_at_target = float(samples[-1].lambda_affine)

        return BackTimeTraceResult(
            status=status,
            target_back_time_s=float(back_time_s),
            target_t_rsc=float(target_t_rsc),
            reached_target=bool(status == "reached_back_time"),
            lambda_at_target=lambda_at_target,
            rounds=int(rounds),
            samples=tuple(samples),
        )

    @staticmethod
    def closest_sample_at_t(samples: Tuple[TraceSample, ...], target_t_rsc: float) -> TraceSample:
        if len(samples) == 0:
            raise ValueError("samples cannot be empty.")
        return min(samples, key=lambda s: abs(s.t - float(target_t_rsc)))

    @staticmethod
    def _truncate_samples_at_target_t(samples: List[TraceSample], target_t_rsc: float) -> List[TraceSample]:
        if len(samples) < 2:
            return list(samples)

        # Find first crossing where t goes from above target to below/equal target.
        cross_i = -1
        for i in range(1, len(samples)):
            if samples[i - 1].t >= target_t_rsc and samples[i].t <= target_t_rsc:
                cross_i = i
                break
        if cross_i < 0:
            return list(samples)

        prev_s = samples[cross_i - 1]
        next_s = samples[cross_i]

        # Exact hit: keep through this sample.
        if abs(next_s.t - target_t_rsc) <= 1e-15:
            return list(samples[: cross_i + 1])

        dt = next_s.t - prev_s.t
        if abs(dt) <= 1e-15:
            return list(samples[: cross_i + 1])

        w = (target_t_rsc - prev_s.t) / dt
        w = max(0.0, min(1.0, float(w)))

        interp = TraceSample(
            lambda_affine=float(prev_s.lambda_affine + w * (next_s.lambda_affine - prev_s.lambda_affine)),
            t=float(target_t_rsc),
            r=float(prev_s.r + w * (next_s.r - prev_s.r)),
            phi=float(prev_s.phi + w * (next_s.phi - prev_s.phi)),
            p_r=float(prev_s.p_r + w * (next_s.p_r - prev_s.p_r)),
            null_residual=float(prev_s.null_residual + w * (next_s.null_residual - prev_s.null_residual)),
        )
        out = list(samples[:cross_i])
        out.append(interp)
        return out

    @staticmethod
    def sample_to_xy(sample: TraceSample) -> Tuple[float, float]:
        return float(sample.r * cos(sample.phi)), float(sample.r * sin(sample.phi))


if __name__ == "__main__":
    solver = NullGeodesicSolverRS()

    # Example: back-trace from observer B and observed incoming angle.
    state0 = solver.initial_state_from_observer(
        observer_b_xyz_rs=(10.0, 0.0, 0.0),
        incoming_angle_at_b_rad=0.25,
    )
    result = solver.trace(state0, lambda_max=120.0, max_spatial_step_rs=0.1)
    last = result.samples[-1]
    print(
        f"status={result.status}, n={len(result.samples)}, "
        f"t_end={last.t:.6f} (Rs/c), r_end={last.r:.6f} (Rs), phi_end={last.phi:.6f} rad"
    )
