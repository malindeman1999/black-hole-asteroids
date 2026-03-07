from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, cos, pi, sin, sqrt
from typing import Callable, List, Literal, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


# SI constants
C = 299_792_458.0
G = 6.67430e-11
SOLAR_MASS_KG = 1.98847e30


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _to_float(x) -> float:
    try:
        return float(x.item())
    except Exception:
        return float(x)


def _get_array_lib(use_gpu: bool):
    if use_gpu and cp is not None:
        return cp
    return np


def _simpson_integral(func: Callable, a: float, b: float, n: int = 384, xp=np) -> float:
    if n < 4:
        n = 4
    if n % 2 == 1:
        n += 1
    x = xp.linspace(a, b, n + 1, dtype=float)
    y = xp.asarray(func(x), dtype=float)
    if y.shape != x.shape:
        y = xp.reshape(y, x.shape)
    if not bool(_to_float(xp.all(xp.isfinite(y)))):
        return float("inf")
    h = (b - a) / n
    total = y[0] + y[-1] + 4.0 * xp.sum(y[1:-1:2]) + 2.0 * xp.sum(y[2:-1:2])
    return _to_float(total * h / 3.0)


def _simpson_integral_batch(func: Callable, a, b, n: int = 256, xp=np):
    if n < 4:
        n = 4
    if n % 2 == 1:
        n += 1
    a = xp.asarray(a, dtype=float)
    b = xp.asarray(b, dtype=float)
    u = xp.linspace(0.0, 1.0, n + 1, dtype=float)
    x = a[:, None] + (b - a)[:, None] * u[None, :]
    y = xp.asarray(func(x), dtype=float)
    if y.shape != x.shape:
        y = xp.reshape(y, x.shape)
    h = (b - a) / n
    total = y[:, 0] + y[:, -1] + 4.0 * xp.sum(y[:, 1:-1:2], axis=1) + 2.0 * xp.sum(y[:, 2:-1:2], axis=1)
    out = total * h / 3.0
    bad = ~xp.all(xp.isfinite(y), axis=1)
    out = xp.where(bad, xp.inf, out)
    return out


@dataclass(frozen=True)
class GeodesicSolution:
    direction: int
    target_azimuth_rad: float
    impact_parameter_m: float
    travel_time_s: float
    branch: str


@dataclass(frozen=True)
class GeodesicResult:
    point_a: Tuple[float, float, float]
    point_b: Tuple[float, float, float]
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    b_before_a: bool
    separation_angle_rad: float
    paths: Tuple[GeodesicSolution, ...]
    lag_between_fastest_two_s: float


@dataclass(frozen=True)
class ObservedRayAtB:
    direction: int
    emission_time_s: float
    arrival_time_s: float
    travel_time_s: float
    impact_parameter_m: float
    gamma_at_b_rad: float


@dataclass(frozen=True)
class EarliestObservedAnglesAtBResult:
    observer_point_b: Tuple[float, float, float]
    observer_time_s: float
    plus: Optional[ObservedRayAtB]
    minus: Optional[ObservedRayAtB]


@dataclass(frozen=True)
class TimelikeTrajectorySample:
    proper_time_s: float
    coordinate_time_s: float
    radius_m: float
    azimuth_rad: float
    position_xyz_m: Tuple[float, float, float]
    radial_speed_local_m_s: float
    tangential_speed_local_m_s: float


@dataclass(frozen=True)
class TimelikeTrajectoryResult:
    initial_position_m: Tuple[float, float, float]
    initial_velocity_m_s: Tuple[float, float, float]
    specific_energy: float
    specific_angular_momentum_m: float
    turning_radii_m: Tuple[float, ...]
    status: Literal["captured", "escaped", "bound", "max_proper_time_reached", "invalid_initial_state"]
    samples: Tuple[TimelikeTrajectorySample, ...]


@dataclass(frozen=True)
class SchwarzschildBlackHole:
    mass_kg: float
    schwarzschild_radius_m: float
    simpson_n_scalar: int = 384
    simpson_n_batch: int = 256
    bisection_iter_scalar: int = 80
    bisection_iter_batch: int = 64
    root_tol: float = 1e-10
    numeric_tol: float = 1e-12

    @classmethod
    def from_diameter_light_seconds(cls, diameter_light_seconds: float = 1.0) -> "SchwarzschildBlackHole":
        rs = 0.5 * diameter_light_seconds * C
        mass = rs * C * C / (2.0 * G)
        return cls(mass_kg=mass, schwarzschild_radius_m=rs)

    def with_quality(self, quality: Literal["high", "medium", "fast"]) -> "SchwarzschildBlackHole":
        presets = {
            "high": dict(
                simpson_n_scalar=384,
                simpson_n_batch=256,
                bisection_iter_scalar=80,
                bisection_iter_batch=64,
                root_tol=1e-10,
                numeric_tol=1e-12,
            ),
            "medium": dict(
                simpson_n_scalar=192,
                simpson_n_batch=128,
                bisection_iter_scalar=48,
                bisection_iter_batch=40,
                root_tol=1e-8,
                numeric_tol=1e-10,
            ),
            "fast": dict(
                simpson_n_scalar=96,
                simpson_n_batch=96,
                bisection_iter_scalar=28,
                bisection_iter_batch=28,
                root_tol=1e-6,
                numeric_tol=1e-8,
            ),
        }
        if quality not in presets:
            raise ValueError("quality must be one of: 'high', 'medium', 'fast'")
        return SchwarzschildBlackHole(
            mass_kg=self.mass_kg,
            schwarzschild_radius_m=self.schwarzschild_radius_m,
            **presets[quality],
        )

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.schwarzschild_radius_m

    @property
    def diameter_light_seconds(self) -> float:
        return self.diameter_m / C

    @property
    def mass_solar(self) -> float:
        return self.mass_kg / SOLAR_MASS_KG

    @property
    def photon_sphere_radius_m(self) -> float:
        return 1.5 * self.schwarzschild_radius_m

    def _effective_potential(self, r: float) -> float:
        rs = self.schwarzschild_radius_m
        return (1.0 - rs / r) / (r * r)

    def _arrival_angle_at_b(self, point_b: Sequence[float], impact_parameter_m: float) -> float:
        bx, by, bz = float(point_b[0]), float(point_b[1]), float(point_b[2])
        r_b = sqrt(bx * bx + by * by + bz * bz)
        rs = self.schwarzschild_radius_m
        if r_b <= rs:
            raise ValueError("Observer point B must be outside the event horizon.")
        s = impact_parameter_m * sqrt(1.0 - rs / r_b) / r_b
        return asin(_clamp(s, -1.0, 1.0))

    def _path_for_direction(self, result: GeodesicResult, direction: int) -> Optional[GeodesicSolution]:
        for path in result.paths:
            if path.direction == direction:
                return path
        return None

    def _timelike_turning_radii(self, specific_energy: float, specific_angular_momentum_m: float) -> Tuple[float, ...]:
        # Turning points satisfy eps^2 - (1-rs/r)(1+l^2/r^2) = 0.
        # With y = 1/r this is a cubic:
        #   rs*l^2*y^3 - l^2*y^2 + rs*y + (eps^2 - 1) = 0.
        rs = self.schwarzschild_radius_m
        l2 = specific_angular_momentum_m * specific_angular_momentum_m
        c0 = specific_energy * specific_energy - 1.0
        if l2 <= 0.0:
            if abs(rs) <= self.numeric_tol:
                return tuple()
            y = -c0 / rs
            if y > self.numeric_tol:
                return (1.0 / y,)
            return tuple()
        coeffs = np.asarray([rs * l2, -l2, rs, c0], dtype=float)
        roots = np.roots(coeffs)
        radii: List[float] = []
        for root in roots:
            if abs(float(np.imag(root))) > 1e-9:
                continue
            y = float(np.real(root))
            if y <= self.numeric_tol:
                continue
            r = 1.0 / y
            if np.isfinite(r) and r > self.schwarzschild_radius_m * (1.0 + 1e-9):
                radii.append(float(r))
        radii.sort()
        dedup: List[float] = []
        for r in radii:
            if (not dedup) or abs(r - dedup[-1]) > 1e-7 * max(1.0, abs(r)):
                dedup.append(r)
        return tuple(dedup)

    def integrate_timelike_trajectory(
        self,
        initial_position_m: Sequence[float],
        initial_velocity_m_s: Sequence[float],
        proper_time_step_s: float = 1e-4,
        max_proper_time_s: float = 10.0,
        max_steps: int = 200_000,
        escape_radius_m: Optional[float] = None,
        integrator: Literal["symplectic", "rk4", "euler"] = "symplectic",
    ) -> TimelikeTrajectoryResult:
        """
        Propagate a massive body geodesic from an initial state in Schwarzschild spacetime.

        The initial velocity is interpreted as the local physical 3-velocity measured by a
        static observer at the initial position. Integration is carried out in proper time.
        """
        x0 = np.asarray(initial_position_m, dtype=float)
        v0 = np.asarray(initial_velocity_m_s, dtype=float)
        if x0.shape != (3,) or v0.shape != (3,):
            raise ValueError("initial_position_m and initial_velocity_m_s must be 3-vectors.")
        rs = self.schwarzschild_radius_m
        r0 = float(np.linalg.norm(x0))
        if r0 <= rs * (1.0 + 1e-12):
            return TimelikeTrajectoryResult(
                initial_position_m=(float(x0[0]), float(x0[1]), float(x0[2])),
                initial_velocity_m_s=(float(v0[0]), float(v0[1]), float(v0[2])),
                specific_energy=float("nan"),
                specific_angular_momentum_m=float("nan"),
                turning_radii_m=tuple(),
                status="invalid_initial_state",
                samples=tuple(),
            )

        v_mag = float(np.linalg.norm(v0))
        if v_mag >= C * (1.0 - 1e-15):
            return TimelikeTrajectoryResult(
                initial_position_m=(float(x0[0]), float(x0[1]), float(x0[2])),
                initial_velocity_m_s=(float(v0[0]), float(v0[1]), float(v0[2])),
                specific_energy=float("nan"),
                specific_angular_momentum_m=float("nan"),
                turning_radii_m=tuple(),
                status="invalid_initial_state",
                samples=tuple(),
            )

        if proper_time_step_s <= 0.0:
            raise ValueError("proper_time_step_s must be > 0.")
        if max_proper_time_s <= 0.0:
            raise ValueError("max_proper_time_s must be > 0.")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if integrator not in ("symplectic", "rk4", "euler"):
            raise ValueError("integrator must be one of: 'symplectic', 'rk4', 'euler'.")

        rhat = x0 / r0
        vr0 = float(np.dot(v0, rhat))
        vtan_vec = v0 - vr0 * rhat
        vtan0 = float(np.linalg.norm(vtan_vec))
        g0 = 1.0 - rs / r0
        gamma = 1.0 / sqrt(max(1e-30, 1.0 - (v_mag * v_mag) / (C * C)))
        specific_energy = gamma * sqrt(g0)
        specific_angular_momentum_m = gamma * r0 * vtan0 / C
        turning_radii = self._timelike_turning_radii(specific_energy, specific_angular_momentum_m)

        # The geodesic stays in a fixed plane: span{r_hat0, tangential_hat0}.
        if vtan0 > 1e-15:
            e1 = rhat
            e2 = vtan_vec / vtan0
        else:
            trial = np.asarray([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(trial, rhat))) > 0.95:
                trial = np.asarray([0.0, 1.0, 0.0], dtype=float)
            e1 = rhat
            e2 = trial - float(np.dot(trial, rhat)) * rhat
            e2 = e2 / float(np.linalg.norm(e2))

        if escape_radius_m is None:
            escape_radius_m = 100.0 * rs
        escape_radius_m = max(float(escape_radius_m), r0)

        tau = 0.0
        t_coord = 0.0
        r = r0
        phi = 0.0
        pr = gamma * sqrt(g0) * vr0
        l2 = specific_angular_momentum_m * specific_angular_momentum_m
        status: Literal["captured", "escaped", "bound", "max_proper_time_reached", "invalid_initial_state"] = (
            "max_proper_time_reached"
        )

        def _dV_dr(radius: float) -> float:
            g = 1.0 - rs / radius
            gp = rs / (radius * radius)
            h = 1.0 + l2 / (radius * radius)
            hp = -2.0 * l2 / (radius * radius * radius)
            return gp * h + g * hp

        def _derivs(radius: float, phi_rad: float, t_s: float, pr_m_s: float):
            del phi_rad, t_s  # derivatives do not depend explicitly on these coordinates
            g = 1.0 - rs / radius
            if g <= 0.0:
                return 0.0, 0.0, float("inf"), 0.0
            dr_dtau = pr_m_s
            dphi_dtau = C * specific_angular_momentum_m / (radius * radius)
            dt_dtau = specific_energy / g
            dpr_dtau = -0.5 * C * C * _dV_dr(radius)
            return dr_dtau, dphi_dtau, dt_dtau, dpr_dtau

        samples: List[TimelikeTrajectorySample] = []
        n_steps = min(max_steps, int(max_proper_time_s / proper_time_step_s) + 2)
        for _ in range(n_steps):
            if r <= rs * (1.0 + 1e-9):
                status = "captured"
                break
            if r >= escape_radius_m:
                status = "escaped"
                break

            g = 1.0 - rs / r
            if g <= 0.0:
                status = "captured"
                break
            dr_dtau = pr
            dphi_dtau = C * specific_angular_momentum_m / (r * r)
            dt_dtau = specific_energy / g

            pos = r * (cos(phi) * e1 + sin(phi) * e2)
            gamma_local = specific_energy / sqrt(g)
            radial_speed_local = dr_dtau / (gamma_local * sqrt(g)) if gamma_local > 0.0 else 0.0
            tangential_speed_local = (r * dphi_dtau / gamma_local) if gamma_local > 0.0 else 0.0
            samples.append(
                TimelikeTrajectorySample(
                    proper_time_s=float(tau),
                    coordinate_time_s=float(t_coord),
                    radius_m=float(r),
                    azimuth_rad=float(phi),
                    position_xyz_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                    radial_speed_local_m_s=float(radial_speed_local),
                    tangential_speed_local_m_s=float(tangential_speed_local),
                )
            )

            dtau = min(float(proper_time_step_s), float(max_proper_time_s - tau))
            if dtau <= 0.0:
                break

            if integrator == "euler":
                dr1, dphi1, dt1, dpr1 = _derivs(r, phi, t_coord, pr)
                r += dtau * dr1
                phi += dtau * dphi1
                t_coord += dtau * dt1
                pr += dtau * dpr1
            elif integrator == "rk4":
                k1 = _derivs(r, phi, t_coord, pr)
                k2 = _derivs(
                    r + 0.5 * dtau * k1[0],
                    phi + 0.5 * dtau * k1[1],
                    t_coord + 0.5 * dtau * k1[2],
                    pr + 0.5 * dtau * k1[3],
                )
                k3 = _derivs(
                    r + 0.5 * dtau * k2[0],
                    phi + 0.5 * dtau * k2[1],
                    t_coord + 0.5 * dtau * k2[2],
                    pr + 0.5 * dtau * k2[3],
                )
                k4 = _derivs(
                    r + dtau * k3[0],
                    phi + dtau * k3[1],
                    t_coord + dtau * k3[2],
                    pr + dtau * k3[3],
                )
                r += (dtau / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
                phi += (dtau / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
                t_coord += (dtau / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
                pr += (dtau / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            else:
                # Symplectic Euler (kick-drift): robust long-time orbital behavior.
                _, _, _, dpr1 = _derivs(r, phi, t_coord, pr)
                pr_next = pr + dtau * dpr1
                r_next = r + dtau * pr_next
                r_mid = max(0.5 * (r + r_next), rs * (1.0 + 1e-12))
                phi += dtau * (C * specific_angular_momentum_m / (r_mid * r_mid))
                g_mid = 1.0 - rs / r_mid
                if g_mid <= 0.0:
                    status = "captured"
                    break
                t_coord += dtau * (specific_energy / g_mid)
                r = r_next
                pr = pr_next

            r = max(r, rs * (1.0 + 1e-12))
            tau += dtau

        if status == "max_proper_time_reached" and len(turning_radii) >= 2:
            status = "bound"

        return TimelikeTrajectoryResult(
            initial_position_m=(float(x0[0]), float(x0[1]), float(x0[2])),
            initial_velocity_m_s=(float(v0[0]), float(v0[1]), float(v0[2])),
            specific_energy=float(specific_energy),
            specific_angular_momentum_m=float(specific_angular_momentum_m),
            turning_radii_m=tuple(float(rp) for rp in turning_radii),
            status=status,
            samples=tuple(samples),
        )

    def _delta_phi_mono(self, r1: float, r2: float, b: float, use_gpu: bool = True) -> float:
        rs = self.schwarzschild_radius_m
        inv_b2 = 1.0 / (b * b)
        tol = self.numeric_tol
        xp = _get_array_lib(use_gpu)

        def integrand(r):
            f = inv_b2 - (1.0 - rs / r) / (r * r)
            f = xp.where((f < 0.0) & (f > -tol), 0.0, f)
            out = xp.full_like(r, xp.inf, dtype=float)
            mask = f > 0.0
            out[mask] = 1.0 / (r[mask] * r[mask] * xp.sqrt(f[mask]))
            out[f == 0.0] = 0.0
            return out

        a, c = (r1, r2) if r1 < r2 else (r2, r1)
        return _simpson_integral(integrand, a, c, n=self.simpson_n_scalar, xp=xp)

    def _time_mono(self, r1: float, r2: float, b: float, use_gpu: bool = True) -> float:
        rs = self.schwarzschild_radius_m
        b2 = b * b
        tol = self.numeric_tol
        xp = _get_array_lib(use_gpu)

        def integrand(r):
            w = 1.0 - b2 * (1.0 - rs / r) / (r * r)
            w = xp.where((w < 0.0) & (w > -tol), 0.0, w)
            g = 1.0 - rs / r
            out = xp.full_like(r, xp.inf, dtype=float)
            mask = (w > 0.0) & (g > 0.0)
            out[mask] = 1.0 / (C * g[mask] * xp.sqrt(w[mask]))
            out[(w == 0.0) & (g > 0.0)] = 0.0
            return out

        a, c = (r1, r2) if r1 < r2 else (r2, r1)
        return _simpson_integral(integrand, a, c, n=self.simpson_n_scalar, xp=xp)

    def _leg_integrals_from_turning_radius(self, r: float, rp: float, use_gpu: bool = True) -> Tuple[float, float]:
        rs = self.schwarzschild_radius_m
        b2 = rp * rp / (1.0 - rs / rp)
        b = sqrt(b2)
        xp = _get_array_lib(use_gpu)

        if abs(r - rp) < 1e-9:
            return 0.0, 0.0

        # Substitution r(u) = rp + (r-rp) u^2 avoids endpoint sqrt singularity.
        dr = r - rp

        def phi_integrand(u):
            ru = rp + dr * u * u
            f = 1.0 / b2 - self._effective_potential(ru)
            f = xp.where((f < 0.0) & (f > -self.numeric_tol), 0.0, f)
            d_r_du = 2.0 * dr * u
            out = xp.zeros_like(u, dtype=float)
            bad = f < 0.0
            out[bad] = xp.inf
            good = (f > 0.0) & (~bad)
            out[good] = d_r_du[good] / (ru[good] * ru[good] * xp.sqrt(f[good]))
            return out

        def time_integrand(u):
            ru = rp + dr * u * u
            g = 1.0 - rs / ru
            w = 1.0 - b2 * (1.0 - rs / ru) / (ru * ru)
            w = xp.where((w < 0.0) & (w > -self.numeric_tol), 0.0, w)
            d_r_du = 2.0 * dr * u
            out = xp.zeros_like(u, dtype=float)
            bad = (g <= 0.0) | (w < 0.0)
            out[bad] = xp.inf
            good = (w > 0.0) & (~bad)
            out[good] = d_r_du[good] / (C * g[good] * xp.sqrt(w[good]))
            return out

        dphi = _simpson_integral(phi_integrand, 0.0, 1.0, n=self.simpson_n_scalar, xp=xp)
        dt = _simpson_integral(time_integrand, 0.0, 1.0, n=self.simpson_n_scalar, xp=xp)
        return abs(dphi), abs(dt)

    def _delta_phi_turning(self, r1: float, r2: float, rp: float, use_gpu: bool = True) -> Tuple[float, float, float]:
        b = sqrt(rp * rp / (1.0 - self.schwarzschild_radius_m / rp))
        dphi1, dt1 = self._leg_integrals_from_turning_radius(r1, rp, use_gpu=use_gpu)
        dphi2, dt2 = self._leg_integrals_from_turning_radius(r2, rp, use_gpu=use_gpu)
        return dphi1 + dphi2, dt1 + dt2, b

    def _find_root_bisection(self, func: Callable[[float], float], lo: float, hi: float, max_iter: int | None = None) -> float:
        if max_iter is None:
            max_iter = self.bisection_iter_scalar
        flo = func(lo)
        fhi = func(hi)
        if not (flo == flo and fhi == fhi):  # NaN check
            raise ValueError("Invalid root bracket (NaN).")
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi > 0.0:
            raise ValueError("Root is not bracketed.")
        a, b = lo, hi
        fa, fb = flo, fhi
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = func(m)
            if abs(fm) < self.root_tol:
                return m
            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    def _solve_for_target_azimuth(self, r1: float, r2: float, target: float, use_gpu: bool = True) -> List[Tuple[float, float, str]]:
        rs = self.schwarzschild_radius_m
        rmin = min(r1, r2)
        rmax = max(r1, r2)
        solutions: List[Tuple[float, float, str]] = []

        # Monotonic branch
        if rmin > rs:
            candidates = [self._effective_potential(rmin), self._effective_potential(rmax)]
            if rmin <= self.photon_sphere_radius_m <= rmax:
                candidates.append(self._effective_potential(self.photon_sphere_radius_m))
            vmax = max(candidates)
            b_hi = 0.999999 / sqrt(vmax)
            b_lo = self.numeric_tol * rs

            try:
                phi_hi = self._delta_phi_mono(r1, r2, b_hi, use_gpu=use_gpu)
                if 0.0 <= target <= phi_hi:
                    f = lambda b: self._delta_phi_mono(r1, r2, b, use_gpu=use_gpu) - target
                    b_star = self._find_root_bisection(f, b_lo, b_hi)
                    t_star = self._time_mono(r1, r2, b_star, use_gpu=use_gpu)
                    solutions.append((b_star, t_star, "monotonic"))
            except Exception:
                pass

        # Turning branch
        rph = self.photon_sphere_radius_m
        if rmin > rph * (1.0 + 1e-6):
            rp_hi = rmin * (1.0 - 1e-8)
            rp_lo = rph * (1.0 + 1e-8)

            try:
                phi_min, _, _ = self._delta_phi_turning(r1, r2, rp_hi, use_gpu=use_gpu)
                phi_max, _, _ = self._delta_phi_turning(r1, r2, rp_lo, use_gpu=use_gpu)
                if phi_min <= target <= phi_max:
                    f = lambda rp: self._delta_phi_turning(r1, r2, rp, use_gpu=use_gpu)[0] - target
                    rp_star = self._find_root_bisection(f, rp_hi, rp_lo)
                    _, t_star, b_star = self._delta_phi_turning(r1, r2, rp_star, use_gpu=use_gpu)
                    solutions.append((b_star, t_star, "turning"))
            except Exception:
                pass

        # Deduplicate near-identical b values
        dedup: List[Tuple[float, float, str]] = []
        for b, t, branch in sorted(solutions, key=lambda x: x[1]):
            if not dedup or abs((b - dedup[-1][0]) / max(1.0, abs(b))) > 1e-6:
                dedup.append((b, t, branch))
        return dedup

    def _bisection_batch(self, func: Callable, lo, hi, valid, xp, max_iter: int | None = None):
        if max_iter is None:
            max_iter = self.bisection_iter_batch
        lo = xp.asarray(lo, dtype=float)
        hi = xp.asarray(hi, dtype=float)
        valid = xp.asarray(valid, dtype=bool)
        flo = xp.asarray(func(lo), dtype=float)
        fhi = xp.asarray(func(hi), dtype=float)
        bracket = valid & xp.isfinite(flo) & xp.isfinite(fhi) & (flo * fhi <= 0.0)
        a = lo.copy()
        b = hi.copy()
        fa = flo.copy()
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = xp.asarray(func(m), dtype=float)
            use_left = bracket & (fa * fm <= 0.0)
            a = xp.where(use_left, a, m)
            b = xp.where(use_left, m, b)
            fa = xp.where(use_left, fa, fm)
        return 0.5 * (a + b), bracket

    def _delta_phi_mono_batch(self, r1, r2, b, xp):
        rs = self.schwarzschild_radius_m
        r1 = xp.asarray(r1, dtype=float)
        r2 = xp.asarray(r2, dtype=float)
        b = xp.asarray(b, dtype=float)
        a = xp.minimum(r1, r2)
        c = xp.maximum(r1, r2)
        inv_b2 = 1.0 / (b * b)
        tol = self.numeric_tol

        def integrand(r):
            f = inv_b2[:, None] - (1.0 - rs / r) / (r * r)
            f = xp.where((f < 0.0) & (f > -tol), 0.0, f)
            out = xp.full_like(r, xp.inf, dtype=float)
            pos = f > 0.0
            out[pos] = 1.0 / (r[pos] * r[pos] * xp.sqrt(f[pos]))
            out[f == 0.0] = 0.0
            return out

        return _simpson_integral_batch(integrand, a, c, n=self.simpson_n_batch, xp=xp)

    def _time_mono_batch(self, r1, r2, b, xp):
        rs = self.schwarzschild_radius_m
        r1 = xp.asarray(r1, dtype=float)
        r2 = xp.asarray(r2, dtype=float)
        b = xp.asarray(b, dtype=float)
        a = xp.minimum(r1, r2)
        c = xp.maximum(r1, r2)
        b2 = b * b
        tol = self.numeric_tol

        def integrand(r):
            w = 1.0 - b2[:, None] * (1.0 - rs / r) / (r * r)
            w = xp.where((w < 0.0) & (w > -tol), 0.0, w)
            g = 1.0 - rs / r
            out = xp.full_like(r, xp.inf, dtype=float)
            mask = (w > 0.0) & (g > 0.0)
            out[mask] = 1.0 / (C * g[mask] * xp.sqrt(w[mask]))
            out[(w == 0.0) & (g > 0.0)] = 0.0
            return out

        return _simpson_integral_batch(integrand, a, c, n=self.simpson_n_batch, xp=xp)

    def _leg_integrals_turning_batch(self, r, rp, xp):
        rs = self.schwarzschild_radius_m
        r = xp.asarray(r, dtype=float)
        rp = xp.asarray(rp, dtype=float)
        dr = r - rp
        b2 = rp * rp / (1.0 - rs / rp)

        # Integrate on u in [0,1] with ru = rp + dr*u^2
        def phi_integrand(u):
            ru = rp[:, None] + dr[:, None] * u * u
            f = 1.0 / b2[:, None] - (1.0 - rs / ru) / (ru * ru)
            f = xp.where((f < 0.0) & (f > -self.numeric_tol), 0.0, f)
            d_r_du = 2.0 * dr[:, None] * u
            out = xp.zeros_like(u, dtype=float)
            bad = f < 0.0
            out[bad] = xp.inf
            good = (f > 0.0) & (~bad)
            out[good] = d_r_du[good] / (ru[good] * ru[good] * xp.sqrt(f[good]))
            return out

        def time_integrand(u):
            ru = rp[:, None] + dr[:, None] * u * u
            g = 1.0 - rs / ru
            w = 1.0 - b2[:, None] * (1.0 - rs / ru) / (ru * ru)
            w = xp.where((w < 0.0) & (w > -self.numeric_tol), 0.0, w)
            d_r_du = 2.0 * dr[:, None] * u
            out = xp.zeros_like(u, dtype=float)
            bad = (g <= 0.0) | (w < 0.0)
            out[bad] = xp.inf
            good = (w > 0.0) & (~bad)
            out[good] = d_r_du[good] / (C * g[good] * xp.sqrt(w[good]))
            return out

        z = xp.zeros_like(r)
        dphi = xp.where(
            xp.abs(dr) < 1e-9,
            z,
            xp.abs(_simpson_integral_batch(phi_integrand, z, z + 1.0, n=self.simpson_n_batch, xp=xp)),
        )
        dt = xp.where(
            xp.abs(dr) < 1e-9,
            z,
            xp.abs(_simpson_integral_batch(time_integrand, z, z + 1.0, n=self.simpson_n_batch, xp=xp)),
        )
        return dphi, dt

    def _delta_phi_turning_batch(self, r1, r2, rp, xp):
        rs = self.schwarzschild_radius_m
        rp = xp.asarray(rp, dtype=float)
        b = xp.sqrt(rp * rp / (1.0 - rs / rp))
        dphi1, dt1 = self._leg_integrals_turning_batch(r1, rp, xp)
        dphi2, dt2 = self._leg_integrals_turning_batch(r2, rp, xp)
        return dphi1 + dphi2, dt1 + dt2, b

    def _solve_target_batch_gpu(self, r1, r2, target, xp):
        rs = self.schwarzschild_radius_m
        rph = self.photon_sphere_radius_m
        r1 = xp.asarray(r1, dtype=float)
        r2 = xp.asarray(r2, dtype=float)
        target = xp.asarray(target, dtype=float)
        rmin = xp.minimum(r1, r2)
        rmax = xp.maximum(r1, r2)
        n = r1.shape[0]

        def v_of(r):
            return (1.0 - rs / r) / (r * r)

        v1 = v_of(rmin)
        v2 = v_of(rmax)
        vph = xp.full_like(v1, v_of(rph))
        cross = (rmin <= rph) & (rph <= rmax)
        vmax = xp.where(cross, xp.maximum(xp.maximum(v1, v2), vph), xp.maximum(v1, v2))

        b_hi = 0.999999 / xp.sqrt(vmax)
        b_lo = xp.full_like(b_hi, self.numeric_tol * rs)

        phi_hi = self._delta_phi_mono_batch(r1, r2, b_hi, xp)
        mono_valid = (rmin > rs) & xp.isfinite(phi_hi) & (target >= 0.0) & (target <= phi_hi)

        def f_mono(b):
            return self._delta_phi_mono_batch(r1, r2, b, xp) - target

        b_star, b_ok = self._bisection_batch(f_mono, b_lo, b_hi, mono_valid, xp)
        t_mono = self._time_mono_batch(r1, r2, b_star, xp)
        mono_ok = b_ok & xp.isfinite(t_mono)

        rp_hi = rmin * (1.0 - 1e-8)
        rp_lo = xp.full_like(rmin, rph * (1.0 + 1e-8))
        base_turn_valid = rmin > rph * (1.0 + 1e-6)
        phi_min, _, _ = self._delta_phi_turning_batch(r1, r2, rp_hi, xp)
        phi_max, _, _ = self._delta_phi_turning_batch(r1, r2, rp_lo, xp)
        turn_valid = base_turn_valid & xp.isfinite(phi_min) & xp.isfinite(phi_max) & (target >= phi_min) & (target <= phi_max)

        def f_turn(rp):
            return self._delta_phi_turning_batch(r1, r2, rp, xp)[0] - target

        rp_star, rp_ok = self._bisection_batch(f_turn, rp_hi, rp_lo, turn_valid, xp)
        _, t_turn, b_turn = self._delta_phi_turning_batch(r1, r2, rp_star, xp)
        turn_ok = rp_ok & xp.isfinite(t_turn) & xp.isfinite(b_turn)

        return (
            xp.asnumpy(b_star),
            xp.asnumpy(t_mono),
            xp.asnumpy(mono_ok),
            xp.asnumpy(b_turn),
            xp.asnumpy(t_turn),
            xp.asnumpy(turn_ok),
        )

    def find_two_shortest_geodesics(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        a_before_b: bool = True,
        use_gpu: bool = True,
    ) -> GeodesicResult:
        ax, ay, az = float(point_a[0]), float(point_a[1]), float(point_a[2])
        bx, by, bz = float(point_b[0]), float(point_b[1]), float(point_b[2])

        # Default ordering: A happens before B (propagate from A to B).
        if a_before_b:
            sx, sy, sz = ax, ay, az
            ex, ey, ez = bx, by, bz
            b_before_a = False
        else:
            sx, sy, sz = bx, by, bz
            ex, ey, ez = ax, ay, az
            b_before_a = True

        r1 = sqrt(sx * sx + sy * sy + sz * sz)
        r2 = sqrt(ex * ex + ey * ey + ez * ez)
        rs = self.schwarzschild_radius_m
        if r1 <= rs or r2 <= rs:
            raise ValueError("Points must be outside the event horizon.")

        dot = sx * ex + sy * ey + sz * ez
        gamma = acos(_clamp(dot / (r1 * r2), -1.0, 1.0))

        # Two geometrically shortest angular options in the center-A-B plane.
        angular_targets = [(+1, gamma), (-1, 2.0 * pi - gamma)]
        candidates: List[GeodesicSolution] = []

        for direction, target in angular_targets:
            if target <= 1e-12:
                continue
            for b, travel_time, branch in self._solve_for_target_azimuth(r1, r2, target, use_gpu=use_gpu):
                candidates.append(
                    GeodesicSolution(
                        direction=direction,
                        target_azimuth_rad=target,
                        impact_parameter_m=b,
                        travel_time_s=travel_time,
                        branch=branch,
                    )
                )

        if not candidates:
            raise RuntimeError("No geodesic solution found for the requested points.")

        # Prefer one shortest path per side (direction) when both sides exist.
        best_by_direction = {}
        for c in sorted(candidates, key=lambda s: s.travel_time_s):
            if c.direction not in best_by_direction:
                best_by_direction[c.direction] = c
        if (+1 in best_by_direction) and (-1 in best_by_direction):
            top2 = tuple(sorted([best_by_direction[+1], best_by_direction[-1]], key=lambda s: s.travel_time_s))
        else:
            top2 = tuple(sorted(candidates, key=lambda s: s.travel_time_s)[:2])
        lag = 0.0
        if len(top2) >= 2:
            lag = top2[1].travel_time_s - top2[0].travel_time_s

        return GeodesicResult(
            point_a=(ax, ay, az),
            point_b=(bx, by, bz),
            start_point=(sx, sy, sz),
            end_point=(ex, ey, ez),
            b_before_a=b_before_a,
            separation_angle_rad=gamma,
            paths=top2,
            lag_between_fastest_two_s=lag,
        )

    def find_two_shortest_geodesics_batch(
        self,
        point_pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
        a_before_b: bool = True,
        max_workers: int | None = None,
        backend: Literal["process", "thread", "serial"] = "process",
        use_gpu: bool = True,
    ) -> List[GeodesicResult]:
        """
        Compute geodesics for many (A, B) point pairs.

        Returns results in the same order as the input sequence.
        """
        pairs = list(point_pairs)
        if not pairs:
            return []
        if use_gpu and cp is not None:
            return self._find_two_shortest_geodesics_batch_gpu(pairs, a_before_b=a_before_b)
        if len(pairs) == 1 or backend == "serial":
            return [self.find_two_shortest_geodesics(a, b, a_before_b=a_before_b, use_gpu=use_gpu) for a, b in pairs]

        tasks = [(self, pair[0], pair[1], a_before_b, use_gpu) for pair in pairs]
        if backend == "thread":
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_solve_pair_worker, tasks))
        if backend == "process":
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_solve_pair_worker, tasks))
        raise ValueError("backend must be one of: 'process', 'thread', 'serial'")

    def _find_two_shortest_geodesics_batch_gpu(
        self,
        point_pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
        a_before_b: bool = True,
    ) -> List[GeodesicResult]:
        xp = cp
        n = len(point_pairs)
        ax = np.asarray([float(p[0][0]) for p in point_pairs], dtype=float)
        ay = np.asarray([float(p[0][1]) for p in point_pairs], dtype=float)
        az = np.asarray([float(p[0][2]) for p in point_pairs], dtype=float)
        bx = np.asarray([float(p[1][0]) for p in point_pairs], dtype=float)
        by = np.asarray([float(p[1][1]) for p in point_pairs], dtype=float)
        bz = np.asarray([float(p[1][2]) for p in point_pairs], dtype=float)

        if a_before_b:
            sx, sy, sz = ax, ay, az
            ex, ey, ez = bx, by, bz
            b_before_a = False
        else:
            sx, sy, sz = bx, by, bz
            ex, ey, ez = ax, ay, az
            b_before_a = True

        r1 = np.sqrt(sx * sx + sy * sy + sz * sz)
        r2 = np.sqrt(ex * ex + ey * ey + ez * ez)
        rs = self.schwarzschild_radius_m
        if np.any(r1 <= rs) or np.any(r2 <= rs):
            raise ValueError("Points must be outside the event horizon.")

        dot = sx * ex + sy * ey + sz * ez
        gamma = np.arccos(np.clip(dot / (r1 * r2), -1.0, 1.0))
        target_short = gamma
        target_long = 2.0 * np.pi - gamma

        b_m1, t_m1, ok_m1, b_t1, t_t1, ok_t1 = self._solve_target_batch_gpu(r1, r2, target_short, xp)
        b_m2, t_m2, ok_m2, b_t2, t_t2, ok_t2 = self._solve_target_batch_gpu(r1, r2, target_long, xp)

        results: List[GeodesicResult] = []
        for i in range(n):
            candidates: List[GeodesicSolution] = []
            if target_short[i] > 1e-12:
                if ok_m1[i]:
                    candidates.append(
                        GeodesicSolution(
                            direction=+1,
                            target_azimuth_rad=float(target_short[i]),
                            impact_parameter_m=float(b_m1[i]),
                            travel_time_s=float(t_m1[i]),
                            branch="monotonic",
                        )
                    )
                if ok_t1[i]:
                    candidates.append(
                        GeodesicSolution(
                            direction=+1,
                            target_azimuth_rad=float(target_short[i]),
                            impact_parameter_m=float(b_t1[i]),
                            travel_time_s=float(t_t1[i]),
                            branch="turning",
                        )
                    )
            if target_long[i] > 1e-12:
                if ok_m2[i]:
                    candidates.append(
                        GeodesicSolution(
                            direction=-1,
                            target_azimuth_rad=float(target_long[i]),
                            impact_parameter_m=float(b_m2[i]),
                            travel_time_s=float(t_m2[i]),
                            branch="monotonic",
                        )
                    )
                if ok_t2[i]:
                    candidates.append(
                        GeodesicSolution(
                            direction=-1,
                            target_azimuth_rad=float(target_long[i]),
                            impact_parameter_m=float(b_t2[i]),
                            travel_time_s=float(t_t2[i]),
                            branch="turning",
                        )
                    )

            # Deduplicate near-identical paths.
            dedup: List[GeodesicSolution] = []
            for c in sorted(candidates, key=lambda s: s.travel_time_s):
                if not dedup:
                    dedup.append(c)
                    continue
                prev = dedup[-1]
                rel_b = abs((c.impact_parameter_m - prev.impact_parameter_m) / max(1.0, abs(c.impact_parameter_m)))
                if rel_b > 1e-6 or c.direction != prev.direction:
                    dedup.append(c)

            if not dedup:
                raise RuntimeError("No geodesic solution found for the requested points.")

            # Prefer one shortest path per side (direction) when both sides exist.
            best_by_direction = {}
            for c in sorted(dedup, key=lambda s: s.travel_time_s):
                if c.direction not in best_by_direction:
                    best_by_direction[c.direction] = c
            if (+1 in best_by_direction) and (-1 in best_by_direction):
                top2 = tuple(sorted([best_by_direction[+1], best_by_direction[-1]], key=lambda s: s.travel_time_s))
            else:
                top2 = tuple(sorted(dedup, key=lambda s: s.travel_time_s)[:2])
            lag = 0.0
            if len(top2) >= 2:
                lag = top2[1].travel_time_s - top2[0].travel_time_s

            results.append(
                GeodesicResult(
                    point_a=(float(ax[i]), float(ay[i]), float(az[i])),
                    point_b=(float(bx[i]), float(by[i]), float(bz[i])),
                    start_point=(float(sx[i]), float(sy[i]), float(sz[i])),
                    end_point=(float(ex[i]), float(ey[i]), float(ez[i])),
                    b_before_a=b_before_a,
                    separation_angle_rad=float(gamma[i]),
                    paths=top2,
                    lag_between_fastest_two_s=float(lag),
                )
            )

        return results

    def find_earliest_observed_angles_at_b(
        self,
        trajectory: Callable[[float], Sequence[float]],
        point_b: Sequence[float],
        t0: float,
        tmin: float,
        tmax: float,
        scan_samples: int = 129,
        root_max_iter: int = 36,
        root_tol_time: float = 1e-6,
        use_gpu: bool = True,
    ) -> EarliestObservedAnglesAtBResult:
        """
        Find earliest emission times on a trajectory that arrive at B at observer time t0.

        Solves, for each side direction d in {+1, -1}:
            te + T_d(x(te), B) = t0
        and returns the earliest root in [tmin, min(tmax, t0)].
        """
        if scan_samples < 3:
            scan_samples = 3
        t_hi = min(float(tmax), float(t0))
        t_lo = float(tmin)
        if t_hi < t_lo:
            raise ValueError("Invalid time window: min(tmax, t0) must be >= tmin.")

        b_fixed = (float(point_b[0]), float(point_b[1]), float(point_b[2]))

        def _eval_many(ts: Sequence[float]):
            pairs = [(trajectory(float(t)), b_fixed) for t in ts]
            results = self.find_two_shortest_geodesics_batch(pairs, a_before_b=True, use_gpu=use_gpu)
            plus = []
            minus = []
            for tt, rr in zip(ts, results):
                p_plus = self._path_for_direction(rr, +1)
                p_minus = self._path_for_direction(rr, -1)
                plus.append(float(tt) + p_plus.travel_time_s - t0 if p_plus is not None else float("inf"))
                minus.append(float(tt) + p_minus.travel_time_s - t0 if p_minus is not None else float("inf"))
            return plus, minus, results

        ts = np.linspace(t_lo, t_hi, scan_samples, dtype=float).tolist()
        f_plus, f_minus, _ = _eval_many(ts)

        def _find_first_bracket(fvals: Sequence[float]) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
            for i in range(len(ts)):
                fi = fvals[i]
                if np.isfinite(fi) and abs(fi) <= 1e-15:
                    return None, ts[i]
            for i in range(len(ts) - 1):
                fa = fvals[i]
                fb = fvals[i + 1]
                if (not np.isfinite(fa)) or (not np.isfinite(fb)):
                    continue
                if fa == 0.0:
                    return None, ts[i]
                if fa * fb <= 0.0:
                    return (ts[i], ts[i + 1]), None
            return None, None

        def _f_dir_at_time(te: float, direction: int) -> float:
            f_p, f_m, _ = _eval_many([te])
            return f_p[0] if direction == +1 else f_m[0]

        def _solve_root(bracket: Tuple[float, float], direction: int) -> float:
            lo, hi = bracket
            flo = _f_dir_at_time(lo, direction)
            fhi = _f_dir_at_time(hi, direction)
            if abs(flo) <= 1e-15:
                return lo
            if abs(fhi) <= 1e-15:
                return hi
            a, b = lo, hi
            fa, fb = flo, fhi
            for _ in range(root_max_iter):
                m = 0.5 * (a + b)
                fm = _f_dir_at_time(m, direction)
                if abs(fm) <= 1e-12 or abs(b - a) <= root_tol_time:
                    return m
                if fa * fm <= 0.0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return 0.5 * (a + b)

        def _build_observed(te: float, direction: int) -> Optional[ObservedRayAtB]:
            _, _, rr = _eval_many([te])
            path = self._path_for_direction(rr[0], direction)
            if path is None:
                return None
            gamma_b = self._arrival_angle_at_b(b_fixed, path.impact_parameter_m)
            return ObservedRayAtB(
                direction=direction,
                emission_time_s=float(te),
                arrival_time_s=float(t0),
                travel_time_s=float(path.travel_time_s),
                impact_parameter_m=float(path.impact_parameter_m),
                gamma_at_b_rad=float(gamma_b),
            )

        plus_bracket, plus_exact = _find_first_bracket(f_plus)
        minus_bracket, minus_exact = _find_first_bracket(f_minus)

        plus_obs: Optional[ObservedRayAtB] = None
        minus_obs: Optional[ObservedRayAtB] = None

        if plus_exact is not None:
            plus_obs = _build_observed(plus_exact, +1)
        elif plus_bracket is not None:
            plus_obs = _build_observed(_solve_root(plus_bracket, +1), +1)

        if minus_exact is not None:
            minus_obs = _build_observed(minus_exact, -1)
        elif minus_bracket is not None:
            minus_obs = _build_observed(_solve_root(minus_bracket, -1), -1)

        return EarliestObservedAnglesAtBResult(
            observer_point_b=b_fixed,
            observer_time_s=float(t0),
            plus=plus_obs,
            minus=minus_obs,
        )


def _solve_pair_worker(task: Tuple[SchwarzschildBlackHole, Sequence[float], Sequence[float], bool, bool]) -> GeodesicResult:
    bh, point_a, point_b, a_before_b, use_gpu = task
    return bh.find_two_shortest_geodesics(point_a, point_b, a_before_b=a_before_b, use_gpu=use_gpu)


if __name__ == "__main__":
    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0)
    print(f"Schwarzschild radius: {bh.schwarzschild_radius_m:.3f} m")
    print(f"Mass: {bh.mass_kg:.6e} kg ({bh.mass_solar:.3f} solar masses)")
    a = (6.0 * bh.schwarzschild_radius_m, 0.0, 0.0)
    b = (4.0 * bh.schwarzschild_radius_m, 4.0 * bh.schwarzschild_radius_m, 0.0)
    result = bh.find_two_shortest_geodesics(a, b)
    print(f"Angular separation: {result.separation_angle_rad:.6f} rad")
    for i, path in enumerate(result.paths, start=1):
        print(
            f"path {i}: dir={path.direction:+d}, branch={path.branch}, "
            f"b={path.impact_parameter_m:.6e} m, t={path.travel_time_s:.6f} s"
        )
    print(f"Lag (path2 - path1): {result.lag_between_fastest_two_s:.6f} s")

    pairs = [
        ((6.0 * bh.schwarzschild_radius_m, 0.0, 0.0), (4.0 * bh.schwarzschild_radius_m, 4.0 * bh.schwarzschild_radius_m, 0.0)),
        ((7.0 * bh.schwarzschild_radius_m, 0.0, 0.0), (4.2 * bh.schwarzschild_radius_m, 3.0 * bh.schwarzschild_radius_m, 0.0)),
    ]
    batch = bh.find_two_shortest_geodesics_batch(pairs, backend="process")
    print(f"Batch solved pairs: {len(batch)}")
