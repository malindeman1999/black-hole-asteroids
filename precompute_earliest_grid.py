from __future__ import annotations

import argparse
from math import asin, cos, pi, sin, sqrt
from pathlib import Path
import json
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Add project root so local module imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_geodesics import SchwarzschildBlackHole, cp


class PrecomputedEarliestInterpolator:
    """
    Interpolate precomputed +1/-1 tables on (B_r, A_r, A_phi).

    Query inputs are:
    - B radial positions in meters (shape [N])
    - A Cartesian positions in meters (shape [N, 3])
    """

    def __init__(
        self,
        rs_m: float,
        a_r_m: np.ndarray,
        a_phi_rad: np.ndarray,
        b_r_m: np.ndarray,
        dt_plus_3d: np.ndarray,
        dt_minus_3d: np.ndarray,
        gamma_b_plus_3d: np.ndarray,
        gamma_b_minus_3d: np.ndarray,
        gamma_a_plus_3d: np.ndarray,
        gamma_a_minus_3d: np.ndarray,
        ok_plus_3d: np.ndarray,
        ok_minus_3d: np.ndarray,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        self.rs_m = float(rs_m)
        self.a_r_m = np.asarray(a_r_m, dtype=float)
        self.a_phi_rad = np.asarray(a_phi_rad, dtype=float)
        self.b_r_m = np.asarray(b_r_m, dtype=float)
        self.dt_plus_3d = np.asarray(dt_plus_3d, dtype=float)
        self.dt_minus_3d = np.asarray(dt_minus_3d, dtype=float)
        self.gamma_b_plus_3d = np.asarray(gamma_b_plus_3d, dtype=float)
        self.gamma_b_minus_3d = np.asarray(gamma_b_minus_3d, dtype=float)
        self.gamma_a_plus_3d = np.asarray(gamma_a_plus_3d, dtype=float)
        self.gamma_a_minus_3d = np.asarray(gamma_a_minus_3d, dtype=float)
        self.ok_plus_3d = np.asarray(ok_plus_3d, dtype=bool)
        self.ok_minus_3d = np.asarray(ok_minus_3d, dtype=bool)
        self.metadata = dict(metadata or {})

        if self.a_r_m.ndim != 1 or self.a_phi_rad.ndim != 1 or self.b_r_m.ndim != 1:
            raise ValueError("Expected 1D coordinate axes for a_r_m, a_phi_rad, and b_r_m.")
        if self.a_r_m.size < 2 or self.a_phi_rad.size < 2 or self.b_r_m.size < 2:
            raise ValueError("Interpolation requires at least 2 samples along each axis.")

        expected_shape = (self.b_r_m.size, self.a_r_m.size, self.a_phi_rad.size)
        for name, arr in [
            ("dt_plus_3d", self.dt_plus_3d),
            ("dt_minus_3d", self.dt_minus_3d),
            ("gamma_b_plus_3d", self.gamma_b_plus_3d),
            ("gamma_b_minus_3d", self.gamma_b_minus_3d),
            ("gamma_a_plus_3d", self.gamma_a_plus_3d),
            ("gamma_a_minus_3d", self.gamma_a_minus_3d),
            ("ok_plus_3d", self.ok_plus_3d),
            ("ok_minus_3d", self.ok_minus_3d),
        ]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} shape {arr.shape} does not match expected {expected_shape}.")

        # A-phi grid is expected regular over [-pi, pi) from precompute.
        self._phi0 = float(self.a_phi_rad[0])
        self._dphi = float(2.0 * pi / self.a_phi_rad.size)
        self._backend_cache: Dict[str, object] = {}

    @classmethod
    def from_npz(cls, npz_path: Path | str) -> "PrecomputedEarliestInterpolator":
        data = np.load(Path(npz_path), allow_pickle=True)
        required = [
            "rs_m",
            "a_r_m",
            "a_phi_rad",
            "b_r_m",
            "delta_t_plus_s",
            "delta_t_minus_s",
            "gamma_at_b_plus_rad",
            "gamma_at_b_minus_rad",
            "ok_plus",
            "ok_minus",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required arrays in npz: {missing}")

        a_r = np.asarray(data["a_r_m"], dtype=float)
        a_phi = np.asarray(data["a_phi_rad"], dtype=float)
        b_r = np.asarray(data["b_r_m"], dtype=float)
        n_b = b_r.size
        n_ar = a_r.size
        n_ap = a_phi.size
        n_a = n_ar * n_ap

        def _reshape3(name: str, arr: np.ndarray) -> np.ndarray:
            arr2 = np.asarray(arr)
            if arr2.ndim != 2:
                raise ValueError(f"{name} expected 2D array [n_b, n_a], got shape {arr2.shape}")
            if arr2.shape != (n_b, n_a):
                raise ValueError(f"{name} has shape {arr2.shape}, expected {(n_b, n_a)}")
            return arr2.reshape(n_b, n_ar, n_ap)

        metadata = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except Exception:
                metadata = {}

        return cls(
            rs_m=float(data["rs_m"]),
            a_r_m=a_r,
            a_phi_rad=a_phi,
            b_r_m=b_r,
            dt_plus_3d=_reshape3("delta_t_plus_s", data["delta_t_plus_s"]),
            dt_minus_3d=_reshape3("delta_t_minus_s", data["delta_t_minus_s"]),
            gamma_b_plus_3d=_reshape3("gamma_at_b_plus_rad", data["gamma_at_b_plus_rad"]),
            gamma_b_minus_3d=_reshape3("gamma_at_b_minus_rad", data["gamma_at_b_minus_rad"]),
            gamma_a_plus_3d=(
                _reshape3("gamma_at_a_plus_rad", data["gamma_at_a_plus_rad"])
                if "gamma_at_a_plus_rad" in data
                else np.full((n_b, n_ar, n_ap), np.nan, dtype=float)
            ),
            gamma_a_minus_3d=(
                _reshape3("gamma_at_a_minus_rad", data["gamma_at_a_minus_rad"])
                if "gamma_at_a_minus_rad" in data
                else np.full((n_b, n_ar, n_ap), np.nan, dtype=float)
            ),
            ok_plus_3d=_reshape3("ok_plus", data["ok_plus"]).astype(bool),
            ok_minus_3d=_reshape3("ok_minus", data["ok_minus"]).astype(bool),
            metadata=metadata,
        )

    def _interp_axis_indices(self, x, axis, xp):
        x_clamped = xp.clip(x, axis[0], axis[-1])
        i0 = xp.searchsorted(axis, x_clamped, side="right") - 1
        i0 = xp.clip(i0, 0, axis.size - 2)
        i1 = i0 + 1
        denom = axis[i1] - axis[i0]
        w = xp.where(denom > 0.0, (x_clamped - axis[i0]) / denom, xp.zeros_like(x_clamped))
        return i0.astype(int), i1.astype(int), w

    def _interp_phi_indices(self, phi, xp):
        # Wrap to [phi0, phi0 + 2*pi) before computing periodic cell index.
        phi_wrapped = xp.mod(phi - self._phi0, 2.0 * pi) + self._phi0
        p = (phi_wrapped - self._phi0) / self._dphi
        j0 = xp.floor(p).astype(int)
        j0 = xp.mod(j0, self.a_phi_rad.size).astype(int)
        j1 = xp.mod(j0 + 1, self.a_phi_rad.size).astype(int)
        w = p - xp.floor(p)
        return j0, j1, w

    @staticmethod
    def _masked_trilinear(arr3, ok3, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp):
        w000 = (1.0 - wb) * (1.0 - wr) * (1.0 - wp)
        w001 = (1.0 - wb) * (1.0 - wr) * wp
        w010 = (1.0 - wb) * wr * (1.0 - wp)
        w011 = (1.0 - wb) * wr * wp
        w100 = wb * (1.0 - wr) * (1.0 - wp)
        w101 = wb * (1.0 - wr) * wp
        w110 = wb * wr * (1.0 - wp)
        w111 = wb * wr * wp

        v000 = arr3[ib0, ir0, ip0]
        v001 = arr3[ib0, ir0, ip1]
        v010 = arr3[ib0, ir1, ip0]
        v011 = arr3[ib0, ir1, ip1]
        v100 = arr3[ib1, ir0, ip0]
        v101 = arr3[ib1, ir0, ip1]
        v110 = arr3[ib1, ir1, ip0]
        v111 = arr3[ib1, ir1, ip1]

        m000 = ok3[ib0, ir0, ip0] & xp.isfinite(v000)
        m001 = ok3[ib0, ir0, ip1] & xp.isfinite(v001)
        m010 = ok3[ib0, ir1, ip0] & xp.isfinite(v010)
        m011 = ok3[ib0, ir1, ip1] & xp.isfinite(v011)
        m100 = ok3[ib1, ir0, ip0] & xp.isfinite(v100)
        m101 = ok3[ib1, ir0, ip1] & xp.isfinite(v101)
        m110 = ok3[ib1, ir1, ip0] & xp.isfinite(v110)
        m111 = ok3[ib1, ir1, ip1] & xp.isfinite(v111)

        ww000 = xp.where(m000, w000, 0.0)
        ww001 = xp.where(m001, w001, 0.0)
        ww010 = xp.where(m010, w010, 0.0)
        ww011 = xp.where(m011, w011, 0.0)
        ww100 = xp.where(m100, w100, 0.0)
        ww101 = xp.where(m101, w101, 0.0)
        ww110 = xp.where(m110, w110, 0.0)
        ww111 = xp.where(m111, w111, 0.0)

        wsum = ww000 + ww001 + ww010 + ww011 + ww100 + ww101 + ww110 + ww111
        vsum = (
            ww000 * v000
            + ww001 * v001
            + ww010 * v010
            + ww011 * v011
            + ww100 * v100
            + ww101 * v101
            + ww110 * v110
            + ww111 * v111
        )
        out = xp.where(wsum > 0.0, vsum / wsum, xp.nan)
        ok = wsum > 0.0
        return out, ok

    def prepare_backend(self, use_gpu: bool = True) -> str:
        """
        Materialize interpolation tables on the selected backend once.
        Returns backend label: 'gpu' or 'cpu'.
        """
        backend = "gpu" if (use_gpu and cp is not None) else "cpu"
        if backend in self._backend_cache:
            return backend
        xp = cp if backend == "gpu" else np
        self._backend_cache[backend] = {
            "xp": xp,
            "b_axis": xp.asarray(self.b_r_m, dtype=float),
            "ar_axis": xp.asarray(self.a_r_m, dtype=float),
            "dt_p": xp.asarray(self.dt_plus_3d, dtype=float),
            "dt_m": xp.asarray(self.dt_minus_3d, dtype=float),
            "gb_p": xp.asarray(self.gamma_b_plus_3d, dtype=float),
            "gb_m": xp.asarray(self.gamma_b_minus_3d, dtype=float),
            "ga_p": xp.asarray(self.gamma_a_plus_3d, dtype=float),
            "ga_m": xp.asarray(self.gamma_a_minus_3d, dtype=float),
            "ok_p": xp.asarray(self.ok_plus_3d, dtype=bool),
            "ok_m": xp.asarray(self.ok_minus_3d, dtype=bool),
        }
        return backend

    def _backend_tables(self, use_gpu: bool):
        backend = self.prepare_backend(use_gpu=use_gpu)
        return self._backend_cache[backend]

    def interpolate_batch(
        self,
        b_r_m: Sequence[float] | np.ndarray,
        a_points_m: Sequence[Sequence[float]] | np.ndarray,
        use_gpu: bool = True,
        batch_size: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate lag/angles for N queries.

        Returns dict fields (shape [N]):
        - delta_t_plus_s, delta_t_minus_s
        - time_lag_signed_s, time_lag_abs_s
        - gamma_at_b_plus_rad, gamma_at_b_minus_rad
        - ok_plus, ok_minus, ok_both
        """
        b_r_np = np.asarray(b_r_m, dtype=float).reshape(-1)
        a_np = np.asarray(a_points_m, dtype=float)
        if a_np.ndim != 2 or a_np.shape[1] != 3:
            raise ValueError("a_points_m must have shape [N, 3].")
        if b_r_np.size != a_np.shape[0]:
            raise ValueError("b_r_m and a_points_m must have the same length N.")

        n = b_r_np.size
        chunk = max(1, int(batch_size))
        bt = self._backend_tables(use_gpu=use_gpu)
        xp = bt["xp"]
        b_axis = bt["b_axis"]
        ar_axis = bt["ar_axis"]
        dt_p = bt["dt_p"]
        dt_m = bt["dt_m"]
        gb_p = bt["gb_p"]
        gb_m = bt["gb_m"]
        ga_p = bt["ga_p"]
        ga_m = bt["ga_m"]
        ok_p = bt["ok_p"]
        ok_m = bt["ok_m"]

        out_dt_p = np.full(n, np.nan, dtype=float)
        out_dt_m = np.full(n, np.nan, dtype=float)
        out_gb_p = np.full(n, np.nan, dtype=float)
        out_gb_m = np.full(n, np.nan, dtype=float)
        out_ga_p = np.full(n, np.nan, dtype=float)
        out_ga_m = np.full(n, np.nan, dtype=float)
        out_ok_p = np.zeros(n, dtype=bool)
        out_ok_m = np.zeros(n, dtype=bool)

        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            br = xp.asarray(b_r_np[lo:hi], dtype=float)
            aa = xp.asarray(a_np[lo:hi], dtype=float)

            ar = xp.sqrt(aa[:, 0] * aa[:, 0] + aa[:, 1] * aa[:, 1] + aa[:, 2] * aa[:, 2])
            aphi = xp.arctan2(aa[:, 1], aa[:, 0])

            ib0, ib1, wb = self._interp_axis_indices(br, b_axis, xp=xp)
            ir0, ir1, wr = self._interp_axis_indices(ar, ar_axis, xp=xp)
            ip0, ip1, wp = self._interp_phi_indices(aphi, xp=xp)

            dtp_chunk, okp_chunk = self._masked_trilinear(dt_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            dtm_chunk, okm_chunk = self._masked_trilinear(dt_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gbp_chunk, _ = self._masked_trilinear(gb_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gbm_chunk, _ = self._masked_trilinear(gb_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gap_chunk, _ = self._masked_trilinear(ga_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gam_chunk, _ = self._masked_trilinear(ga_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)

            if xp is np:
                out_dt_p[lo:hi] = dtp_chunk
                out_dt_m[lo:hi] = dtm_chunk
                out_gb_p[lo:hi] = gbp_chunk
                out_gb_m[lo:hi] = gbm_chunk
                out_ga_p[lo:hi] = gap_chunk
                out_ga_m[lo:hi] = gam_chunk
                out_ok_p[lo:hi] = okp_chunk
                out_ok_m[lo:hi] = okm_chunk
            else:
                out_dt_p[lo:hi] = xp.asnumpy(dtp_chunk)
                out_dt_m[lo:hi] = xp.asnumpy(dtm_chunk)
                out_gb_p[lo:hi] = xp.asnumpy(gbp_chunk)
                out_gb_m[lo:hi] = xp.asnumpy(gbm_chunk)
                out_ga_p[lo:hi] = xp.asnumpy(gap_chunk)
                out_ga_m[lo:hi] = xp.asnumpy(gam_chunk)
                out_ok_p[lo:hi] = xp.asnumpy(okp_chunk)
                out_ok_m[lo:hi] = xp.asnumpy(okm_chunk)

        lag_signed = out_dt_m - out_dt_p
        lag_abs = np.abs(lag_signed)
        ok_both = out_ok_p & out_ok_m & np.isfinite(out_dt_p) & np.isfinite(out_dt_m)
        lag_signed[~ok_both] = np.nan
        lag_abs[~ok_both] = np.nan

        return {
            "delta_t_plus_s": out_dt_p,
            "delta_t_minus_s": out_dt_m,
            "time_lag_signed_s": lag_signed,
            "time_lag_abs_s": lag_abs,
            "gamma_at_b_plus_rad": out_gb_p,
            "gamma_at_b_minus_rad": out_gb_m,
            "gamma_at_a_plus_rad": out_ga_p,
            "gamma_at_a_minus_rad": out_ga_m,
            "ok_plus": out_ok_p,
            "ok_minus": out_ok_m,
            "ok_both": ok_both,
        }

    def interpolate_pairs_3d(
        self,
        a_points_m: Sequence[Sequence[float]] | np.ndarray,
        b_points_m: Sequence[Sequence[float]] | np.ndarray,
        use_gpu: bool = True,
        batch_size: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate for 3D (A,B) pairs by reducing each pair to its local 2D orbital plane.

        For each pair:
        1) Build a local plane basis with B on +x.
        2) Interpolate +1/-1 dt and gamma_at_b from precomputed tables.
        3) Convert branch directions back to global 3D arrival directions at B.

        Returns all fields from interpolate_batch plus:
        - arrival_dir_plus_xyz, arrival_dir_minus_xyz   shape [N,3]
        - arrival_az_plus_rad, arrival_az_minus_rad     shape [N]
        - arrival_el_plus_rad, arrival_el_minus_rad     shape [N]
        """
        a_np = np.asarray(a_points_m, dtype=float)
        b_np = np.asarray(b_points_m, dtype=float)
        if a_np.ndim != 2 or a_np.shape[1] != 3:
            raise ValueError("a_points_m must have shape [N,3].")
        if b_np.ndim != 2 or b_np.shape[1] != 3:
            raise ValueError("b_points_m must have shape [N,3].")
        if a_np.shape[0] != b_np.shape[0]:
            raise ValueError("a_points_m and b_points_m must have the same length N.")

        n = a_np.shape[0]
        chunk = max(1, int(batch_size))
        bt = self._backend_tables(use_gpu=use_gpu)
        xp = bt["xp"]
        eps = 1e-12

        b_axis = bt["b_axis"]
        ar_axis = bt["ar_axis"]
        dt_p = bt["dt_p"]
        dt_m = bt["dt_m"]
        gb_p = bt["gb_p"]
        gb_m = bt["gb_m"]
        ga_p = bt["ga_p"]
        ga_m = bt["ga_m"]
        ok_p = bt["ok_p"]
        ok_m = bt["ok_m"]

        out_dt_p = np.full(n, np.nan, dtype=float)
        out_dt_m = np.full(n, np.nan, dtype=float)
        out_gb_p = np.full(n, np.nan, dtype=float)
        out_gb_m = np.full(n, np.nan, dtype=float)
        out_ga_p = np.full(n, np.nan, dtype=float)
        out_ga_m = np.full(n, np.nan, dtype=float)
        out_ok_p = np.zeros(n, dtype=bool)
        out_ok_m = np.zeros(n, dtype=bool)
        out_dir_p = np.full((n, 3), np.nan, dtype=float)
        out_dir_m = np.full((n, 3), np.nan, dtype=float)
        out_az_p = np.full(n, np.nan, dtype=float)
        out_az_m = np.full(n, np.nan, dtype=float)
        out_el_p = np.full(n, np.nan, dtype=float)
        out_el_m = np.full(n, np.nan, dtype=float)

        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            aa = xp.asarray(a_np[lo:hi], dtype=float)
            bb = xp.asarray(b_np[lo:hi], dtype=float)
            m = hi - lo

            r_b = xp.sqrt(xp.sum(bb * bb, axis=1))
            if xp.any(r_b <= self.rs_m):
                raise ValueError("All B points must be outside the event horizon.")
            er = bb / r_b[:, None]

            # Build in-plane tangential basis ephi from A-perpendicular component.
            x_a = xp.sum(aa * er, axis=1)
            a_perp = aa - x_a[:, None] * er
            n_perp = xp.sqrt(xp.sum(a_perp * a_perp, axis=1))

            # Fallback basis when A and B are nearly collinear.
            ref = xp.zeros((m, 3), dtype=float)
            ref[:, 2] = 1.0
            use_y_ref = xp.abs(er[:, 2]) > 0.9
            ref[use_y_ref, 1] = 1.0
            ref[use_y_ref, 2] = 0.0
            ephi_fallback = xp.cross(ref, er)
            n_fb = xp.sqrt(xp.sum(ephi_fallback * ephi_fallback, axis=1))
            ephi_fallback = ephi_fallback / xp.maximum(n_fb[:, None], eps)

            ephi = xp.where((n_perp > eps)[:, None], a_perp / xp.maximum(n_perp[:, None], eps), ephi_fallback)
            y_a = xp.sum(aa * ephi, axis=1)

            # Local 2D representation (B on +x).
            a_local = xp.zeros((m, 3), dtype=float)
            a_local[:, 0] = x_a
            a_local[:, 1] = y_a

            # Interpolation indices/weights in local coordinates.
            a_r = xp.sqrt(x_a * x_a + y_a * y_a)
            a_phi = xp.arctan2(y_a, x_a)
            ib0, ib1, wb = self._interp_axis_indices(r_b, b_axis, xp=xp)
            ir0, ir1, wr = self._interp_axis_indices(a_r, ar_axis, xp=xp)
            ip0, ip1, wp = self._interp_phi_indices(a_phi, xp=xp)

            dtp_chunk, okp_chunk = self._masked_trilinear(dt_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            dtm_chunk, okm_chunk = self._masked_trilinear(dt_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gbp_chunk, _ = self._masked_trilinear(gb_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gbm_chunk, _ = self._masked_trilinear(gb_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gap_chunk, _ = self._masked_trilinear(ga_p, ok_p, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)
            gam_chunk, _ = self._masked_trilinear(ga_m, ok_m, ib0, ib1, wb, ir0, ir1, wr, ip0, ip1, wp, xp=xp)

            # Branch orientation sign in local 2D frame.
            # short branch (+1) follows signed shortest angular change A->B; long is opposite.
            dtheta_short = xp.mod(-a_phi + pi, 2.0 * pi) - pi
            short_sign = xp.where(dtheta_short >= 0.0, 1.0, -1.0)
            orient_plus = short_sign
            orient_minus = -short_sign

            # Arrival radial sign at B (A inside => outward arrival; A outside => inward arrival).
            r_a3 = xp.sqrt(xp.sum(aa * aa, axis=1))
            radial_sign = xp.where(r_a3 <= r_b, 1.0, -1.0)

            def _build_arrival_dir(gamma_chunk, ok_chunk, orient_sign):
                d = xp.full((m, 3), xp.nan, dtype=float)
                good = ok_chunk & xp.isfinite(gamma_chunk)
                if xp.any(good):
                    dr = radial_sign * xp.cos(gamma_chunk)
                    dtan = orient_sign * xp.sin(gamma_chunk)
                    vv = dr[:, None] * er + dtan[:, None] * ephi
                    nv = xp.sqrt(xp.sum(vv * vv, axis=1))
                    vv = vv / xp.maximum(nv[:, None], eps)
                    d[good] = vv[good]
                return d

            dir_p = _build_arrival_dir(gbp_chunk, okp_chunk, orient_plus)
            dir_m = _build_arrival_dir(gbm_chunk, okm_chunk, orient_minus)

            az_p = xp.arctan2(dir_p[:, 1], dir_p[:, 0])
            az_m = xp.arctan2(dir_m[:, 1], dir_m[:, 0])
            el_p = xp.arcsin(xp.clip(dir_p[:, 2], -1.0, 1.0))
            el_m = xp.arcsin(xp.clip(dir_m[:, 2], -1.0, 1.0))

            if xp is np:
                out_dt_p[lo:hi] = dtp_chunk
                out_dt_m[lo:hi] = dtm_chunk
                out_gb_p[lo:hi] = gbp_chunk
                out_gb_m[lo:hi] = gbm_chunk
                out_ga_p[lo:hi] = gap_chunk
                out_ga_m[lo:hi] = gam_chunk
                out_ok_p[lo:hi] = okp_chunk
                out_ok_m[lo:hi] = okm_chunk
                out_dir_p[lo:hi] = dir_p
                out_dir_m[lo:hi] = dir_m
                out_az_p[lo:hi] = az_p
                out_az_m[lo:hi] = az_m
                out_el_p[lo:hi] = el_p
                out_el_m[lo:hi] = el_m
            else:
                out_dt_p[lo:hi] = xp.asnumpy(dtp_chunk)
                out_dt_m[lo:hi] = xp.asnumpy(dtm_chunk)
                out_gb_p[lo:hi] = xp.asnumpy(gbp_chunk)
                out_gb_m[lo:hi] = xp.asnumpy(gbm_chunk)
                out_ga_p[lo:hi] = xp.asnumpy(gap_chunk)
                out_ga_m[lo:hi] = xp.asnumpy(gam_chunk)
                out_ok_p[lo:hi] = xp.asnumpy(okp_chunk)
                out_ok_m[lo:hi] = xp.asnumpy(okm_chunk)
                out_dir_p[lo:hi] = xp.asnumpy(dir_p)
                out_dir_m[lo:hi] = xp.asnumpy(dir_m)
                out_az_p[lo:hi] = xp.asnumpy(az_p)
                out_az_m[lo:hi] = xp.asnumpy(az_m)
                out_el_p[lo:hi] = xp.asnumpy(el_p)
                out_el_m[lo:hi] = xp.asnumpy(el_m)

        lag_signed = out_dt_m - out_dt_p
        lag_abs = np.abs(lag_signed)
        ok_both = out_ok_p & out_ok_m & np.isfinite(out_dt_p) & np.isfinite(out_dt_m)
        lag_signed[~ok_both] = np.nan
        lag_abs[~ok_both] = np.nan

        return {
            "delta_t_plus_s": out_dt_p,
            "delta_t_minus_s": out_dt_m,
            "time_lag_signed_s": lag_signed,
            "time_lag_abs_s": lag_abs,
            "gamma_at_b_plus_rad": out_gb_p,
            "gamma_at_b_minus_rad": out_gb_m,
            "gamma_at_a_plus_rad": out_ga_p,
            "gamma_at_a_minus_rad": out_ga_m,
            "ok_plus": out_ok_p,
            "ok_minus": out_ok_m,
            "ok_both": ok_both,
            "arrival_dir_plus_xyz": out_dir_p,
            "arrival_dir_minus_xyz": out_dir_m,
            "arrival_az_plus_rad": out_az_p,
            "arrival_az_minus_rad": out_az_m,
            "arrival_el_plus_rad": out_el_p,
            "arrival_el_minus_rad": out_el_m,
        }


def interpolate_precomputed_lag_and_angles(
    npz_path: Path | str,
    b_r_m: Sequence[float] | np.ndarray,
    a_points_m: Sequence[Sequence[float]] | np.ndarray,
    use_gpu: bool = True,
    batch_size: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper: load precompute file, then interpolate in batches.
    """
    interp = PrecomputedEarliestInterpolator.from_npz(npz_path)
    return interp.interpolate_batch(b_r_m=b_r_m, a_points_m=a_points_m, use_gpu=use_gpu, batch_size=batch_size)


def interpolate_precomputed_lag_and_angles_3d(
    npz_path: Path | str,
    a_points_m: Sequence[Sequence[float]] | np.ndarray,
    b_points_m: Sequence[Sequence[float]] | np.ndarray,
    use_gpu: bool = True,
    batch_size: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper for 3D A/B pair interpolation with back-rotation to global orientation.
    """
    interp = PrecomputedEarliestInterpolator.from_npz(npz_path)
    return interp.interpolate_pairs_3d(
        a_points_m=a_points_m,
        b_points_m=b_points_m,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _arrival_gamma_at_b(rs: float, r_b: float, impact_parameter_m: float) -> float:
    s = impact_parameter_m * sqrt(1.0 - rs / r_b) / r_b
    return asin(_clamp(s, -1.0, 1.0))


def _local_gamma_at_radius(rs: float, r: float, impact_parameter_m: float) -> float:
    s = impact_parameter_m * sqrt(1.0 - rs / r) / r
    return asin(_clamp(s, -1.0, 1.0))


def _density_factor_from_error(max_interp_error_rel: float) -> float:
    # Bilinear interpolation errors generally scale ~O(h^2), so step size h ~ sqrt(err).
    # We anchor 5% => factor 1.0 and increase grid density as error tolerance decreases.
    err = max(1e-4, float(max_interp_error_rel))
    return sqrt(0.05 / err)


def _regular_phi_grid(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("phi grid count must be >= 1")
    return np.linspace(-pi, pi, n, endpoint=False, dtype=float)


def _radial_grid_clustered(rs: float, rmin_rs: float, rmax_rs: float, n: int, exponent: float) -> np.ndarray:
    if n < 1:
        raise ValueError("radial grid count must be >= 1")
    if rmax_rs <= rmin_rs:
        raise ValueError("rmax-rs multiplier must be > rmin-rs multiplier")
    u = np.linspace(0.0, 1.0, n, dtype=float)
    rmin = rmin_rs * rs
    rmax = rmax_rs * rs
    return rmin + (rmax - rmin) * np.power(u, exponent)


def _build_a_points(a_r: np.ndarray, a_phi: np.ndarray) -> np.ndarray:
    out = np.empty((a_r.size * a_phi.size, 3), dtype=float)
    k = 0
    for rr in a_r:
        for ph in a_phi:
            out[k, 0] = rr * cos(float(ph))
            out[k, 1] = rr * sin(float(ph))
            out[k, 2] = 0.0
            k += 1
    return out


def _build_b_points(b_r: np.ndarray, b_phi: np.ndarray, b_on_x_axis: bool) -> np.ndarray:
    if b_on_x_axis:
        out = np.zeros((b_r.size, 3), dtype=float)
        out[:, 0] = b_r
        return out

    out = np.empty((b_r.size * b_phi.size, 3), dtype=float)
    k = 0
    for rr in b_r:
        for ph in b_phi:
            out[k, 0] = rr * cos(float(ph))
            out[k, 1] = rr * sin(float(ph))
            out[k, 2] = 0.0
            k += 1
    return out


def _get_path_by_direction(paths, direction: int):
    for p in paths:
        if p.direction == direction:
            return p
    return None


def _solve_pairs_robust(
    bh: SchwarzschildBlackHole,
    pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
    use_gpu: bool,
):
    n = len(pairs)
    out = [None] * n

    def solve_span(lo: int, hi: int) -> None:
        subset = pairs[lo:hi]
        try:
            rr = bh.find_two_shortest_geodesics_batch(subset, a_before_b=True, use_gpu=use_gpu)
            if len(rr) != (hi - lo):
                raise RuntimeError("Unexpected batch result length mismatch.")
            for i, result in enumerate(rr):
                out[lo + i] = result
            return
        except Exception:
            if hi - lo == 1:
                out[lo] = None
                return
            mid = lo + (hi - lo) // 2
            solve_span(lo, mid)
            solve_span(mid, hi)

    solve_span(0, n)
    return out


def _auto_grid_sizes(error_rel: float) -> Tuple[int, int, int, int, float]:
    factor = _density_factor_from_error(error_rel)

    # Base at 5% interpolation tolerance.
    a_r_n = max(12, int(round(36 * factor)))
    a_phi_n = max(24, int(round(96 * factor)))
    b_r_n = max(10, int(round(28 * factor)))
    b_phi_n = max(24, int(round(96 * factor)))

    # Stronger radial concentration for denser runs; keeps samples near rs.
    radial_exponent = 2.2 + 0.6 * min(3.0, factor)
    return a_r_n, a_phi_n, b_r_n, b_phi_n, radial_exponent


def _save_npz(path: Path, arrays: Dict[str, np.ndarray], metadata: Dict[str, object]) -> None:
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, indent=2), dtype=object)
    np.savez_compressed(path, **payload)


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


def _cumulative_trapezoid(values: Sequence[float], x: Sequence[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(values)):
        dx = x[i] - x[i - 1]
        out.append(out[-1] + 0.5 * (values[i - 1] + values[i]) * dx)
    return out


def _find_turning_radius(bh: SchwarzschildBlackHole, impact_b: float, r_cap: float) -> float:
    rs = bh.schwarzschild_radius_m
    rph = bh.photon_sphere_radius_m
    lo = rph * (1.0 + 1e-9)
    hi = r_cap * (1.0 - 1e-9)

    def f(r: float) -> float:
        return 1.0 / (impact_b * impact_b) - (1.0 - rs / r) / (r * r)

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
    bh: SchwarzschildBlackHole, r_start: float, r_end: float, impact_b: float, target_phi: float, branch: str, n: int = 900
) -> Tuple[List[float], List[float]]:
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

    return r_samples, phi_samples


def _unit_xy(vec_xy: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec_xy))
    if n <= 0.0:
        return np.asarray([1.0, 0.0], dtype=float)
    return vec_xy / n


def _direction_from_angle_at_a(a_point: np.ndarray, gamma_at_a: float, side: int) -> np.ndarray:
    er = _unit_xy(a_point[:2])
    ephi = np.asarray([-er[1], er[0]], dtype=float)
    d = (-cos(gamma_at_a)) * er + (side * sin(gamma_at_a)) * ephi
    return _unit_xy(d)


def _plot_single_b_debug(
    bh: SchwarzschildBlackHole,
    out_path: Path,
    rs: float,
    a_points: np.ndarray,
    b_point: np.ndarray,
    gamma_a_plus_row: np.ndarray,
    gamma_a_minus_row: np.ndarray,
    ok_plus_row: np.ndarray,
    ok_minus_row: np.ndarray,
    vector_length_rs: float,
    show_plot: bool,
    use_gpu: bool,
    endpoint_tol_rs: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Debug plot skipped (matplotlib import failed): {exc}")
        return

    vec_len = float(vector_length_rs) * rs
    bx, by = float(b_point[0]), float(b_point[1])
    r_b = sqrt(bx * bx + by * by)
    th_b = float(np.arctan2(by, bx))

    a_xy = np.asarray(a_points[:, :2], dtype=float)
    a_r_m = np.linalg.norm(a_xy, axis=1)
    a_th = np.arctan2(a_xy[:, 1], a_xy[:, 0])
    a_r_rs = a_r_m / rs

    r_lim_rs = 1.12 * max(
        float(np.max(a_r_rs + (vec_len / rs))) if a_r_rs.size > 0 else 1.0,
        (r_b + vec_len) / rs,
        1.5,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), subplot_kw={"projection": "polar"})

    t = np.linspace(0.0, 2.0 * pi, 361)
    ax.plot(t, np.ones_like(t), "k-", lw=1.2)
    ax.scatter(a_th, a_r_rs, s=10, c="0.45", alpha=0.7)
    ax.scatter([th_b], [r_b / rs], c="green", s=44, marker="x", linewidths=1.7)

    for a_i in range(a_points.shape[0]):
        a = a_points[a_i]
        axx, axy = float(a[0]), float(a[1])
        th0 = float(np.arctan2(axy, axx))
        r0 = float(sqrt(axx * axx + axy * axy)) / rs

        if bool(ok_plus_row[a_i]) and np.isfinite(gamma_a_plus_row[a_i]):
            d = _direction_from_angle_at_a(a, float(gamma_a_plus_row[a_i]), side=+1)
            x1, y1 = axx + vec_len * d[0], axy + vec_len * d[1]
            th1 = float(np.arctan2(y1, x1))
            r1 = float(sqrt(x1 * x1 + y1 * y1)) / rs
            th_pair = np.unwrap(np.asarray([th0, th1], dtype=float))
            ax.plot(th_pair, [r0, r1], color="red", lw=1.0, alpha=0.9)
        else:
            ax.scatter([th0], [r0], c="red", s=16, alpha=0.9)

        if bool(ok_minus_row[a_i]) and np.isfinite(gamma_a_minus_row[a_i]):
            d = _direction_from_angle_at_a(a, float(gamma_a_minus_row[a_i]), side=-1)
            x1, y1 = axx + vec_len * d[0], axy + vec_len * d[1]
            th1 = float(np.arctan2(y1, x1))
            r1 = float(sqrt(x1 * x1 + y1 * y1)) / rs
            th_pair = np.unwrap(np.asarray([th0, th1], dtype=float))
            ax.plot(th_pair, [r0, r1], color="blue", lw=1.0, alpha=0.9)
        else:
            ax.scatter([th0], [r0], c="blue", s=26, alpha=0.9)

    rng = random.Random(42)
    eligible_a = [i for i in range(a_points.shape[0]) if float(np.linalg.norm(a_points[i, :2])) >= 5.0 * rs]
    sample_n = min(5, len(eligible_a))
    checked_paths = 0
    failed_paths = 0
    max_start_err_m = 0.0
    max_end_err_m = 0.0
    endpoint_tol_m = float(endpoint_tol_rs) * rs
    for a_i in rng.sample(eligible_a, sample_n):
        a = a_points[a_i]
        try:
            result = bh.find_two_shortest_geodesics(a, b_point, a_before_b=True, use_gpu=use_gpu)
        except Exception:
            continue
        r_start = float(np.linalg.norm(a))
        r_end = float(np.linalg.norm(b_point))
        th_start = float(np.arctan2(float(a[1]), float(a[0])))
        th_end = float(np.arctan2(float(b_point[1]), float(b_point[0])))
        dtheta_short = ((th_end - th_start + pi) % (2.0 * pi)) - pi
        short_sign = +1.0 if dtheta_short >= 0.0 else -1.0
        gamma_short = abs(dtheta_short)
        gamma_long = 2.0 * pi - gamma_short

        for path in result.paths:
            try:
                r_samples, phi_samples = _build_path_profile(
                    bh=bh,
                    r_start=r_start,
                    r_end=r_end,
                    impact_b=float(path.impact_parameter_m),
                    target_phi=float(path.target_azimuth_rad),
                    branch=str(path.branch),
                    n=900,
                )
                is_short = abs(float(path.target_azimuth_rad) - gamma_short) <= abs(
                    float(path.target_azimuth_rad) - gamma_long
                )
                orient_sign = short_sign if is_short else -short_sign
                theta = np.asarray([th_start + orient_sign * p for p in phi_samples], dtype=float)
                theta = np.unwrap(theta)
                rr = np.asarray(r_samples, dtype=float) / rs

                # Verify reconstructed profile starts near A and ends near B.
                r_m = np.asarray(r_samples, dtype=float)
                x = r_m * np.cos(theta)
                y = r_m * np.sin(theta)
                start_err = float(np.hypot(x[0] - float(a[0]), y[0] - float(a[1])))
                end_err = float(np.hypot(x[-1] - float(b_point[0]), y[-1] - float(b_point[1])))
                checked_paths += 1
                max_start_err_m = max(max_start_err_m, start_err)
                max_end_err_m = max(max_end_err_m, end_err)
                if start_err > endpoint_tol_m or end_err > endpoint_tol_m:
                    failed_paths += 1

                ax.plot(theta, rr, color="green", lw=3.5, alpha=0.6)
                # Mark reconstructed geodesic endpoints for visual verification.
                ax.scatter(
                    [theta[0], theta[-1]],
                    [rr[0], rr[-1]],
                    s=34,
                    facecolors="none",
                    edgecolors="purple",
                    linewidths=1.2,
                    zorder=4,
                )
            except Exception:
                continue

    if checked_paths > 0:
        print(
            "Debug geodesic endpoint check: "
            f"checked={checked_paths}, failed={failed_paths}, "
            f"max_start_err={max_start_err_m/rs:.4f} rs, max_end_err={max_end_err_m/rs:.4f} rs, "
            f"tol={endpoint_tol_rs:.4f} rs"
        )
        if failed_paths > 0:
            print("Warning: some debug geodesic overlays do not terminate within endpoint tolerance.")

    ax.set_rmax(r_lim_rs)
    ax.set_rlabel_position(135)
    ax.set_ylabel("r / rs")
    ax.grid(alpha=0.24)
    ax.set_title(
        "Single-B debug view (polar: theta, r/rs)\n"
        f"r_B/rs={r_b/rs:.3f}, red=+1, blue=-1, green=sample geodesics",
        fontsize=10,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved debug plot: {out_path}")

    if show_plot and plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute +1/-1 geodesic travel-time and arrival-angle tables for A/B grids "
            "for fast earliest-angle interpolation."
        )
    )
    parser.add_argument("--quality", choices=["high", "medium", "fast"], default="fast")
    parser.add_argument(
        "--max-interp-error",
        type=float,
        default=0.05,
        help="Target relative interpolation error; lower values produce denser grids (default: 0.05).",
    )
    parser.add_argument(
        "--rmin-rs",
        type=float,
        # Keep default just outside the Schwarzschild photon sphere at 1.5*rs.
        default=1.6,
        help="Minimum radial grid multiplier of rs (default: 1.6).",
    )
    parser.add_argument("--rmax-rs", type=float, default=10.0, help="Maximum radial grid multiplier of rs.")
    parser.add_argument(
        "--a-r-count",
        type=int,
        default=0,
        help="Override A radial grid count (0 => auto from max-interp-error).",
    )
    parser.add_argument(
        "--a-phi-count",
        type=int,
        default=0,
        help="Override A phi grid count over [-pi, pi) (0 => auto from max-interp-error).",
    )
    parser.add_argument(
        "--b-r-count",
        type=int,
        default=0,
        help="Override B radial grid count (0 => auto from max-interp-error).",
    )
    parser.add_argument(
        "--b-phi-count",
        type=int,
        default=0,
        help="Override B phi grid count over [-pi, pi) when B is not restricted to x-axis.",
    )
    parser.add_argument(
        "--b-on-x-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict B to +x axis (default: true).",
    )
    parser.add_argument(
        "--avoid-axis-degeneracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When B is on +x axis, force odd A phi count to avoid exact A=0 deg samples "
            "while preserving boundary coverage at -pi (default: true)."
        ),
    )
    parser.add_argument(
        "--single-b-index",
        type=int,
        default=-1,
        help=(
            "If >=0, only precompute a single flattened B grid index. "
            "Useful for fast debugging runs."
        ),
    )
    parser.add_argument(
        "--radial-exponent",
        type=float,
        default=0.0,
        help="Override radial clustering exponent (>1 concentrates near rs; 0 => auto).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of (A,B) pairs per GPU/CPU batch solve (default: 5000).",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU batch solve when CuPy is available (default: true).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "earliest_angles_precompute_10rs.npz",
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--debug-plot-output",
        type=Path,
        default=None,
        help=(
            "Optional debug plot path for a single-B run. "
            "If omitted and --single-b-index is used, defaults to figures/<output_stem>_debug_b<index>.png."
        ),
    )
    parser.add_argument(
        "--debug-show-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show debug plot interactively when backend supports it (default: false).",
    )
    parser.add_argument(
        "--debug-vector-length-rs",
        type=float,
        default=0.5,
        help="Debug vector length in units of rs (default: 0.5).",
    )
    parser.add_argument(
        "--debug-endpoint-tol-rs",
        type=float,
        default=0.05,
        help="Endpoint verification tolerance for debug geodesic overlays in rs units (default: 0.05).",
    )
    args = parser.parse_args()

    bh = SchwarzschildBlackHole.from_diameter_light_seconds(1.0).with_quality(args.quality)
    rs = bh.schwarzschild_radius_m

    auto_a_r_n, auto_a_phi_n, auto_b_r_n, auto_b_phi_n, auto_exp = _auto_grid_sizes(args.max_interp_error)

    a_r_n = args.a_r_count if args.a_r_count > 0 else auto_a_r_n
    a_phi_n = args.a_phi_count if args.a_phi_count > 0 else auto_a_phi_n
    b_r_n = args.b_r_count if args.b_r_count > 0 else auto_b_r_n
    b_phi_n = args.b_phi_count if args.b_phi_count > 0 else auto_b_phi_n
    radial_exp = args.radial_exponent if args.radial_exponent > 0.0 else auto_exp
    a_phi_forced_odd_for_degeneracy = False
    if args.b_on_x_axis and args.avoid_axis_degeneracy and (a_phi_n % 2 == 0):
        # Odd counts keep phi=-pi on-grid but exclude phi=0 from the endpoint=False grid.
        # This avoids A/B collinearity at +x without losing interpolation support near +/-pi.
        a_phi_n += 1
        a_phi_forced_odd_for_degeneracy = True

    if args.rmin_rs <= 1.0:
        raise ValueError("rmin-rs must be > 1.0 (outside event horizon).")

    a_r = _radial_grid_clustered(rs, args.rmin_rs, args.rmax_rs, a_r_n, radial_exp)
    b_r = _radial_grid_clustered(rs, args.rmin_rs, args.rmax_rs, b_r_n, radial_exp)
    a_phi = _regular_phi_grid(a_phi_n)
    b_phi = np.asarray([0.0], dtype=float) if args.b_on_x_axis else _regular_phi_grid(b_phi_n)

    a_points = _build_a_points(a_r, a_phi)
    b_points_full = _build_b_points(b_r, b_phi, b_on_x_axis=args.b_on_x_axis)

    selected_b_original_index = -1
    if args.single_b_index >= 0:
        if args.single_b_index >= b_points_full.shape[0]:
            raise ValueError(
                f"single-b-index {args.single_b_index} out of range [0, {b_points_full.shape[0] - 1}]"
            )
        selected_b_original_index = int(args.single_b_index)
        b_points = b_points_full[selected_b_original_index : selected_b_original_index + 1]
    else:
        b_points = b_points_full

    n_a = a_points.shape[0]
    n_b = b_points.shape[0]
    n_pairs = n_a * n_b

    use_gpu_active = bool(args.use_gpu and cp is not None)
    mode = "GPU-batch" if use_gpu_active else "CPU-batch"

    print(f"Mode: {mode} | quality={args.quality}")
    print(f"Target max interpolation error: {args.max_interp_error:.5f}")
    print(
        "Grid sizes: "
        f"A_r={a_r_n}, A_phi={a_phi_n}, B_r={b_r_n}, B_phi={b_phi.size}, radial_exp={radial_exp:.3f}"
    )
    if a_phi_forced_odd_for_degeneracy:
        print("Adjusted A-phi count to odd to avoid A=0 deg x-axis degeneracy when B is on +x.")
    print(f"Total A points: {n_a}")
    print(f"Total B points: {n_b}")
    if selected_b_original_index >= 0:
        bp = b_points[0]
        print(
            "Single-B mode: "
            f"using original B index {selected_b_original_index} at "
            f"({bp[0]:.6e}, {bp[1]:.6e}, {bp[2]:.6e}) m"
        )
    print(f"Total pairs: {n_pairs}")

    dt_plus = np.full((n_b, n_a), np.nan, dtype=float)
    dt_minus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_plus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_minus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_a_plus = np.full((n_b, n_a), np.nan, dtype=float)
    gamma_a_minus = np.full((n_b, n_a), np.nan, dtype=float)
    impact_plus = np.full((n_b, n_a), np.nan, dtype=float)
    impact_minus = np.full((n_b, n_a), np.nan, dtype=float)
    ok_plus = np.zeros((n_b, n_a), dtype=bool)
    ok_minus = np.zeros((n_b, n_a), dtype=bool)

    solve_t0 = time.perf_counter()

    # Pair indexing: flat_i = b_i * n_a + a_i
    chunk = max(1, int(args.chunk_size))
    n_chunks = (n_pairs + chunk - 1) // chunk
    failed_pairs = 0
    for chunk_idx, start in enumerate(range(0, n_pairs, chunk), start=1):
        stop = min(n_pairs, start + chunk)
        pairs: List[Tuple[Sequence[float], Sequence[float]]] = []
        flat_indices: List[int] = []

        for flat_i in range(start, stop):
            b_i = flat_i // n_a
            a_i = flat_i - b_i * n_a
            pairs.append((a_points[a_i], b_points[b_i]))
            flat_indices.append(flat_i)

        results = _solve_pairs_robust(bh=bh, pairs=pairs, use_gpu=args.use_gpu)

        for flat_i, result in zip(flat_indices, results):
            b_i = flat_i // n_a
            a_i = flat_i - b_i * n_a
            r_b = float(sqrt(float(np.dot(b_points[b_i], b_points[b_i]))))
            r_a = float(sqrt(float(np.dot(a_points[a_i], a_points[a_i]))))
            if result is None:
                failed_pairs += 1
                continue

            p_plus = _get_path_by_direction(result.paths, +1)
            if p_plus is not None:
                ok_plus[b_i, a_i] = True
                dt_plus[b_i, a_i] = float(p_plus.travel_time_s)
                impact_plus[b_i, a_i] = float(p_plus.impact_parameter_m)
                gamma_plus[b_i, a_i] = _local_gamma_at_radius(rs, r_b, float(p_plus.impact_parameter_m))
                gamma_a_plus[b_i, a_i] = _local_gamma_at_radius(rs, r_a, float(p_plus.impact_parameter_m))

            p_minus = _get_path_by_direction(result.paths, -1)
            if p_minus is not None:
                ok_minus[b_i, a_i] = True
                dt_minus[b_i, a_i] = float(p_minus.travel_time_s)
                impact_minus[b_i, a_i] = float(p_minus.impact_parameter_m)
                gamma_minus[b_i, a_i] = _local_gamma_at_radius(rs, r_b, float(p_minus.impact_parameter_m))
                gamma_a_minus[b_i, a_i] = _local_gamma_at_radius(rs, r_a, float(p_minus.impact_parameter_m))

        done = stop
        pct = 100.0 * done / n_pairs
        elapsed_s = time.perf_counter() - solve_t0
        print(
            f"Chunk {chunk_idx}/{n_chunks}: solved {done}/{n_pairs} pairs "
            f"({pct:.1f}%) in {elapsed_s:.2f} s"
        )

    total_solve_s = time.perf_counter() - solve_t0
    print(f"Total solve time: {total_solve_s:.3f} s")
    if failed_pairs > 0:
        print(f"Unsolved pairs (stored as NaN): {failed_pairs}")
        frac = failed_pairs / max(1, n_pairs)
        if frac > 0.25:
            print("Warning: high unsolved fraction. Consider increasing --rmin-rs (e.g., 4.8).")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if selected_b_original_index >= 0:
        b_r_out = np.asarray([float(np.linalg.norm(b_points[0, :2]))], dtype=float)
        b_phi_out = np.asarray([float(np.arctan2(b_points[0, 1], b_points[0, 0]))], dtype=float)
    else:
        b_r_out = np.asarray(b_r, dtype=float)
        b_phi_out = np.asarray(b_phi, dtype=float)

    arrays: Dict[str, np.ndarray] = {
        "rs_m": np.asarray(rs, dtype=float),
        "a_r_m": np.asarray(a_r, dtype=float),
        "a_phi_rad": np.asarray(a_phi, dtype=float),
        "b_r_m": b_r_out,
        "b_phi_rad": b_phi_out,
        "a_points_m": np.asarray(a_points, dtype=float),
        "b_points_m": np.asarray(b_points, dtype=float),
        "delta_t_plus_s": dt_plus,
        "delta_t_minus_s": dt_minus,
        "gamma_at_b_plus_rad": gamma_plus,
        "gamma_at_b_minus_rad": gamma_minus,
        "gamma_at_a_plus_rad": gamma_a_plus,
        "gamma_at_a_minus_rad": gamma_a_minus,
        "impact_parameter_plus_m": impact_plus,
        "impact_parameter_minus_m": impact_minus,
        "ok_plus": ok_plus,
        "ok_minus": ok_minus,
    }

    metadata: Dict[str, object] = {
        "description": (
            "Precomputed +1/-1 geodesic tables for interpolation-based earliest-angle solves. "
            "Rows index B points, columns index A points."
        ),
        "mode": mode,
        "quality": args.quality,
        "use_gpu_requested": bool(args.use_gpu),
        "gpu_available": bool(cp is not None),
        "max_interp_error_rel": float(args.max_interp_error),
        "rmin_rs": float(args.rmin_rs),
        "rmax_rs": float(args.rmax_rs),
        "radial_exponent": float(radial_exp),
        "a_r_count": int(a_r_n),
        "a_phi_count": int(a_phi_n),
        "b_r_count": int(b_r_n),
        "b_phi_count": int(b_phi_out.size),
        "b_on_x_axis": bool(args.b_on_x_axis),
        "avoid_axis_degeneracy": bool(args.avoid_axis_degeneracy),
        "a_phi_forced_odd_for_degeneracy": bool(a_phi_forced_odd_for_degeneracy),
        "single_b_index": int(selected_b_original_index),
        "total_pairs": int(n_pairs),
        "failed_pairs": int(failed_pairs),
        "total_solve_s": float(total_solve_s),
        "delta_t_semantics": "travel_time_s from A to B along the selected side-direction geodesic",
        "gamma_semantics": "arrival angle at B relative to the local radial direction",
        "gamma_a_semantics": "emission angle at A relative to the local radial direction",
        "shape_note": "2D tables are [n_b_points, n_a_points]",
    }

    _save_npz(args.output, arrays=arrays, metadata=metadata)
    print(f"Saved precompute table: {args.output}")

    if selected_b_original_index >= 0:
        debug_out = args.debug_plot_output
        if debug_out is None:
            debug_out = Path("figures") / f"{args.output.stem}_debug_b{selected_b_original_index}.png"
        _plot_single_b_debug(
            bh=bh,
            out_path=debug_out,
            rs=rs,
            a_points=a_points,
            b_point=b_points[0],
            gamma_a_plus_row=gamma_a_plus[0],
            gamma_a_minus_row=gamma_a_minus[0],
            ok_plus_row=ok_plus[0],
            ok_minus_row=ok_minus[0],
            vector_length_rs=float(args.debug_vector_length_rs),
            show_plot=bool(args.debug_show_plot),
            use_gpu=bool(args.use_gpu),
            endpoint_tol_rs=float(args.debug_endpoint_tol_rs),
        )


if __name__ == "__main__":
    main()
