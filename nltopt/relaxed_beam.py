r"""Relaxed backbone coefficient of a clamped-clamped SYMMETRIC laminated beam
(B = 0), the one-dimensional analogue of :mod:`nltopt.relaxed_plate`.

With B = 0 the linear stiffness decouples into an axial block K_uu (from A) and a
transverse block K_ww (from D).  The fundamental (w-block Ritz pair, mass
normalised) is phi; its stretch m = phi'^2 (= 2 eps_NL) drives the axial DOF,
whose static condensation (the order-2 SSM slaving, exact when the axial modes
are much faster than 2 omega_1) gives the relaxed cubic

    gamma = knl - 2 g^T K_uu^{-1} g,   knl = 1/2 int A m^2,   g = 1/2 int A m N_u',

and the frequency-normalised detuning J = gamma / (omega_1^2 phi_mid^2), so that
Omega/omega_1 = 1 + (3/8) J a^2 at mid-span amplitude a.

For immovable ends the condensed axial force is uniform, N = (1/2 int m)/int A^{-1},
so gamma = (int m)^2 / (2 int A^{-1}) (mesh-converged): the HARMONIC mean of A.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .laminated_beam import _shape_derivs, _GW
from .laminate import RHO


class RelaxedBeam:
    def __init__(self, n_elem=60, length=0.30, thickness=0.002, width=0.02):
        self.ne, self.L, self.h, self.b = int(n_elem), float(length), float(thickness), float(width)
        self.Le = self.L / self.ne
        self.rows = _shape_derivs(self.Le)
        self.ndof = 3 * (self.ne + 1)
        fixed = [0, 1, 2, self.ndof - 3, self.ndof - 2, self.ndof - 1]
        self.free = np.array([d for d in range(self.ndof) if d not in fixed])
        self.xe = (np.arange(self.ne) + 0.5) * self.Le
        self.rhoA = RHO * self.b * self.h
        # w-dof and u-dof index sets among the free DOFs
        typ = np.array([d % 3 for d in self.free])
        self.iu = np.where(typ == 0)[0]
        self.iw = np.where(typ != 0)[0]
        self.mid_dof = list(self.free).index(3 * (self.ne // 2) + 1)

    def evaluate(self, E):
        """E: (n_elem,) axial modulus of each element (single-ply laminate:
        A = b h E, D = b h^3/12 E).  Returns omega1, J, gamma, knl, relax."""
        A = self.b * self.h * np.asarray(E, float)
        D = self.b * self.h ** 3 / 12.0 * np.asarray(E, float)
        n = self.ndof
        Kw = np.zeros((n, n)); Mw = np.zeros((n, n)); Ku = np.zeros((n, n))
        J = 0.5 * self.Le
        for e in range(self.ne):
            d = np.arange(3 * e, 3 * e + 6); ix = np.ix_(d, d)
            for (Bu, Bw, Bww, Nu, Nw), wg in zip(self.rows, _GW):
                wl = wg * J
                Kw[ix] += wl * D[e] * np.outer(Bww, Bww)
                Ku[ix] += wl * A[e] * np.outer(Bu, Bu)
                Mw[ix] += wl * self.rhoA * np.outer(Nw, Nw)
        f = self.free
        Kww = Kw[np.ix_(f, f)][np.ix_(self.iw, self.iw)]
        Mww = Mw[np.ix_(f, f)][np.ix_(self.iw, self.iw)]
        Kuu = Ku[np.ix_(f, f)][np.ix_(self.iu, self.iu)]
        lam, V = eigh(Kww, Mww, subset_by_index=[0, 0])
        om2 = lam[0]
        phi = np.zeros(n); phi[f[self.iw]] = V[:, 0]
        mid = phi[3 * (self.ne // 2) + 1]
        if mid < 0:
            phi = -phi; mid = -mid
        knl = 0.0; gfull = np.zeros(n); mint = 0.0; ainv = 0.0
        for e in range(self.ne):
            d = np.arange(3 * e, 3 * e + 6)
            for (Bu, Bw, Bww, Nu, Nw), wg in zip(self.rows, _GW):
                wl = wg * J
                mq = (Bw @ phi[d]) ** 2
                knl += 0.5 * wl * A[e] * mq * mq
                gfull[d] += 0.5 * wl * A[e] * mq * Bu
                mint += wl * mq; ainv += wl / A[e]
        g = gfull[f[self.iu]]
        relax = 2.0 * g @ np.linalg.solve(Kuu, g)
        gamma = knl - relax
        return dict(omega1=np.sqrt(om2), J=gamma / (om2 * mid ** 2), gamma=gamma,
                    knl=knl, relax=relax, gamma_harmonic=mint ** 2 / (2.0 * ainv), mid=mid)
