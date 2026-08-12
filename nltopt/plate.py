r"""Multi-mode von Karman composite plate (assumed-modes / Galerkin).

The flagship physics: a simply-supported rectangular laminated plate reduced onto
its lowest ``n_modes`` bending modes, with the geometrically-nonlinear von Karman
membrane stretching.  The fiber-orientation field ``theta(x,y)`` sets the
per-point laminate bending and membrane stiffness ``D(theta)``, ``A(theta)`` (CLT,
symmetric layup so ``B=0``); the modal mass ``M``, linear stiffness ``K(theta)``
and the cubic von Karman nonlinear force follow by numerical quadrature over the
plate, so a spatially-varying tow layout is handled exactly.

Kinematics (SS modes ``phi_i = sin(m_i pi x/a) sin(n_i pi y/b)``, transverse
``w = sum_i q_i phi_i``):
  * curvature      ``kappa = [-w_xx, -w_yy, -2 w_xy]``  -> bending ``K``,
  * membrane strain ``eps = [w_x^2/2, w_y^2/2, w_x w_y]`` (restrained edges)
    -> membrane force ``N = A eps`` -> cubic modal force
    ``f_i = int N . d eps/d q_i``  with ``d eps/d q_i =
    [w_x phi_{i,x}, w_y phi_{i,y}, w_x phi_{i,y}+w_y phi_{i,x}]``.

Unlike the single-mode prototype, several modes are kept, so the SSM captures the
inter-modal **slaving** that shapes the real backbone.  ``to_system()`` returns a
``pyhb.MechanicalSystem`` consumed by :func:`nltopt.ssm_from_system`.
"""

from __future__ import annotations

import numpy as np

from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import GeneralNonlinearElement

from .laminate import Qbar, lamina_Q, RHO


def _mode_list(n_modes, a, b):
    """The ``n_modes`` lowest simply-supported modes (m,n), by m^2/a^2+n^2/b^2."""
    cand = [(m, n) for m in range(1, n_modes + 2) for n in range(1, n_modes + 2)]
    cand.sort(key=lambda mn: (mn[0] / a) ** 2 + (mn[1] / b) ** 2)
    return cand[:n_modes]


class VonKarmanPlate:
    """Simply-supported laminated von Karman plate, assumed-modes reduction."""

    def __init__(self, a, b, h, n_modes=6, nq=10, Q=None):
        self.a, self.b, self.h = float(a), float(b), float(h)
        self.Q = lamina_Q() if Q is None else Q
        self.modes = _mode_list(n_modes, a, b)
        self.nm = len(self.modes)
        self.CB, self.CM = h ** 3 / 12.0, h          # D=CB*Qbar, A=CM*Qbar (B=0)

        # Gauss-Legendre quadrature grid over [0,a] x [0,b]
        gx, wx = np.polynomial.legendre.leggauss(nq)
        gy, wy = np.polynomial.legendre.leggauss(nq)
        X = 0.5 * a * (gx + 1.0); Y = 0.5 * b * (gy + 1.0)
        WX = 0.5 * a * wx;        WY = 0.5 * b * wy
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        self.qx, self.qy = XX.ravel(), YY.ravel()
        self.qw = np.outer(WX, WY).ravel()           # (Nq,)
        self.Nq = self.qx.size

        # mode fields + derivatives at quadrature points
        self.PHI = np.empty((self.nm, self.Nq))
        self.PHIx = np.empty((self.nm, self.Nq))
        self.PHIy = np.empty((self.nm, self.Nq))
        self.KAP = np.empty((self.nm, 3, self.Nq))   # [-w_xx,-w_yy,-2w_xy]
        for i, (m, n) in enumerate(self.modes):
            kx, ky = m * np.pi / a, n * np.pi / b
            sx, sy = np.sin(kx * self.qx), np.sin(ky * self.qy)
            cx, cy = np.cos(kx * self.qx), np.cos(ky * self.qy)
            self.PHI[i] = sx * sy
            self.PHIx[i] = kx * cx * sy
            self.PHIy[i] = ky * sx * cy
            self.KAP[i, 0] = kx ** 2 * sx * sy       # -w_xx
            self.KAP[i, 1] = ky ** 2 * sx * sy       # -w_yy
            self.KAP[i, 2] = -2.0 * kx * ky * cx * cy  # -2 w_xy

        # modal mass (constant): M_ij = rho h int phi_i phi_j
        self.M = RHO * h * np.einsum("q,iq,jq->ij", self.qw, self.PHI, self.PHI)
        self.set_theta(np.zeros(self.Nq))

    # -- orientation field -------------------------------------------------- #
    def set_theta(self, theta_q):
        """Set the fiber angle at each quadrature point (rad); assemble D,A,K."""
        theta_q = np.broadcast_to(np.asarray(theta_q, float), (self.Nq,))
        self.theta_q = theta_q.copy()
        Dq = np.empty((self.Nq, 3, 3)); Aq = np.empty((self.Nq, 3, 3))
        for q in range(self.Nq):
            Qb = Qbar(theta_q[q], self.Q)
            Dq[q] = self.CB * Qb
            Aq[q] = self.CM * Qb
        self.Dq, self.Aq = Dq, Aq
        # K_ij = int kappa_i^T D kappa_j
        self.K = np.einsum("q,iaq,qab,jbq->ij", self.qw, self.KAP, Dq, self.KAP)
        return self

    def natural_frequencies(self):
        from scipy.linalg import eigh
        w2 = eigh(self.K, self.M, eigvals_only=True)
        return np.sqrt(np.clip(w2, 0.0, None))

    def midspan_mode(self):
        """Index of the mode with the largest mid-plate transverse value."""
        xc = np.array([0.5 * self.a]); yc = np.array([0.5 * self.b])
        vals = [abs(np.sin(m * np.pi * 0.5) * np.sin(n * np.pi * 0.5))
                for (m, n) in self.modes]
        return int(np.argmax(vals))

    # -- PyHB system -------------------------------------------------------- #
    def to_system(self):
        return MechanicalSystem(self.M, np.zeros_like(self.M), self.K,
                                [_VKPlateForce(self)])


class _VKPlateForce(GeneralNonlinearElement):
    """von Karman cubic membrane force in the modal basis (restrained edges)."""

    def __init__(self, plate: VonKarmanPlate):
        self.p = plate
        self.n = plate.nm

    def forces(self, q_time, u_time):
        p = self.p
        Ns = q_time.shape[0]
        f = np.zeros((Ns, self.n))
        Kt = np.zeros((Ns, self.n, self.n))
        qw, PHIx, PHIy, Aq = p.qw, p.PHIx, p.PHIy, p.Aq
        for s in range(Ns):
            q = q_time[s]
            wx = q @ PHIx                            # (Nq,)
            wy = q @ PHIy
            eps = np.stack([0.5 * wx ** 2, 0.5 * wy ** 2, wx * wy])   # (3,Nq)
            Nf = np.einsum("qab,bq->aq", Aq, eps)    # membrane force (3,Nq)
            # d eps / d q_i  = [wx PHIx_i, wy PHIy_i, wx PHIy_i + wy PHIx_i]
            dNx = PHIx; dNy = PHIy
            # f_i = sum_q qw ( Nf0 wx dNx_i + Nf1 wy dNy_i + Nf2 (wx dNy_i + wy dNx_i) )
            f[s] = (qw * Nf[0] * wx) @ dNx.T + (qw * Nf[1] * wy) @ dNy.T \
                + (qw * Nf[2]) @ (wx * dNy + wy * dNx).T
            # tangent: dep/dq_j gives B_j = [wx PHIx_j, wy PHIy_j, wx PHIy_j+wy PHIx_j]
            Bmat = np.stack([wx * PHIx, wy * PHIy, wx * PHIy + wy * PHIx])  # (3,nm,Nq)
            AB = np.einsum("qab,bjq->ajq", Aq, Bmat)                        # (3,nm,Nq)
            Kt[s] = np.einsum("q,aiq,ajq->ij", qw, Bmat, AB)               # material
            # geometric part: N acting on d2 eps / dq_i dq_j
            Kt[s] += (PHIx * (qw * Nf[0])) @ PHIx.T + (PHIy * (qw * Nf[1])) @ PHIy.T \
                + (PHIx * (qw * Nf[2])) @ PHIy.T + (PHIy * (qw * Nf[2])) @ PHIx.T
        return f, Kt, np.zeros((Ns, self.n, self.n))
