r"""Multi-mode von Karman composite plate (assumed-modes / Galerkin).

Simply-supported rectangular laminated plate reduced onto assumed modes, with the
geometrically-nonlinear von Karman membrane stretching.  The fiber-orientation
field ``theta(x,y)`` sets the per-point bending/membrane stiffness ``D(theta)``,
``A(theta)`` (CLT, symmetric layup so ``B = 0``); everything is assembled by
Gauss quadrature so a spatially-varying tow layout is exact.

Kinematics.  Transverse ``w = sum_i q^w_i phi_i`` (``phi_i = sin(m pi x/a)
sin(n pi y/b)``); in-plane ``u = sum q^u_j psi^u_j``, ``v = sum q^v_k psi^v_k``
(``sin(i pi x/a) sin(j pi y/b)``, immovable edges).  The membrane strain

    eps = eps_L(u,v) + eps_NL(w),
    eps_L = [u_x, v_y, u_y+v_x],   eps_NL = [w_x^2/2, w_y^2/2, w_x w_y],

and curvature ``kappa = [-w_xx, -w_yy, -2 w_xy]`` give the energy

    U = 1/2 int kappa^T D kappa  +  1/2 int eps^T A eps.

Expanding the membrane term,
    1/2 int eps_L^T A eps_L      (in-plane linear stiffness K_mem),
    + int eps_L^T A eps_NL       (QUADRATIC u<->w^2 coupling),
    + 1/2 int eps_NL^T A eps_NL  (CUBIC w^4 -> hardening).

With ``n_in = 0`` the in-plane DOF are dropped (edges fully restrained) and the
model is purely cubic -- an upper bound on the hardening.  With ``n_in > 0`` the
high-frequency in-plane modes are kept; the SSM slaves them to ``w^2`` through the
quadratic coupling, and that membrane **relaxation reduces the cubic hardening**
-- the physically correct backbone.

``to_system()`` returns a ``pyhb.MechanicalSystem`` for :func:`nltopt.ssm_from_system`.
"""

from __future__ import annotations

import numpy as np

from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import GeneralNonlinearElement

from .laminate import Qbar, lamina_Q, RHO


def _mode_list(n_modes, a, b):
    cand = [(m, n) for m in range(1, n_modes + 2) for n in range(1, n_modes + 2)]
    cand.sort(key=lambda mn: (mn[0] / a) ** 2 + (mn[1] / b) ** 2)
    return cand[:n_modes]


class VonKarmanPlate:
    """Simply-supported laminated von Karman plate, assumed-modes reduction.

    Parameters
    ----------
    n_modes : int
        transverse (bending) modes retained.
    n_in : int
        in-plane modes per direction (``i, j = 1..n_in`` for both ``u`` and
        ``v``); ``0`` = restrained-membrane (immovable-edge upper bound).
    """

    def __init__(self, a, b, h, n_modes=6, n_in=4, nq=12, Q=None):
        self.a, self.b, self.h = float(a), float(b), float(h)
        self.Q = lamina_Q() if Q is None else Q
        self.CB, self.CM = h ** 3 / 12.0, h
        self.w_modes = _mode_list(n_modes, a, b)
        self.nw = len(self.w_modes)
        self.in_modes = [(i, j) for i in range(1, n_in + 1) for j in range(1, n_in + 1)]
        self.nu = self.nv = len(self.in_modes)
        self.ndof = self.nw + self.nu + self.nv

        gx, wx = np.polynomial.legendre.leggauss(nq)
        gy, wy = np.polynomial.legendre.leggauss(nq)
        X = 0.5 * a * (gx + 1.0); Y = 0.5 * b * (gy + 1.0)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        self.qx, self.qy = XX.ravel(), YY.ravel()
        self.qw = np.outer(0.5 * a * wx, 0.5 * b * wy).ravel()
        self.Nq = self.qx.size

        # transverse mode fields
        self.PHI = np.empty((self.nw, self.Nq))
        self.PHIx = np.empty((self.nw, self.Nq))
        self.PHIy = np.empty((self.nw, self.Nq))
        self.KAP = np.empty((self.nw, 3, self.Nq))
        for i, (m, n) in enumerate(self.w_modes):
            kx, ky = m * np.pi / a, n * np.pi / b
            sx, sy = np.sin(kx * self.qx), np.sin(ky * self.qy)
            cx, cy = np.cos(kx * self.qx), np.cos(ky * self.qy)
            self.PHI[i], self.PHIx[i], self.PHIy[i] = sx*sy, kx*cx*sy, ky*sx*cy
            self.KAP[i, 0] = kx**2 * sx*sy
            self.KAP[i, 1] = ky**2 * sx*sy
            self.KAP[i, 2] = -2.0*kx*ky*cx*cy

        # in-plane mode fields (same basis for u and v)
        self.PSI = np.empty((self.nu, self.Nq))
        self.PSIx = np.empty((self.nu, self.Nq))
        self.PSIy = np.empty((self.nu, self.Nq))
        for i, (mi, ni) in enumerate(self.in_modes):
            kx, ky = mi * np.pi / a, ni * np.pi / b
            sx, sy = np.sin(kx * self.qx), np.sin(ky * self.qy)
            cx, cy = np.cos(kx * self.qx), np.cos(ky * self.qy)
            self.PSI[i], self.PSIx[i], self.PSIy[i] = sx*sy, kx*cx*sy, ky*sx*cy

        # structural density (SIMP-weighted stiffness, linear mass); default solid
        self.dens_E = np.ones(self.Nq)
        self.dens_m = np.ones(self.Nq)
        self._assemble_mass()
        self.set_theta(np.zeros(self.Nq))

    def _assemble_mass(self):
        """(Density-weighted) modal mass: transverse + in-plane translational inertia."""
        m_w = RHO * self.h * np.einsum("q,q,iq,jq->ij", self.dens_m, self.qw, self.PHI, self.PHI)
        m_in = RHO * self.h * np.einsum("q,q,iq,jq->ij", self.dens_m, self.qw, self.PSI, self.PSI)
        self.M = np.zeros((self.ndof, self.ndof))
        self.M[:self.nw, :self.nw] = m_w
        s = self.nw
        self.M[s:s+self.nu, s:s+self.nu] = m_in
        self.M[s+self.nu:, s+self.nu:] = m_in

    def set_density(self, rho_q, penal=3.0, eps=1e-4):
        """SIMP-weight the plate by a per-quadrature-point density in [0,1]:
        stiffness x (eps+(1-eps)rho^penal), mass x rho.  The multi-mode
        eigenproblem then sees the perforated structure, so the master mode
        ADAPTS to the topology (no fixed-mode assumption).  Reassembles M, K."""
        rho = np.clip(np.broadcast_to(np.asarray(rho_q, float), (self.Nq,)), 0.0, 1.0)
        self.dens_E = eps + (1.0 - eps) * rho ** penal
        self.dens_m = rho.copy()
        self._assemble_mass()
        self.set_theta(self.theta_q)
        return self

    # slices
    @property
    def _sw(self): return slice(0, self.nw)
    @property
    def _su(self): return slice(self.nw, self.nw + self.nu)
    @property
    def _sv(self): return slice(self.nw + self.nu, self.ndof)

    def set_theta(self, theta_q):
        theta_q = np.broadcast_to(np.asarray(theta_q, float), (self.Nq,))
        self.theta_q = theta_q.copy()
        Dq = np.empty((self.Nq, 3, 3)); Aq = np.empty((self.Nq, 3, 3))
        for q in range(self.Nq):
            Qb = Qbar(theta_q[q], self.Q)
            Dq[q] = self.dens_E[q] * self.CB * Qb; Aq[q] = self.dens_E[q] * self.CM * Qb
        self.Dq, self.Aq = Dq, Aq
        # bending stiffness (w-w block)
        Kb = np.einsum("q,iaq,qab,jbq->ij", self.qw, self.KAP, Dq, self.KAP)
        # linear membrane stiffness (in-plane) from eps_L^T A eps_L
        self._force = _VKPlateForce(self)
        Kmem = self._force.membrane_tangent(np.zeros(self.ndof))   # linear part
        self.K = Kmem.copy()
        self.K[self._sw, self._sw] += Kb
        self._Kmem = Kmem
        return self

    def natural_frequencies(self):
        from scipy.linalg import eigh
        return np.sqrt(np.clip(eigh(self.K, self.M, eigvals_only=True), 0.0, None))

    def midspan_mode(self):
        vals = [abs(np.sin(m*np.pi*0.5) * np.sin(n*np.pi*0.5)) for (m, n) in self.w_modes]
        return int(np.argmax(vals))

    def to_system(self):
        return MechanicalSystem(self.M, np.zeros_like(self.M), self.K, [self._force])


class _VKPlateForce(GeneralNonlinearElement):
    """von Karman membrane force minus its linear (in-plane) part; modal basis."""

    def __init__(self, plate: VonKarmanPlate):
        self.p = plate
        self.n = plate.ndof

    # -- shared kernel: full membrane internal force + tangent -------------- #
    def _internal(self, q):
        p = self.p
        qw, PHIx, PHIy, PSIx, PSIy, Aq = p.qw, p.PHIx, p.PHIy, p.PSIx, p.PSIy, p.Aq
        wx = q[p._sw] @ PHIx; wy = q[p._sw] @ PHIy
        ux = q[p._su] @ PSIx; uy = q[p._su] @ PSIy
        vx = q[p._sv] @ PSIx; vy = q[p._sv] @ PSIy
        eps = np.stack([ux + 0.5*wx**2, vy + 0.5*wy**2, uy + vx + wx*wy])   # (3,Nq)
        Nf = np.einsum("qab,bq->aq", Aq, eps)                              # (3,Nq)
        f = np.zeros(self.n)
        # d eps / d q^w_i = [wx PHIx_i, wy PHIy_i, wx PHIy_i + wy PHIx_i]
        f[p._sw] = (qw*Nf[0]*wx) @ PHIx.T + (qw*Nf[1]*wy) @ PHIy.T \
            + (qw*Nf[2]) @ (wx*PHIy + wy*PHIx).T
        # d eps / d q^u_j = [PSIx_j, 0, PSIy_j] ;  d eps / d q^v_k = [0, PSIy_k, PSIx_k]
        f[p._su] = (qw*Nf[0]) @ PSIx.T + (qw*Nf[2]) @ PSIy.T
        f[p._sv] = (qw*Nf[1]) @ PSIy.T + (qw*Nf[2]) @ PSIx.T
        return f, wx, wy, Nf

    def membrane_tangent(self, q):
        """Consistent tangent of the full membrane internal force at ``q``."""
        p = self.p
        qw, PHIx, PHIy, PSIx, PSIy, Aq = p.qw, p.PHIx, p.PHIy, p.PSIx, p.PSIy, p.Aq
        _, wx, wy, Nf = self._internal(q)
        Z = np.zeros((1, p.Nq))
        # strain-displacement B (3, ndof, Nq): columns d eps / d q
        Bw = np.stack([wx*PHIx, wy*PHIy, wx*PHIy + wy*PHIx])                # (3,nw,Nq)
        Bu = np.stack([PSIx, np.zeros_like(PSIx), PSIy])                    # (3,nu,Nq)
        Bv = np.stack([np.zeros_like(PSIx), PSIy, PSIx])                    # (3,nv,Nq)
        B = np.concatenate([Bw, Bu, Bv], axis=1)                           # (3,ndof,Nq)
        AB = np.einsum("qab,bjq->ajq", Aq, B)
        Kt = np.einsum("q,aiq,ajq->ij", qw, B, AB)                         # material
        # geometric: N acting on d2 eps / d q^w_i d q^w_j (only w-w block)
        gww = (PHIx*(qw*Nf[0])) @ PHIx.T + (PHIy*(qw*Nf[1])) @ PHIy.T \
            + (PHIx*(qw*Nf[2])) @ PHIy.T + (PHIy*(qw*Nf[2])) @ PHIx.T
        Kt[p._sw, p._sw] += gww
        return Kt

    def forces(self, q_time, u_time):
        p = self.p
        Ns = q_time.shape[0]
        f = np.zeros((Ns, self.n)); Kt = np.zeros((Ns, self.n, self.n))
        Kmem = getattr(p, "_Kmem", None)
        if Kmem is None:
            Kmem = self.membrane_tangent(np.zeros(self.n))
        for s in range(Ns):
            q = q_time[s]
            fi, *_ = self._internal(q)
            f[s] = fi - Kmem @ q                       # subtract linear membrane part
            Kt[s] = self.membrane_tangent(q) - Kmem
        return f, Kt, np.zeros((Ns, self.n, self.n))
