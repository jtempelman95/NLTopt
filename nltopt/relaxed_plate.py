r"""Relaxed backbone coefficient of a simply-supported laminated von Karman plate,
for ARBITRARY per-point laminate stiffnesses -- the evaluator behind every
mechanism study of the paper (uniform angle, stacking sequence, steering, and
anisotropy dose-response all go through the same code).

Model (symmetric laminate, B = 0; mass per area rho*h fixed).  The transverse
fundamental is the lowest Ritz pair of the bending operator over a sine basis
{phi_k = sin(m pi x/a) sin(n pi y/b)},

    K_b(D) v = omega_1^2 M v,     v^T M v = 1,     w(x) = sum_k v_k phi_k(x),

so the mode ADAPTS to the stiffness field (no fixed-shape assumption).  Its
von Karman stretch m = [w_x^2, w_y^2, 2 w_x w_y] (= 2 eps_NL) drives in-plane
displacements s = sum_j s_j psi_j (sine basis, immovable edges); statically
condensing them (the order-2 SSM slaving) gives the RELAXED cubic

    gamma = min_s 1/2 int (m - 2 B_L s)^T A (m - 2 B_L s)
          = 1/2 int eps_r^T A eps_r,     eps_r = m - 2 eps_u,   eps_u = B_L K_in^{-1} g,

with K_in = int B_L^T A B_L and g = 1/2 int B_L^T A m.  Equivalently
gamma = knl - 2 g^T K_in^{-1} g with knl = 1/2 int m^T A m (raw, restrained cubic).

The single-mode reduced dynamics is the Duffing  q'' + omega_1^2 q + gamma q^3 = 0,
whose backbone, written in the PHYSICAL mid-plate amplitude a = mid * q
(mid = w(a/2, b/2)), is

    Omega/omega_1 = 1 + (3/8) J a^2,     J = gamma / (omega_1^2 mid^2)   [1/m^2].

J is the design quantity ("frequency-normalised amplitude detuning").  The two
energy-density fields that the frequency and the hardening integrate are

    omega_1^2 = int e_K,  e_K = kappa_v^T D kappa_v;      gamma = int e_G,  e_G = 1/2 eps_r^T A eps_r.

Rotation.  A laminate rotated by theta (fibres turned through theta) transforms
as  C(theta) = T(theta) C_0 T(theta)^T  (engineering-shear Voigt), which for an
orthotropic C_0 is the classical Qbar(theta).  ``rotate`` handles a general
(e.g. stacked, D16 != 0) C_0.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .laminate import lamina_Q, RHO


# --------------------------------------------------------------------------- #
#  laminate algebra (vectorized)                                              #
# --------------------------------------------------------------------------- #
def Tmat(theta):
    """(..., 3, 3) rotation T(theta) with C(theta) = T C0 T^T (engineering shear)."""
    th = np.asarray(theta, float)
    c, s = np.cos(th), np.sin(th)
    T = np.empty(th.shape + (3, 3))
    T[..., 0, 0] = c * c; T[..., 0, 1] = s * s; T[..., 0, 2] = -2 * s * c
    T[..., 1, 0] = s * s; T[..., 1, 1] = c * c; T[..., 1, 2] = 2 * s * c
    T[..., 2, 0] = s * c; T[..., 2, 1] = -s * c; T[..., 2, 2] = c * c - s * s
    return T


def dTmat(theta):
    th = np.asarray(theta, float)
    c, s = np.cos(th), np.sin(th)
    dT = np.empty(th.shape + (3, 3))
    c2, s2 = c * c - s * s, 2 * s * c
    dT[..., 0, 0] = -s2; dT[..., 0, 1] = s2; dT[..., 0, 2] = -2 * c2
    dT[..., 1, 0] = s2; dT[..., 1, 1] = -s2; dT[..., 1, 2] = 2 * c2
    dT[..., 2, 0] = c2; dT[..., 2, 1] = -c2; dT[..., 2, 2] = -2 * s2
    return dT


def rotate(C0, theta):
    """C(theta) = T C0 T^T for every theta (returns (..., 3, 3))."""
    T = Tmat(theta)
    return np.einsum("...ab,bc,...dc->...ad", T, C0, T)


def drotate(C0, theta):
    """d/dtheta of rotate(C0, theta)."""
    T, dT = Tmat(theta), dTmat(theta)
    X = np.einsum("...ab,bc,...dc->...ad", dT, C0, T)
    return X + np.swapaxes(X, -1, -2)


def stack_AD(angles, h, Q=None):
    """Symmetric laminate A, D (3x3) of total thickness h; ``angles`` lists the
    plies from the MID-PLANE OUTWARD for one half (the stack is mirrored)."""
    Q = lamina_Q() if Q is None else Q
    ang = np.atleast_1d(np.asarray(angles, float))
    n = ang.size
    t = 0.5 * h / n                                   # ply thickness
    A = np.zeros((3, 3)); D = np.zeros((3, 3))
    for k, th in enumerate(ang):
        z0, z1 = k * t, (k + 1) * t                   # upper half; mirror doubles
        Qb = rotate(Q, th)
        A += 2.0 * Qb * (z1 - z0)
        D += 2.0 * Qb * (z1 ** 3 - z0 ** 3) / 3.0
    return A, D


def Q_family(s, Q=None):
    """One-parameter anisotropy family: Q(s) = Q_iso + s (Q_cfrp - Q_iso), with Q_iso
    the isotropic tensor sharing Q_cfrp's rotation invariants U1 and U5 (U4 = U1 - 2U5).
    s = 0 is isotropic (orientation is a dead variable), s = 1 is the CFRP lamina;
    the anisotropic Tsai-Pagano invariants U2, U3 scale exactly as s."""
    Q = lamina_Q() if Q is None else Q
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    U1 = (3 * Q11 + 3 * Q22 + 2 * Q12 + 4 * Q66) / 8.0
    U5 = (Q11 + Q22 - 2 * Q12 + 4 * Q66) / 8.0
    U4 = U1 - 2.0 * U5
    Qiso = np.array([[U1, U4, 0.0], [U4, U1, 0.0], [0.0, 0.0, U5]])
    return Qiso + s * (Q - Qiso)


def _beam_fn(bc, m, L, x):
    """1-D transverse trial function X_m and its first two derivatives."""
    k = m * np.pi / L
    if bc == "ss":
        s, c = np.sin(k * x), np.cos(k * x)
        return s, k * c, -k * k * s
    if bc == "clamped":                       # sin(pi x/L) sin(m pi x/L): X = X' = 0 at 0, L
        k1 = np.pi / L
        s1, c1, sm, cm = np.sin(k1 * x), np.cos(k1 * x), np.sin(k * x), np.cos(k * x)
        return (s1 * sm, k1 * c1 * sm + k * s1 * cm,
                -(k1 * k1 + k * k) * s1 * sm + 2 * k1 * k * c1 * cm)
    raise ValueError(bc)


# --------------------------------------------------------------------------- #
#  the plate                                                                  #
# --------------------------------------------------------------------------- #
class RelaxedPlate:
    """Adapted-mode, in-plane-relaxed backbone of a SS laminated plate.

    Parameters
    ----------
    a, b, h : plate dimensions (m).
    mmax : transverse basis = all (m, n) with m, n <= mmax.
    n_in : in-plane basis = sin(i pi x/a) sin(j pi y/b), i, j <= n_in, for u and v.
    nx, ny : midpoint quadrature grid (exact for the trigonometric integrands
        while the summed wavenumber stays below 2 nx).
    bc : transverse support, "ss" (simply supported, phi = sin sin; the paper's
        plate) or "clamped" (phi = sin(pi x/a) sin(m pi x/a) * (same in y), which
        satisfies w = dw/dn = 0 on every edge term by term).
    inplane : "immovable" (u = v = 0 on the edges; sine basis; the paper's plate) or
        "movable" (in-plane traction-free edges; Chebyshev polynomial basis of degree
        <= n_in without the constant -- the opposite extreme of edge restraint; the
        rigid in-plane rotation is in the basis, so K_in is solved in least squares).
    """

    def __init__(self, a=0.20, b=0.15, h=0.002, mmax=8, n_in=6, nx=96, ny=72, rho=RHO,
                 bc="ss", inplane="immovable"):
        self.a, self.b, self.h, self.rho = float(a), float(b), float(h), float(rho)
        xs = (np.arange(nx) + 0.5) * a / nx
        ys = (np.arange(ny) + 0.5) * b / ny
        X, Y = np.meshgrid(xs, ys, indexing="xy")          # (ny, nx), row-major in y
        self.nx, self.ny = nx, ny
        self.X, self.Y = X, Y
        self.qx, self.qy = X.ravel(), Y.ravel()
        self.Nq = self.qx.size
        self.w = np.full(self.Nq, a * b / self.Nq)

        self.bc, self.inplane = bc, inplane
        self.modes = [(m, n) for m in range(1, mmax + 1) for n in range(1, mmax + 1)]
        nw = self.nw = len(self.modes)
        self.PHI = np.empty((nw, self.Nq)); self.PHx = np.empty((nw, self.Nq))
        self.PHy = np.empty((nw, self.Nq)); self.KAP = np.empty((nw, 3, self.Nq))
        self.PHc = np.empty(nw)
        for k, (m, n) in enumerate(self.modes):
            X, X1, X2 = _beam_fn(bc, m, a, self.qx)
            Y, Y1, Y2 = _beam_fn(bc, n, b, self.qy)
            self.PHI[k] = X * Y; self.PHx[k] = X1 * Y; self.PHy[k] = X * Y1
            # kappa = [-w_xx, -w_yy, -2 w_xy]
            self.KAP[k, 0] = -X2 * Y
            self.KAP[k, 1] = -X * Y2
            self.KAP[k, 2] = -2 * X1 * Y1
            self.PHc[k] = _beam_fn(bc, m, a, np.array([a / 2]))[0][0] * _beam_fn(bc, n, b, np.array([b / 2]))[0][0]
        self.M = self.rho * self.h * np.einsum("q,iq,jq->ij", self.w, self.PHI, self.PHI)

        if inplane == "immovable":
            inm = [(i, j) for i in range(1, n_in + 1) for j in range(1, n_in + 1)]
        elif inplane == "movable":                  # Chebyshev T_i(xi) T_j(eta), degree <= n_in
            inm = [(i, j) for i in range(0, n_in + 1) for j in range(0, n_in + 1) if (i, j) != (0, 0)]
            from numpy.polynomial import chebyshev as _ch
            xi, eta = 2 * self.qx / a - 1, 2 * self.qy / b - 1
            def _T(i, t, d):
                c = np.zeros(i + 1); c[i] = 1.0
                return _ch.chebval(t, _ch.chebder(c, d) if d else c)
        else:
            raise ValueError(inplane)
        cols = []
        for comp in ("u", "v"):
            for (i, j) in inm:
                km, kn = i * np.pi / a, j * np.pi / b
                if inplane == "immovable":          # sin sin: zero on every edge
                    px = km * np.cos(km * self.qx) * np.sin(kn * self.qy)
                    py = kn * np.sin(km * self.qx) * np.cos(kn * self.qy)
                else:                               # polynomial: free edges
                    px = (2 / a) * _T(i, xi, 1) * _T(j, eta, 0)
                    py = (2 / b) * _T(i, xi, 0) * _T(j, eta, 1)
                z = np.zeros(self.Nq)
                cols.append(np.vstack([px, z, py]) if comp == "u" else np.vstack([z, py, px]))
        self.BL = np.stack(cols, axis=1)                    # (3, n_in_dof, Nq)
        self.nin = self.BL.shape[1]
        # moment tensors for the fast UNIFORM-laminate path (constant D, A):
        # K_b = sum_ab D_ab IK[a,b],  K_in = sum_ab A_ab IIN[a,b]
        self.IK = np.einsum("q,iaq,jbq->abij", self.w, self.KAP, self.KAP)
        self.IIN = np.einsum("q,aiq,bjq->abij", self.w, self.BL, self.BL)

    def _solve_in(self, Kin, g):
        if self.inplane == "immovable":
            return np.linalg.solve(Kin, g)
        return np.linalg.lstsq(Kin, g, rcond=1e-12)[0]     # rigid rotation: null space, g _|_ it

    def set_mass(self, rhoh=None):
        """Mass per unit area at every quadrature point (array of Nq, or None for the
        uniform rho*h).  Used only by the as-built (gap/overlap) evaluation; every
        design in the paper has the uniform mass."""
        rhoh = self.rho * self.h if rhoh is None else np.broadcast_to(rhoh, (self.Nq,))
        self.M = np.einsum("q,iq,jq->ij", self.w * rhoh, self.PHI, self.PHI)

    # ------------------------------------------------------------------ #
    def evaluate(self, Dq, Aq, full=True):
        """Forward model for per-point stiffnesses Dq, Aq of shape (Nq,3,3) or (3,3).

        Returns a dict with omega1, mid, gamma, knl, relax (= 2 g^T u*), J and the
        fields needed by ``gradient`` and by the mechanism analysis."""
        uniform = np.ndim(Dq) == 2 and np.ndim(Aq) == 2
        if uniform and not full:
            return self._evaluate_uniform(np.asarray(Dq), np.asarray(Aq))
        Dq = np.broadcast_to(Dq, (self.Nq, 3, 3)); Aq = np.broadcast_to(Aq, (self.Nq, 3, 3))
        w, KAP = self.w, self.KAP
        DK = np.einsum("qab,jbq->jaq", Dq, KAP)             # D kappa_j  (nw,3,Nq)
        Kb = np.einsum("q,iaq,jaq->ij", w, KAP, DK)
        lam, V = eigh(Kb, self.M)
        v, om2 = V[:, 0], lam[0]
        mid = float(v @ self.PHc)
        if mid < 0:                                         # sign convention: mid > 0
            v = -v; V[:, 0] = v; mid = -mid
        wx, wy = v @ self.PHx, v @ self.PHy
        m = np.vstack([wx * wx, wy * wy, 2 * wx * wy])      # 2 eps_NL   (3,Nq)
        Am = np.einsum("qab,bq->aq", Aq, m)
        knl = 0.5 * np.sum(w * np.sum(m * Am, axis=0))
        ABL = np.einsum("qab,bjq->ajq", Aq, self.BL)
        Kin = np.einsum("q,aiq,ajq->ij", w, self.BL, ABL)
        g = 0.5 * np.einsum("q,aiq,aq->i", w, self.BL, Am)
        ustar = self._solve_in(Kin, g)
        eps_u = np.einsum("aiq,i->aq", self.BL, ustar)
        relax = 2.0 * g @ ustar
        gamma = knl - relax
        J = gamma / (om2 * mid ** 2)
        out = dict(omega1=np.sqrt(om2), om2=om2, mid=mid, gamma=gamma, knl=knl,
                   relax=relax, J=J, v=v)
        if full:
            kv = np.einsum("k,kaq->aq", v, KAP)
            eps_r = m - 2.0 * eps_u
            out.update(lam=lam, V=V, wx=wx, wy=wy, m=m, kv=kv, eps_u=eps_u, eps_r=eps_r,
                       eK=np.einsum("aq,qab,bq->q", kv, Dq, kv),
                       eG=0.5 * np.einsum("aq,qab,bq->q", eps_r, Aq, eps_r),
                       Dq=Dq, Aq=Aq)
        return out

    def _evaluate_uniform(self, D, A):
        """Same model, constant D and A (uniform laminate): operators from the
        precomputed moment tensors, so a stacking-sequence scan is cheap."""
        Kb = np.einsum("ab,abij->ij", D, self.IK)
        lam, V = eigh(Kb, self.M, subset_by_index=[0, 0])
        v, om2 = V[:, 0], lam[0]
        mid = float(v @ self.PHc)
        if mid < 0:
            v = -v; mid = -mid
        wx, wy = v @ self.PHx, v @ self.PHy
        m = np.vstack([wx * wx, wy * wy, 2 * wx * wy])
        Am = A @ m
        knl = 0.5 * np.sum(self.w * np.sum(m * Am, axis=0))
        Kin = np.einsum("ab,abij->ij", A, self.IIN)
        g = 0.5 * np.einsum("aiq,aq->i", self.BL, self.w * Am)
        ustar = self._solve_in(Kin, g)
        relax = 2.0 * g @ ustar
        gamma = knl - relax
        return dict(omega1=np.sqrt(om2), om2=om2, mid=mid, gamma=gamma, knl=knl,
                    relax=relax, J=gamma / (om2 * mid ** 2), v=v)

    def detuning(self, st, a_over_h):
        """Relative backbone shift Omega/omega_1 - 1 at mid-plate amplitude a/h."""
        return 0.375 * st["J"] * (a_over_h * self.h) ** 2

    # ------------------------------------------------------------------ #
    def gradient(self, st, dDq, dAq):
        """Pointwise derivatives of (J, omega1, gamma) with respect to a scalar design
        field that changes D and A at each point by dDq, dAq (Nq,3,3) -- e.g. the
        rotation derivative for a steered field.  Mass is design-independent.

        dgamma_q = w_q [ 1/2 eps_r^T dA eps_r  +  kappa_beta^T dD kappa_v ]
        dom2_q   = w_q kappa_v^T dD kappa_v                         (Fox-Kapoor)
        dmid_q   = w_q kappa_mid^T dD kappa_v
        with the eigenvector adjoints kappa_beta = sum_{j>0} c_j kappa_{V_j},
        c_j = (dgamma/dv . V_j)/(lam_0 - lam_j), and likewise for mid.  The in-plane
        state needs no adjoint: gamma is stationary in s (envelope theorem)."""
        V, lam, w = st["V"], st["lam"], self.w
        Aq, eps_r, kv = st["Aq"], st["eps_r"], st["kv"]
        wx, wy = st["wx"], st["wy"]
        F = w * np.einsum("qab,bq->aq", Aq, eps_r)             # dgamma/dm  (3,Nq)
        dg_dv = (np.sum(F[0] * 2 * wx * self.PHx, axis=1)
                 + np.sum(F[1] * 2 * wy * self.PHy, axis=1)
                 + np.sum(F[2] * 2 * (wx * self.PHy + wy * self.PHx), axis=1))
        inv = 1.0 / (lam[0] - lam[1:])
        cb = V[:, 1:] @ ((dg_dv @ V[:, 1:]) * inv)
        cm = V[:, 1:] @ ((self.PHc @ V[:, 1:]) * inv)
        kb = np.einsum("k,kaq->aq", cb, self.KAP)
        km = np.einsum("k,kaq->aq", cm, self.KAP)
        dgam = w * (0.5 * np.einsum("aq,qab,bq->q", eps_r, dAq, eps_r)
                    + np.einsum("aq,qab,bq->q", kb, dDq, kv))
        dom2 = w * np.einsum("aq,qab,bq->q", kv, dDq, kv)
        dmid = w * np.einsum("aq,qab,bq->q", km, dDq, kv)
        om2, mid, gam = st["om2"], st["mid"], st["gamma"]
        dJ = dgam / (om2 * mid ** 2) - gam * dom2 / (om2 ** 2 * mid ** 2) \
            - 2 * gam * dmid / (om2 * mid ** 3)
        return dict(J=dJ, omega1=dom2 / (2 * np.sqrt(om2)), gamma=dgam, om2=dom2)


# --------------------------------------------------------------------------- #
#  steered single-material laminate: theta(x) rotates a fixed stack            #
# --------------------------------------------------------------------------- #
class SteeredPlate:
    """theta(x) rotates a fixed reference laminate (D0, A0) pointwise.  With the
    default single-ply reference D0 = (h^3/12) Q, A0 = h Q this is the model of the
    paper's steered designs; a stacked (D0, A0) gives the 'steered stack'."""

    def __init__(self, plate: RelaxedPlate, D0=None, A0=None, Q=None):
        self.p = plate
        Q = lamina_Q() if Q is None else Q
        self.D0 = plate.h ** 3 / 12.0 * Q if D0 is None else D0
        self.A0 = plate.h * Q if A0 is None else A0

    def forward(self, theta_q, full=True):
        st = self.p.evaluate(rotate(self.D0, theta_q), rotate(self.A0, theta_q), full=full)
        st["theta"] = np.broadcast_to(theta_q, (self.p.Nq,)).copy()
        return st

    def grad(self, st):
        th = st["theta"]
        return self.p.gradient(st, drotate(self.D0, th), drotate(self.A0, th))
