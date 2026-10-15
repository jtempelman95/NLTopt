r"""Two-dimensional spectral submanifold (SSM) at cubic order.

This is the *entry point* of the NLTopt program: an autonomous, two-dimensional
spectral submanifold attached to a single vibration mode, computed to cubic
order, from which we read the **backbone curve** (the frequency-amplitude
relation of a nonlinear normal mode) in physical coordinates.

It reproduces, at leading nonlinear order, the reduction used by Jain & Haller
(*Nonlinear Dyn.* 2022) and by Pozzi, Marconi, Jain, Li & Braghin (2025, the
paper this repo builds on) -- here written in a compact, verifiable form for a
2-D SSM, as the foundation on which the adjoint sensitivity w.r.t. composite
fiber-orientation design variables will be built.

Why "2-D"?  A *two-dimensional* SSM is parametrized by a single complex-
conjugate pair of reduced coordinates ``p = (p1, p2 = conj(p1))`` -- i.e. it is
attached to one oscillatory mode (its ``lambda, conj(lambda)`` eigenvalue pair).
This is the canonical, minimal SSM for a single nonlinear normal mode / backbone
curve, and is exactly the object the reference paper restricts to ("two-
dimensional SSMs up to arbitrary polynomial orders").  Higher-dimensional
(``2m``-D) SSMs are needed only for internally resonant mode groups (1:2, 1:3,
...) or multi-mode ROMs; a 2-D SSM assumes the master mode is *not* in low-order
internal resonance with a neighbour (extending the dimension is a roadmap item).

Method (autonomous first-order form, following the reference paper eqs 5-16)
---------------------------------------------------------------------------
The second-order system

    M x'' + C x' + K x + f_nl(x) = 0,   f_nl(x) = B2(x, x) + T3(x, x, x)

is written in first-order form ``B z' = A z + F(z)`` with ``z = [x; x']``,

    B = [[C, M], [M, 0]],   A = [[-K, 0], [0, M]],   F(z) = [-f_nl(x); 0].

A 2-D SSM tangent to the master mode ``(omega0, phi)`` is parametrized by two
complex-conjugate reduced coordinates ``p = (p1, p2)``, ``p2 = conj(p1)``:

    z = W(p) = sum_m  w_m p^m,    p' = R(p) = sum_m R_m p^m,   m = (m1, m2).

Substituting into the invariance equation ``B (dW/dp) R(p) = A W(p) + F(W(p))``
and collecting order by order gives, at each multi-index ``m``, the homological
equation

    (Lambda_m B - A) w_m + B [v1 R1_m + v2 R2_m] = G_m,

where ``Lambda_m = m1 lambda1 + m2 lambda2`` and ``G_m`` is the known
nonlinear forcing assembled from lower orders.  Non-resonant ``m`` set
``R_m = 0`` and solve for ``w_m`` directly; the near-resonant index ``(2,1)``
(``Lambda_(2,1) = lambda1``) carries the reduced cubic coefficient ``gamma``,
extracted by the adjoint/left-eigenvector solvability condition.  The backbone
follows from ``p1' = (lambda1 + gamma |p1|^2) p1``:

    Omega(rho)   = Im(lambda1) + Im(gamma) rho^2 / 4      (instantaneous freq.)
    alpha(rho)   = Re(lambda1) + Re(gamma) rho^2 / 4      (decay rate)

with reduced amplitude ``|p1| = rho / 2``.  Physical amplitudes are read back
through ``x = [I 0] W(p)``.

Validated (see ``tests/test_ssm.py``) against:
  * the analytic Duffing backbone  ``Omega = omega0 (1 + 3 knl A^2 / (8 omega0^2))``,
  * PyHB's EPMC nonlinear-mode continuation (independent full-order method),
    for a quadratically-coupled 2-DOF system and a von Karman beam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import solve, lstsq
from scipy.linalg import eig, eigh

from .nonlinearity import multilinear_forms


@dataclass
class SSM2D:
    """A cubic-order 2-D spectral submanifold and its backbone.

    Attributes
    ----------
    omega0, zeta0 : float
        Linear natural frequency and (viscous) damping ratio of the master mode.
    lambda1 : complex
        Master eigenvalue of the first-order system, ``-zeta0 omega0 + i omega_d``.
    gamma : complex
        Reduced cubic coefficient (the ``R1_(2,1)`` normal-form coefficient).
        ``Im(gamma) > 0`` is hardening, ``< 0`` softening.
    phi : ndarray
        Mass-normalized master mode shape (length ``n``).
    n : int
        Number of physical degrees of freedom.
    ref_dof : int
        Default DOF for physical-amplitude read-out.
    w : dict[tuple[int, int], ndarray]
        SSM coefficients ``w_m`` (length ``2n``, state space), for the
        independent multi-indices; conjugate partners are generated on demand.
    """

    omega0: float
    zeta0: float
    lambda1: complex
    gamma: complex
    phi: np.ndarray
    n: int
    ref_dof: int = 0
    w: dict = field(default_factory=dict)
    red: dict = field(default_factory=dict)   # {k: gamma_{2k+1}} reduced coeffs
    order: int = 3
    u1: np.ndarray = None                     # left (adjoint) eigenvector, u1* B v1 = 1

    # -- reduced-dynamics complex frequency on the orbit ------------------- #
    def _reduced(self, rho):
        r"""``lambda1 + sum_k gamma_{2k+1} (rho/2)^{2k}`` = d(ln p1)/dt on the orbit.

        Its imaginary part is the instantaneous backbone frequency, its real
        part the decay rate.  ``k=1`` is the cubic term (``gamma``), ``k=2`` the
        quintic term that lets the backbone turn from softening to hardening.
        """
        rho = np.asarray(rho, dtype=float)
        s = np.full(rho.shape, self.lambda1, dtype=complex)
        for k, c in self.red.items():
            s = s + c * (0.5 * rho) ** (2 * k)
        return s

    # -- damped forced response (leading-order SSM) ------------------------- #
    def forced_response(self, force, dof=None, n=400):
        r"""Leading-order SSM forced-response curve (FRC) for harmonic modal
        forcing ``force * cos(Omega t)`` on the physical DOFs (a damped system).

        Seeking a periodic response ``p1 = a e^{i Omega t}`` on the SSM reduces
        the forced dynamics to ``[i Omega - s(rho)] a = f1/2`` with the modal
        forcing ``f1 = u1^* [force; 0]``; hence the FRC
        ``[(Omega - omega(rho))^2 + alpha(rho)^2] rho^2 = |f1|^2``,
        whose two frequency branches close at the peak where ``Omega = omega(rho)``
        -- i.e. the resonance peak rides the backbone.  Requires a damped SSM
        (nonzero ``alpha``).

        Returns a dict with ``amplitude`` (physical, at ``dof``) and the two
        frequency branches ``omega_lo``/``omega_hi`` (NaN where the FRC does not
        exist), plus the backbone ``omega``.
        """
        if self.u1 is None:
            raise ValueError("forced_response needs u1 (rebuild the SSM)")
        dof = self.ref_dof if dof is None else dof
        force = np.atleast_1d(np.asarray(force, dtype=float))
        f1 = np.vdot(self.u1[:self.n], force)              # u1^* [force;0]
        alpha0 = abs(self.lambda1.real) + 1e-30
        rho = np.linspace(1e-9, 1.6 * abs(f1) / alpha0, n)
        s = self._reduced(rho)
        om, al = s.imag, s.real
        disc = np.abs(f1) ** 2 / rho ** 2 - al ** 2        # (Omega-omega)^2
        root = np.sqrt(np.clip(disc, 0.0, None))
        root[disc < 0] = np.nan
        amp = self.physical_amplitude(rho, dof=dof)
        return {"amplitude": np.atleast_1d(amp), "omega": om,
                "omega_lo": om - root, "omega_hi": om + root, "rho": rho}

    # -- hardening / softening sign ---------------------------------------- #
    @property
    def hardening(self) -> bool:
        return self.gamma.imag > 0.0

    def backbone_coeff(self) -> float:
        """First-order backbone curvature ``kappa`` in ``Omega ~ omega0 + kappa A^2``.

        Defined through the *physical* fundamental amplitude ``A`` at
        ``ref_dof`` (small-amplitude limit).  For a single-DOF Duffing
        oscillator this equals ``3 knl / (8 omega0)``.
        """
        # A ~ |phi[ref]| rho at leading order, so Omega = omega0 + Im(gamma)/4 rho^2
        # => kappa = Im(gamma) / (4 phi_ref^2)
        pr = self.phi[self.ref_dof]
        return self.gamma.imag / (4.0 * pr * pr)

    # -- evaluate the manifold: state z(p) on the reduced orbit ------------ #
    def _all_coeffs(self):
        """Return every ``w_m`` including conjugate partners ``w_(b,a)=conj w_(a,b)``."""
        full = {}
        for (a, b), wm in self.w.items():
            full[(a, b)] = wm
            if a != b:
                full[(b, a)] = np.conjugate(wm)
        return full

    def _x_of_theta(self, rho, thetas):
        """Physical displacement history ``x(theta)`` on the orbit of amplitude ``rho``.

        ``p1 = (rho/2) e^{i theta}``, ``p2 = conj(p1)``; returns real array
        ``(len(thetas), n)``.
        """
        r = 0.5 * rho
        p1 = r * np.exp(1j * thetas)
        p2 = np.conjugate(p1)
        z = np.zeros((thetas.size, 2 * self.n), dtype=complex)
        for (a, b), wm in self._all_coeffs().items():
            z += np.outer(p1 ** a * p2 ** b, wm)
        x = z[:, : self.n]
        return x.real

    def physical_amplitude(self, rho, dof=None, n_theta=256):
        """Fundamental-harmonic amplitude at ``dof`` for reduced amplitude ``rho``.

        Matches PyHB's ``harmonic_amplitude`` convention: ``sqrt(a1^2 + b1^2)``
        of the first Fourier harmonic of ``x_dof(theta)``.
        """
        dof = self.ref_dof if dof is None else dof
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        out = np.empty(rho.size)
        for i, r in enumerate(rho):
            xd = self._x_of_theta(r, thetas)[:, dof]
            a1 = 2.0 / n_theta * np.sum(xd * np.cos(thetas))
            b1 = 2.0 / n_theta * np.sum(xd * np.sin(thetas))
            out[i] = np.hypot(a1, b1)
        return out if out.size > 1 else float(out[0])

    def rms_amplitude(self, rho, dof=None, n_theta=256):
        """RMS amplitude of ``x_dof(theta)`` over one period for reduced amp ``rho``.

        This is the amplitude measure used by Pozzi et al. (2025) for the
        backbone target points.  For a pure fundamental it equals
        ``fundamental_amplitude / sqrt(2)``; it also captures the quadratic
        nonlinearity's DC drift and higher harmonics.
        """
        dof = self.ref_dof if dof is None else dof
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        out = np.empty(rho.size)
        for i, r in enumerate(rho):
            xd = self._x_of_theta(r, thetas)[:, dof]
            out[i] = np.sqrt(np.mean((xd - xd.mean()) ** 2))
        return out if out.size > 1 else float(out[0])

    def omega_at_amplitude(self, target, dof=None, kind="rms", rho_hi=None):
        """Backbone frequency ``Omega`` at a prescribed physical amplitude.

        Inverts the (monotone, at low amplitude) amplitude-vs-``rho`` relation
        for the reduced amplitude, then evaluates ``Omega(rho)``.  ``kind`` is
        ``"rms"`` (default) or ``"fundamental"``.

        Returns ``(Omega, rho)``.  Raises ``ValueError`` if ``target`` is not
        bracketed below ``rho_hi``.
        """
        from scipy.optimize import brentq
        dof = self.ref_dof if dof is None else dof
        amp = (self.rms_amplitude if kind == "rms" else self.physical_amplitude)
        if rho_hi is None:
            rho_hi = 4.0 * target / max(abs(self.phi[dof]), 1e-12)
        f = lambda r: amp(r, dof=dof) - target
        if f(rho_hi) < 0:                       # extend the bracket if needed
            rho_hi *= 4.0
            if f(rho_hi) < 0:
                raise ValueError("amplitude target not reachable below rho_hi")
        rho = brentq(f, 1e-9, rho_hi, xtol=1e-12, rtol=1e-10)
        omega = float(self._reduced(rho).imag)
        return omega, rho

    def backbone(self, rho, dof=None, n_theta=256):
        """Backbone curve over an array of reduced amplitudes ``rho``.

        Returns a dict with ``rho``, ``omega`` (instantaneous frequency),
        ``decay`` (modal decay rate), ``damping`` (amplitude-dependent damping
        ratio) and ``amplitude`` (physical fundamental amplitude at ``dof``).
        """
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        s = self._reduced(rho)
        omega = s.imag
        decay = s.real
        damping = -decay / np.hypot(omega, decay)
        amp = self.physical_amplitude(rho, dof=dof, n_theta=n_theta)
        amp = np.atleast_1d(amp)
        return {"rho": rho, "omega": omega, "decay": decay,
                "damping": damping, "amplitude": amp}


# --------------------------------------------------------------------------- #
# Core: assemble the cubic-order 2-D SSM
# --------------------------------------------------------------------------- #
def _lift(fx, n):
    """Lift a physical force ``-f`` into the state-space forcing ``[-f; 0]``."""
    g = np.zeros(2 * n, dtype=complex)
    g[:n] = fx
    return g


def compute_ssm_2d(M, C, K, bilinear=None, trilinear=None, mode=0, ref_dof=0,
                   order=3):
    """Compute the 2-D SSM of the ``mode``-th mode, to arbitrary polynomial ``order``.

    Solves the autonomous invariance equation order by order in the first-order
    state space.  At each multi-index ``m`` the homological equation

        (Lambda_m B - A) w_m + B [v1 R1_m + v2 R2_m] = G_m - known cross-terms

    is solved: near-resonant indices ``(k+1, k)`` carry the reduced-dynamics
    coefficients ``gamma_{2k+1} = R1_{(k+1,k)}`` (extracted by the left-
    eigenvector / adjoint projection ``u1^* .``), all others give ``w_m``
    directly.  The backbone reads ``d(ln p1)/dt = lambda1 + sum_k gamma_{2k+1}
    |p1|^{2k}``; ``k=1`` softens/hardens, ``k=2`` (order >= 5) lets a softening
    backbone turn back and harden.

    Parameters
    ----------
    M, C, K : ndarray, shape (n, n)
        Mass, (viscous) damping and stiffness matrices.
    bilinear, trilinear : callable or None
        Symmetric multilinear forms with ``f_nl(x) = bilinear(x,x) +
        trilinear(x,x,x)``; either may be ``None``.
    mode, ref_dof : int
    order : int
        Highest polynomial order retained (>= 3, made odd internally so the top
        reduced coefficient is complete).

    Returns
    -------
    SSM2D
    """
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    n = M.shape[0]
    order = max(3, int(order))

    # --- master mode of the underlying linear system ----------------------- #
    w2, Phi = eigh(K, M)
    omega = np.sqrt(np.clip(w2, 0.0, None))
    omega0 = float(omega[mode])
    phi = Phi[:, mode].copy()
    phi /= np.sqrt(phi @ M @ phi)
    zeta0 = float(phi @ C @ phi) / (2.0 * omega0) if omega0 > 0 else 0.0

    # --- first-order state-space pencil (A, B) ----------------------------- #
    Z = np.zeros((n, n))
    A = np.block([[-K, Z], [Z, M]])
    B = np.block([[C, M], [M, Z]])
    lam_target = -zeta0 * omega0 + 1j * omega0 * np.sqrt(max(1.0 - zeta0 ** 2, 0.0))

    symmetric = (np.allclose(K, K.T, rtol=0, atol=1e-12 * np.abs(K).max())
                 and np.allclose(C, C.T, rtol=0, atol=1e-12 * (np.abs(C).max() + 1e-300))
                 and np.allclose(M, M.T, rtol=0, atol=1e-12 * np.abs(M).max()))
    if symmetric:
        # For symmetric M, C, K the pencil (A, B) is complex-SYMMETRIC, so the left
        # eigenvector is the conjugate of the right one (u1 = conj v1) and the master
        # pair is obtained to machine precision by Rayleigh-quotient inverse iteration
        # seeded from the second-order mode.  (A dense QZ of the stiff first-order
        # pencil can lose many digits in the master eigenvector -- condition numbers
        # ~1e14 occur for laminated beams -- which corrupts gamma.)
        lam1 = lam_target
        v1 = np.concatenate([phi, lam1 * phi]).astype(complex)
        for _ in range(6):
            try:
                y = solve(A - lam1 * B, B @ v1)
            except np.linalg.LinAlgError:              # seed already an exact eigenpair
                break
            if not np.all(np.isfinite(y)):
                break
            v1 = y / np.linalg.norm(y)
            lam_new = (v1 @ A @ v1) / (v1 @ B @ v1)
            if abs(lam_new - lam1) <= 1e-14 * abs(lam1):
                lam1 = lam_new
                break
            lam1 = lam_new
        v1 = v1 / ((phi @ v1[:n]) / (phi @ phi))      # scale so v1[:n] == phi
        u1 = np.conjugate(v1)
        u1 = u1 / np.conjugate(np.vdot(u1, B @ v1))    # u1^* B v1 = 1
    else:
        lr, Vr = eig(A, B)
        ll, Vl = eig(A.T, B.T)
        i1 = int(np.argmin(np.abs(lr - lam_target)))
        lam1 = lr[i1]
        v1 = Vr[:, i1]
        v1 = v1 / ((phi @ v1[:n]) / (phi @ phi))      # scale so v1[:n] == phi
        j1 = int(np.argmin(np.abs(ll - np.conjugate(lam1))))
        u1 = Vl[:, j1]
        u1 = u1 / np.conjugate(np.vdot(u1, B @ v1))    # u1^* B v1 = 1
    v2 = np.conjugate(v1)
    lam2 = np.conjugate(lam1)

    Bv1 = B @ v1
    ez = [(1, 0), (0, 1)]

    def bil(a, b):
        return bilinear(a, b) if bilinear is not None else np.zeros(n)

    def tri(a, b, c):
        return trilinear(a, b, c) if trilinear is not None else np.zeros(n)

    # working coefficient tables (both conjugate partners kept during assembly)
    w = {(1, 0): v1, (0, 1): v2}
    R1 = {(1, 0): lam1}          # reduced dynamics of p1
    R2 = {(0, 1): lam2}          # reduced dynamics of p2 (= conjugate system)

    def wx(mi):
        return w[mi][:n]

    def forcing(m):
        """State-space nonlinear forcing ``G_m = [-(Q+C)(x)|_m ; 0]``."""
        m1, m2 = m
        fx = np.zeros(n, dtype=complex)
        if bilinear is not None:                       # quadratic convolution
            for a1 in range(m1 + 1):
                for a2 in range(m2 + 1):
                    if a1 + a2 == 0:
                        continue
                    b1, b2 = m1 - a1, m2 - a2
                    if b1 + b2 == 0:
                        continue
                    fx += bil(wx((a1, a2)), wx((b1, b2)))
        if trilinear is not None:                      # cubic convolution
            for a1 in range(m1 + 1):
                for a2 in range(m2 + 1):
                    if a1 + a2 == 0:
                        continue
                    for b1 in range(m1 - a1 + 1):
                        for b2 in range(m2 - a2 + 1):
                            if b1 + b2 == 0:
                                continue
                            c1, c2 = m1 - a1 - b1, m2 - a2 - b2
                            if c1 < 0 or c2 < 0 or c1 + c2 == 0:
                                continue
                            fx += tri(wx((a1, a2)), wx((b1, b2)), wx((c1, c2)))
        return _lift(-fx, n)

    def known_lhs(m):
        """``B * sum a_j w_a R^j_b`` over already-known nonlinear reduced terms."""
        m1, m2 = m
        tot = np.zeros(2 * n, dtype=complex)
        for table, j in ((R1, 1), (R2, 2)):
            ej = ez[j - 1]
            for b, Rval in table.items():
                if b[0] + b[1] < 3 or Rval == 0:       # only nonlinear R terms
                    continue
                a = (m1 + ej[0] - b[0], m2 + ej[1] - b[1])
                if a[0] < 0 or a[1] < 0 or a not in w:
                    continue
                aj = a[j - 1]
                if aj:
                    tot += aj * w[a] * Rval
        return B @ tot

    for q in range(2, order + 1):
        for m1 in range(q, (q - 1) // 2, -1):          # only m1 >= m2 (>, and == when q even)
            m2 = q - m1
            if m1 < m2:
                continue
            m = (m1, m2)
            RHS = forcing(m) - known_lhs(m)
            Lam = m1 * lam1 + m2 * lam2
            if m1 - m2 == 1:                            # near-resonant -> reduced dyn.
                R1m = np.vdot(u1, RHS)                  # u1^* RHS
                R1[m] = R1m
                rhs = RHS - Bv1 * R1m
                if abs(Lam - lam1) < 1e-9:              # exact resonance (undamped)
                    wm, *_ = lstsq(Lam * B - A, rhs, rcond=None)
                    wm = wm - np.vdot(u1, B @ wm) * v1
                else:
                    wm = solve(Lam * B - A, rhs)
                w[m] = wm
            else:
                w[m] = solve(Lam * B - A, RHS)
            if m1 != m2:                                # conjugate partner
                w[(m2, m1)] = np.conjugate(w[m])
                if m in R1:
                    R2[(m2, m1)] = np.conjugate(R1[m])

    red = {kk: R1[(kk + 1, kk)] for kk in range(1, (order + 1) // 2)
           if (kk + 1, kk) in R1}
    gamma = red.get(1, 0j)
    w_canon = {mi: wm for mi, wm in w.items() if mi[0] >= mi[1]}
    return SSM2D(omega0=omega0, zeta0=zeta0, lambda1=lam1, gamma=gamma,
                 phi=phi, n=n, ref_dof=ref_dof, w=w_canon, red=red, order=order,
                 u1=u1)


# --------------------------------------------------------------------------- #
# Convenience: build straight from a PyHB MechanicalSystem
# --------------------------------------------------------------------------- #
def ssm_from_system(system, mode=0, ref_dof=0, order=3):
    """Build the 2-D SSM (to polynomial ``order``) from a :class:`pyhb.MechanicalSystem`.

    The nonlinearity is read off the system's elements via
    :func:`nltopt.nonlinearity.multilinear_forms`, guaranteeing consistency with
    PyHB's harmonic-balance / EPMC model.  ``order=5`` captures softening-to-
    hardening backbones; ``order=3`` is the (monotone) cubic default.
    """
    bilinear, trilinear = multilinear_forms(system)
    return compute_ssm_2d(system.M, system.D, system.K,
                          bilinear=bilinear, trilinear=trilinear,
                          mode=mode, ref_dof=ref_dof, order=order)
