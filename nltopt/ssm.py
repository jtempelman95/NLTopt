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
        omega = self.lambda1.imag + self.gamma.imag * rho ** 2 / 4.0
        return omega, rho

    def backbone(self, rho, dof=None, n_theta=256):
        """Backbone curve over an array of reduced amplitudes ``rho``.

        Returns a dict with ``rho``, ``omega`` (instantaneous frequency),
        ``decay`` (modal decay rate), ``damping`` (amplitude-dependent damping
        ratio) and ``amplitude`` (physical fundamental amplitude at ``dof``).
        """
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        omega = self.lambda1.imag + self.gamma.imag * rho ** 2 / 4.0
        decay = self.lambda1.real + self.gamma.real * rho ** 2 / 4.0
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


def compute_ssm_2d(M, C, K, bilinear=None, trilinear=None, mode=0, ref_dof=0):
    """Compute the cubic-order 2-D SSM of the ``mode``-th mode.

    Parameters
    ----------
    M, C, K : ndarray, shape (n, n)
        Mass, (viscous) damping and stiffness matrices.
    bilinear, trilinear : callable or None
        Symmetric multilinear forms of the nonlinearity such that
        ``f_nl(x) = bilinear(x, x) + trilinear(x, x, x)``.  Either may be
        ``None`` (absent degree).  See :func:`nltopt.nonlinearity.multilinear_forms`.
    mode : int
        Index (ascending frequency) of the master mode.
    ref_dof : int
        Default DOF for physical-amplitude read-out.

    Returns
    -------
    SSM2D
    """
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    n = M.shape[0]

    # --- master mode of the underlying linear conservative system ---------- #
    w2, Phi = eigh(K, M)
    omega = np.sqrt(np.clip(w2, 0.0, None))
    omega0 = float(omega[mode])
    phi = Phi[:, mode].copy()
    phi /= np.sqrt(phi @ M @ phi)  # mass-normalize defensively
    zeta0 = float(phi @ C @ phi) / (2.0 * omega0) if omega0 > 0 else 0.0

    # --- first-order state-space pencil (A, B) ----------------------------- #
    Z = np.zeros((n, n))
    A = np.block([[-K, Z], [Z, M]])
    B = np.block([[C, M], [M, Z]])

    # target master eigenvalue (paper eq. 8)
    lam_target = -zeta0 * omega0 + 1j * omega0 * np.sqrt(max(1.0 - zeta0 ** 2, 0.0))

    lr, Vr = eig(A, B)                       # right: A v = lam B v
    ll, Vl = eig(A.T, B.T)                    # left : A^T u = lam B^T u

    i1 = int(np.argmin(np.abs(lr - lam_target)))
    lam1 = lr[i1]
    v1 = Vr[:, i1]
    # rescale so the displacement part of v1 equals phi (fixes the p-scaling)
    alpha = (phi @ v1[:n]) / (phi @ phi)
    v1 = v1 / alpha
    v2 = np.conjugate(v1)
    lam2 = np.conjugate(lam1)

    # left eigenvector paired with v1: eigenvalue conj(lam1); normalize u1^* B v1 = 1
    j1 = int(np.argmin(np.abs(ll - np.conjugate(lam1))))
    u1 = Vl[:, j1]
    u1 = u1 / np.conjugate(np.vdot(u1, B @ v1))
    # (np.vdot(u1, x) = u1^* . x)

    def solve_nonres(Lam_m, G):
        return solve(Lam_m * B - A, G)

    # order 1: the master eigenvector itself (its conjugate partner w_(0,1)=v2
    # is generated on demand).  Needed for the physical-amplitude read-out.
    w = {(1, 0): v1}

    # ----- order 2 (only if a quadratic part exists) ----------------------- #
    if bilinear is not None:
        B2pp = bilinear(phi, phi)
        w20 = solve_nonres(2 * lam1, _lift(-B2pp, n))
        w11 = solve_nonres(lam1 + lam2, _lift(-2.0 * B2pp, n))
        w[(2, 0)] = w20
        w[(1, 1)] = w11
        w20x, w11x = w20[:n], w11[:n]
    else:
        w20x = w11x = np.zeros(n)

    # ----- order 3 --------------------------------------------------------- #
    T3ppp = trilinear(phi, phi, phi) if trilinear is not None else np.zeros(n)

    # (3,0) non-resonant
    q30 = 2.0 * bilinear(phi, w20x) if bilinear is not None else 0.0
    G30 = _lift(-(q30 + T3ppp), n)
    w[(3, 0)] = solve_nonres(3 * lam1, G30)

    # (2,1) near-resonant -> carries gamma
    q21 = (2.0 * bilinear(phi, w11x) + 2.0 * bilinear(phi, w20x)
           if bilinear is not None else 0.0)
    G21 = _lift(-(q21 + 3.0 * T3ppp), n)

    gamma = np.vdot(u1, G21)                          # u1^* G21  (u1^* B v1 = 1)
    Lm = lam1 * B - A                                 # singular (rank 2n-1)
    rhs = G21 - (B @ v1) * gamma
    w21, *_ = lstsq(Lm, rhs, rcond=None)
    # remove the null-space component along v1 so u1^* B w21 = 0
    w21 = w21 - np.vdot(u1, B @ w21) * v1
    w[(2, 1)] = w21

    return SSM2D(omega0=omega0, zeta0=zeta0, lambda1=lam1, gamma=gamma,
                 phi=phi, n=n, ref_dof=ref_dof, w=w)


# --------------------------------------------------------------------------- #
# Convenience: build straight from a PyHB MechanicalSystem
# --------------------------------------------------------------------------- #
def ssm_from_system(system, mode=0, ref_dof=0):
    """Build the cubic-order 2-D SSM directly from a :class:`pyhb.MechanicalSystem`.

    The nonlinearity is read off the system's elements via
    :func:`nltopt.nonlinearity.multilinear_forms`, guaranteeing consistency with
    PyHB's harmonic-balance / EPMC model.
    """
    bilinear, trilinear = multilinear_forms(system)
    return compute_ssm_2d(system.M, system.D, system.K,
                          bilinear=bilinear, trilinear=trilinear,
                          mode=mode, ref_dof=ref_dof)
