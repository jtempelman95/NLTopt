"""Verification of the relaxed-backbone evaluators used by every mechanism study."""
import numpy as np
import pytest

from nltopt.laminate import Qbar, lamina_Q
from nltopt.relaxed_plate import (RelaxedPlate, SteeredPlate, rotate, drotate,
                                  stack_AD, Q_family)
from nltopt.relaxed_beam import RelaxedBeam

Q = lamina_Q()


@pytest.fixture(scope="module")
def plate():
    return RelaxedPlate(mmax=5, n_in=4, nx=64, ny=48)


def test_rotate_matches_classical_qbar():
    for th in (-1.2, 0.3, 0.9):
        assert np.allclose(rotate(Q, th), Qbar(th, Q), rtol=0, atol=1e-6 * Q.max())


def test_drotate_fd():
    th, e = 0.37, 1e-6
    fd = (rotate(Q, th + e) - rotate(Q, th - e)) / (2 * e)
    assert np.allclose(drotate(Q, th), fd, atol=1e-6 * Q.max())


def test_local_fourier_form():
    """s^T Qbar(theta) s = U1(e1^2+e2^2) + 2U4 e1e2 + U2(e1^2-e2^2)cos2psi + U3(e1-e2)^2 cos4psi."""
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    U1 = (3 * Q11 + 3 * Q22 + 2 * Q12 + 4 * Q66) / 8
    U2 = (Q11 - Q22) / 2
    U3 = (Q11 + Q22 - 2 * Q12 - 4 * Q66) / 8
    U4 = (Q11 + Q22 + 6 * Q12 - 4 * Q66) / 8
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.standard_normal(3)
        T = np.array([[s[0], s[2] / 2], [s[2] / 2, s[1]]])     # tensor strain
        ev, evec = np.linalg.eigh(T)
        e1, e2 = ev[1], ev[0]
        phi = np.arctan2(evec[1, 1], evec[0, 1])
        for th in (0.1, 0.8, -0.6):
            lhs = s @ rotate(Q, th) @ s
            psi = th - phi
            rhs = (U1 * (e1 ** 2 + e2 ** 2) + 2 * U4 * e1 * e2
                   + U2 * (e1 ** 2 - e2 ** 2) * np.cos(2 * psi) + U3 * (e1 - e2) ** 2 * np.cos(4 * psi))
            assert abs(lhs - rhs) < 1e-9 * abs(Q11) * (s @ s)


def test_isotropy_annihilates_orientation(plate):
    sp = SteeredPlate(plate, D0=plate.h ** 3 / 12 * Q_family(0.0), A0=plate.h * Q_family(0.0))
    th = 0.5 + 0.4 * np.sin(np.pi * plate.qx / plate.a)
    st = sp.forward(th)
    g = sp.grad(st)
    assert np.abs(g["J"]).max() < 1e-10 * abs(st["J"])
    assert abs(sp.forward(0.0, full=False)["J"] / st["J"] - 1) < 1e-10


def test_gradient_fd(plate):
    sp = SteeredPlate(plate)
    th = 0.4 + 0.5 * np.sin(np.pi * plate.qx / plate.a) * np.sin(np.pi * plate.qy / plate.b)
    st = sp.forward(th); g = sp.grad(st)
    d = np.sin(2 * np.pi * plate.qx / plate.a + 0.3) * np.cos(3 * np.pi * plate.qy / plate.b)
    e = 1e-6
    p, m = sp.forward(th + e * d, full=False), sp.forward(th - e * d, full=False)
    for k in ("J", "omega1", "gamma"):
        fd = (p[k] - m[k]) / (2 * e)
        assert abs(np.sum(g[k] * d) - fd) < 1e-5 * abs(fd)


def test_uniform_fast_path_equals_general(plate):
    A, D = stack_AD(np.deg2rad([45, -45, 90, 0]), plate.h)
    a = plate.evaluate(D, A, full=False)
    b = plate.evaluate(D, A, full=True)
    for k in ("J", "omega1", "gamma", "knl"):
        assert abs(a[k] / b[k] - 1) < 1e-10


def test_relaxed_gamma_is_variational(plate):
    """gamma = knl - 2 g^T K^-1 g equals 1/2 int eps_r^T A eps_r (eps_r = m - 2 eps_u)."""
    st = SteeredPlate(plate).forward(0.3 + 0.2 * plate.qy / plate.b)
    assert abs(np.sum(plate.w * st["eG"]) / st["gamma"] - 1) < 1e-10
    assert abs(np.sum(plate.w * st["eK"]) / st["om2"] - 1) < 1e-10


def test_ud_matches_assumed_modes_paper_value(plate):
    st = SteeredPlate(plate).forward(0.0, full=False)
    assert abs(st["omega1"] / (2 * np.pi) - 256.3) < 0.1
    assert abs(100 * plate.detuning(st, 0.4) - 10.96) < 0.02


def test_beam_uniform_invariance_and_harmonic_identity():
    rb = RelaxedBeam(60)
    u = rb.evaluate(np.full(rb.ne, Qbar(0.0)[0, 0]))
    for t in (0.3, 0.9):
        s = rb.evaluate(np.full(rb.ne, Qbar(t)[0, 0]))
        assert abs(s["J"] / u["J"] - 1) < 1e-8                  # single-ply uniform: J invariant
    E = Qbar(0.0)[0, 0] * (1 - 0.8 * np.exp(-((rb.xe / rb.L - 0.25) / 0.05) ** 2))
    s = rb.evaluate(E)
    assert abs(s["gamma"] / s["gamma_harmonic"] - 1) < 0.02    # mesh-converging identity


def test_ssm_scale_invariance_regression():
    """A uniformly rotated single-ply beam is an exact rescaling of the UD beam, so
    gamma/omega must be invariant.  The QZ-based master eigenvector used to lose
    accuracy at 45 deg (eigenvector condition ~1e14) and returned a 10x wrong gamma."""
    from nltopt.laminated_beam import LaminatedBeam
    from nltopt import ssm_from_system
    vals = []
    for t in (0.0, 45.0):
        b = LaminatedBeam(20, 0.3, 0.002, 0.02)
        th = np.full(20, np.deg2rad(t))
        b.set_ply_angles(np.vstack([th, th]))
        s = ssm_from_system(b.to_system(), mode=0, ref_dof=b.midspan_w_dof(), order=3)
        vals.append(s.gamma.imag / s.omega0)
        assert abs(s.gamma.real) < 1e-8 * abs(s.gamma.imag)
    assert abs(vals[1] / vals[0] - 1) < 1e-8
