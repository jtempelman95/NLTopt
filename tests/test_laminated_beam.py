"""Fiber-steered laminated von Karman beam FE."""

import numpy as np
import pytest

from nltopt.laminated_beam import LaminatedBeam
from nltopt import ssm_from_system

d2r = np.deg2rad


def _beam(ne=8):
    return LaminatedBeam(ne, 0.30, 0.002, 0.02, base_angle=0.0)


def test_nonlinear_tangent_matches_fd():
    """The nonlinear element's tangent equals the FD of its force (needed for EPMC)."""
    beam = _beam()
    beam.set_angles(d2r(np.linspace(0, 50, beam.n_elem)))
    el = beam.to_system().nonlinear_elements[0]
    nf = beam.free.size
    q = 1e-4 * np.random.default_rng(0).standard_normal(nf)
    f0, Kt, _ = el.forces(q[None, :], np.zeros((1, nf)))
    Kt = Kt[0]
    eps = 1e-8
    for i in range(nf):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        fp = el.forces(qp[None, :], np.zeros((1, nf)))[0][0]
        fm = el.forces(qm[None, :], np.zeros((1, nf)))[0][0]
        col = (fp - fm) / (2 * eps)
        assert np.allclose(col, Kt[:, i], rtol=1e-4, atol=1e-2 * np.max(np.abs(Kt)) + 1e-6)


def test_symmetric_layup_is_hardening_straight_beam():
    beam = _beam()
    beam.set_angles(np.zeros(beam.n_elem))          # [0/0] symmetric -> B = 0
    assert np.max(np.abs(beam.B)) < 1e-6 * np.max(np.abs(beam.A))
    ssm = ssm_from_system(beam.to_system(), mode=0, ref_dof=beam.midspan_w_dof(), order=3)
    assert ssm.gamma.imag > 0                        # von Karman beam hardens
    # first-mode frequency matches Euler-Bernoulli 22.37 sqrt(EI/(rhoA L^4))
    from nltopt.laminate import beam_abd
    A11, B11, D11, rhoA = beam_abd([0.0, 0.0], 0.002, 0.02)
    f_eb = (4.73 ** 2) * np.sqrt(D11 / (rhoA * 0.30 ** 4)) / (2 * np.pi)
    assert ssm.omega0 / (2 * np.pi) == pytest.approx(f_eb, rel=0.03)


def test_steering_reduces_hardening():
    beam = _beam()
    beam.set_angles(np.zeros(beam.n_elem))
    g_sym = ssm_from_system(beam.to_system(), ref_dof=beam.midspan_w_dof()).gamma.imag
    beam.set_angles(np.full(beam.n_elem, d2r(45)))   # unsymmetric [45/0]
    g_uns = ssm_from_system(beam.to_system(), ref_dof=beam.midspan_w_dof()).gamma.imag
    assert 0 < g_uns < g_sym                          # coupling softens the hardening
