"""Multi-mode von Karman composite plate (flagship physics)."""

import numpy as np
import pytest

from nltopt.plate import VonKarmanPlate
from nltopt.laminate import lamina_Q, RHO
from nltopt import ssm_from_system
from pyhb.nonlinear_modes import nonlinear_mode

A, B, H = 0.20, 0.15, 0.002


def test_linear_frequency_matches_analytic():
    """SS plate fundamental (1,1) matches the closed-form composite-plate value."""
    pl = VonKarmanPlate(A, B, H, n_modes=6, nq=12)
    D = (H ** 3 / 12) * lamina_Q()
    f_ana = np.sqrt((np.pi ** 4 / (RHO * H)) *
                    (D[0, 0] / A**4 + 2 * (D[0, 1] + 2 * D[2, 2]) / (A**2 * B**2)
                     + D[1, 1] / B**4)) / (2 * np.pi)
    assert pl.natural_frequencies()[0] / (2 * np.pi) == pytest.approx(f_ana, rel=1e-6)


def test_nonlinear_tangent_matches_fd():
    pl = VonKarmanPlate(A, B, H, n_modes=3, n_in=3, nq=10)
    el = pl.to_system().nonlinear_elements[0]
    nm = pl.ndof
    q = 1e-3 * np.random.default_rng(0).standard_normal(nm)
    _, Kt, _ = el.forces(q[None, :], np.zeros((1, nm)))
    Kt = Kt[0]
    for i in range(nm):
        qp = q.copy(); qp[i] += 1e-7
        qm = q.copy(); qm[i] -= 1e-7
        col = (el.forces(qp[None, :], 0*qp[None, :])[0][0]
               - el.forces(qm[None, :], 0*qm[None, :])[0][0]) / 2e-7
        assert np.allclose(col, Kt[:, i], rtol=1e-4, atol=1e-4 * np.max(np.abs(Kt)))


def test_plate_hardens_and_backbone_matches_epmc():
    pl = VonKarmanPlate(A, B, H, n_modes=3, nq=12)
    ref = pl.midspan_mode()
    sys = pl.to_system()
    ssm = ssm_from_system(sys, mode=0, ref_dof=ref, order=3)
    assert ssm.gamma.imag > 0                                   # von Karman plate hardens

    # SSM backbone coefficient vs full-order EPMC (small amplitude)
    amax = 0.03 * H
    a_end = np.log10(1.2 * amax / abs(ssm.phi[ref]))
    nmres = nonlinear_mode(sys, H=5, mode_index=0, ref_dof=ref,
                           log_a_start=a_end - 1.6, log_a_end=a_end,
                           ds=0.1, ds_max=0.15, max_steps=300)
    Ae, Oe = nmres["phys_amplitude"], nmres["omega"]
    o = np.argsort(Ae); Ae, Oe = Ae[o], Oe[o]
    sel = Ae <= amax
    kap_epmc = np.polyfit(Ae[sel] ** 2, Oe[sel], 1)[0]
    assert ssm.backbone_coeff() == pytest.approx(kap_epmc, rel=0.05)


def test_membrane_relaxation_softens_and_converges():
    """In-plane DOF slave to w^2 and reduce the cubic hardening; it converges."""
    def bc(n_in):
        pl = VonKarmanPlate(A, B, H, n_modes=3, n_in=n_in, nq=12)
        return ssm_from_system(pl.to_system(), mode=0,
                               ref_dof=pl.midspan_mode(), order=3).backbone_coeff()
    restrained = bc(0)
    relaxed = bc(4)
    assert 0 < relaxed < restrained                 # relaxation softens hardening
    assert relaxed == pytest.approx(bc(5), rel=2e-2)  # converged in n_in


def test_fiber_angle_tunes_normalized_hardening():
    """The 2-D anisotropy makes theta tune the *normalized* hardening, not just
    the frequency (unlike a 1-D beam where EI, EA scale together)."""
    def bc(thd):
        pl = VonKarmanPlate(A, B, H, n_modes=3, nq=12)
        pl.set_theta(np.deg2rad(thd) * np.ones(pl.Nq))
        return ssm_from_system(pl.to_system(), mode=0,
                               ref_dof=pl.midspan_mode(), order=3).backbone_coeff()
    b0, b30, b90 = bc(0), bc(30), bc(90)
    assert b30 < b0                       # 30 deg lowers the amplitude detuning
    assert b90 > b0                       # 90 deg raises it -> a real tuning range
