"""Verification of the cubic-order 2-D SSM.

Three independent checks, in increasing order of what they exercise:

1. **Duffing (1 DOF)** -- the reduced cubic coefficient and physical-amplitude
   read-out against the *analytic* backbone
   ``Omega = omega0 (1 + 3 knl A^2 / (8 omega0^2))``.  No slaving; a pure test
   of ``gamma`` and the amplitude map.

2. **Quadratically-coupled 2-DOF** -- the *manifold* (order-2 slaving) part: a
   quadratic coupling to a second mode feeds the effective cubic backbone
   coefficient through ``w_(2,0)`` / ``w_(1,1)``.  Checked against PyHB EPMC.

3. **Clamped-clamped von Karman beam** -- the literature replication (Pozzi
   et al. 2025, Sec. 5.2 setting): a distributed cubic nonlinearity, hardening
   backbone, multi-DOF.  Checked against PyHB EPMC.
"""

import numpy as np
import pytest

from pyhb import MechanicalSystem, CubicSpring, BeamModel
from pyhb.nonlinear_elements import VonKarmanBeamForce, GeneralNonlinearElement
from pyhb.nonlinear_modes import nonlinear_mode

from nltopt import ssm_from_system, compute_ssm_2d


# --------------------------------------------------------------------------- #
# A distributed quadratic element realizing exactly f(x) = [b x1 x2 ; (b/2) x1^2]
# (energy-consistent: derivable from potential V = (b/2) x1^2 x2), used to give
# EPMC and the SSM the *same* nonlinearity for the slaving cross-check.
# --------------------------------------------------------------------------- #
class ExactQuadCoupling2DOF(GeneralNonlinearElement):
    def __init__(self, b):
        self.b = float(b)
        self.n = 2

    def forces(self, q_time, u_time):
        b = self.b
        x1, x2 = q_time[:, 0], q_time[:, 1]
        N = q_time.shape[0]
        f = np.zeros((N, 2))
        f[:, 0] = b * x1 * x2
        f[:, 1] = 0.5 * b * x1 ** 2
        Kt = np.zeros((N, 2, 2))
        Kt[:, 0, 0] = b * x2
        Kt[:, 0, 1] = b * x1
        Kt[:, 1, 0] = b * x1
        Ct = np.zeros((N, 2, 2))
        return f, Kt, Ct


# --------------------------------------------------------------------------- #
def _epmc_backbone_slope(system, mode_index, log_a_start, log_a_end, H=7,
                         ref_dof=0, amp_max=None):
    """Small-amplitude slope d(Omega)/d(A^2) from an EPMC backbone."""
    nm = nonlinear_mode(system, H=H, mode_index=mode_index, ref_dof=ref_dof,
                        log_a_start=log_a_start, log_a_end=log_a_end,
                        ds=0.05, ds_max=0.08)
    A = nm["phys_amplitude"]
    Om = nm["omega"]
    order = np.argsort(A)
    A, Om = A[order], Om[order]
    if amp_max is None:
        amp_max = 0.3 * A.max()
    sel = A <= amp_max
    p = np.polyfit(A[sel] ** 2, Om[sel], 1)
    return p[0], Om[sel][0]  # slope, omega0


def _ssm_backbone_slope(ssm, amp_max):
    """Small-amplitude slope d(Omega)/d(A^2) from the *sampled* SSM backbone.

    The reduced-amplitude grid is scaled to the physical amplitude (``A ~
    phi_ref rho``) so enough points fall under ``amp_max`` for any system.
    """
    pr = max(abs(ssm.phi[ssm.ref_dof]), 1e-12)
    rho = np.linspace(1e-6, 1.3 * amp_max / pr, 60)
    bb = ssm.backbone(rho)
    A, Om = bb["amplitude"], bb["omega"]
    sel = A <= amp_max
    p = np.polyfit(A[sel] ** 2, Om[sel], 1)
    return p[0]


# --------------------------------------------------------------------------- #
def test_duffing_analytic_backbone():
    """SSM cubic coefficient reproduces the exact first-order Duffing backbone."""
    for omega0_sq, knl in [(1.0, 1.0), (4.0, 0.5), (2.0, 3.0)]:
        sys = MechanicalSystem([[1.0]], [[0.0]], [[omega0_sq]],
                               [CubicSpring([1.0], knl)])
        ssm = ssm_from_system(sys)
        omega0 = np.sqrt(omega0_sq)

        # analytic: Omega = omega0 + (3 knl / (8 omega0)) A^2
        kappa_exact = 3.0 * knl / (8.0 * omega0)
        assert ssm.gamma.real == pytest.approx(0.0, abs=1e-9)  # conservative
        assert ssm.backbone_coeff() == pytest.approx(kappa_exact, rel=1e-6)

        # and via the sampled physical-amplitude backbone
        kappa_num = _ssm_backbone_slope(ssm, amp_max=0.15)
        assert kappa_num == pytest.approx(kappa_exact, rel=2e-2)


def test_duffing_fifth_order_coefficient():
    """Order-5 SSM reproduces the Duffing A^4 backbone coefficient (-21/256).

    ``Omega = omega0 (1 + 3/8 eps A^2 - 21/256 eps^2 A^4 + ...)`` for
    ``x'' + omega0^2 x + eps x^3 = 0``; here omega0=eps=1.
    """
    sys = MechanicalSystem([[1.0]], [[0.0]], [[1.0]], [CubicSpring([1.0], 1.0)])
    ssm5 = ssm_from_system(sys, order=5)
    rho = np.linspace(1e-4, 0.3, 200)
    bb = ssm5.backbone(rho)
    c = np.polyfit(bb["amplitude"] ** 2, bb["omega"] - 1.0, 2)  # c0*(A^2)^2 + c1*A^2 + c2
    assert c[1] == pytest.approx(0.375, rel=2e-3)       # A^2: exact 3/8
    assert c[0] == pytest.approx(-21.0 / 256.0, rel=0.05)  # A^4: exact -21/256

    # order 5 is strictly more accurate than order 3 at moderate amplitude
    from scipy.special import ellipk
    A = 0.4
    om_exact = (np.pi / 2) * np.sqrt(1 + A ** 2) / ellipk(A ** 2 / (2 * (1 + A ** 2)))

    def om_at(order):
        s = ssm_from_system(sys, order=order)
        rho = np.linspace(1e-4, 0.7, 400)
        bb = s.backbone(rho)
        return np.interp(A, bb["amplitude"], bb["omega"])

    assert abs(om_at(5) - om_exact) < abs(om_at(3) - om_exact) / 5.0


def test_duffing_matches_epmc():
    """SSM and EPMC agree on the small-amplitude Duffing backbone slope."""
    sys = MechanicalSystem([[1.0]], [[0.0]], [[1.0]], [CubicSpring([1.0], 1.0)])
    ssm = ssm_from_system(sys)
    slope_ssm = _ssm_backbone_slope(ssm, amp_max=0.12)
    slope_epmc, _ = _epmc_backbone_slope(sys, 0, -2.5, -0.8, amp_max=0.12)
    assert slope_ssm == pytest.approx(slope_epmc, rel=5e-2)


def test_quadratic_coupling_slaving_vs_epmc():
    """Order-2 slaving: a quadratic coupling corrects the cubic backbone.

    Two modes with a quadratic coupling; the master (mode 0) backbone picks up
    a cubic correction routed through the slaved mode 1.  A single-mode Galerkin
    projection (which ignores slaving) would get this wrong; the SSM must match
    the full-order EPMC.
    """
    # modal-space system: master omega1, slave omega2, quadratic coupling.
    #   x1'' + w1^2 x1 + b x1 x2   = 0
    #   x2'' + w2^2 x2 + (b/2) x1^2 = 0     (energy-consistent quadratic)
    w1, w2, b = 1.0, 3.5, 0.8
    M = np.eye(2)
    C = np.zeros((2, 2))
    K = np.diag([w1 ** 2, w2 ** 2])
    element = ExactQuadCoupling2DOF(b)
    sys = MechanicalSystem(M, C, K, [element])

    # SSM reads the same element (via polarization) -> order-2 slaving included
    ssm = ssm_from_system(sys, mode=0, ref_dof=0)
    slope_ssm = _ssm_backbone_slope(ssm, amp_max=0.10)

    # EPMC on the identical system (independent full-order backbone)
    slope_epmc, _ = _epmc_backbone_slope(sys, 0, -3.0, -1.3, H=9, amp_max=0.10)

    assert slope_ssm < 0.0                       # quadratic coupling -> softening
    assert slope_ssm == pytest.approx(slope_epmc, rel=8e-2)


def test_vonkarman_beam_vs_epmc():
    """Literature replication: clamped-clamped von Karman beam hardening backbone."""
    beam = BeamModel(n_elem=8, length=1.0, EI=1.0, rhoA=1.0, EA=1.0).clamped_clamped()
    coeff = beam.EA / (2.0 * beam.length)
    vk = VonKarmanBeamForce(beam.Kg, coeff)
    sys = MechanicalSystem(beam.M, np.zeros_like(beam.M), beam.K, [vk])

    ref = beam.free_index_of(beam.node_dof(beam.n_elem // 2, "w"))  # mid-span w
    ssm = ssm_from_system(sys, mode=0, ref_dof=ref)
    assert ssm.hardening                                    # beam stretching -> hardening

    # EPMC needs a moderate amplitude; compare small-amplitude slopes.
    A_scale = 0.02 * beam.length
    slope_ssm = _ssm_backbone_slope(ssm, amp_max=A_scale)
    slope_epmc, w0 = _epmc_backbone_slope(sys, 0, -3.5, -1.5, H=7,
                                          ref_dof=ref, amp_max=A_scale)
    assert slope_ssm > 0.0 and slope_epmc > 0.0
    assert slope_ssm == pytest.approx(slope_epmc, rel=8e-2)
