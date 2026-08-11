"""Curved von Karman beam model + backbone-shape optimization.

Fast checks (no EPMC continuation):

1. **Straight limit** -- ``w0 = 0`` must reproduce PyHB's straight-beam
   ``VonKarmanBeamForce`` cubic coefficient exactly.
2. **Curvature -> softening** -- a moderate initial curvature flips the
   first-mode backbone from hardening to softening (the mechanism behind
   Pozzi et al. 2025, Fig. 3).
3. **Backbone target optimization** -- shaping the beam to place its backbone at
   a prescribed softening point (the core of the Fig. 3 replication) succeeds.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import VonKarmanBeamForce

from nltopt import ssm_from_system
from nltopt.beam import curved_vk_beam


def test_curved_straight_limit_matches_vonkarman():
    d = curved_vk_beam(0.0, 0.0, h=0.01, length=1.0, n_elem=10)
    ssm = ssm_from_system(d["system"], mode=0, ref_dof=d["ref"])
    beam = d["beam"]
    sys_ref = MechanicalSystem(beam.M, np.zeros_like(beam.M), beam.K,
                               [VonKarmanBeamForce(beam.Kg, d["EA"] / (2 * 1.0))])
    ssm_ref = ssm_from_system(sys_ref, mode=0, ref_dof=d["ref"])
    assert ssm.gamma.imag > 0                                  # straight beam hardens
    assert ssm.gamma.imag == pytest.approx(ssm_ref.gamma.imag, rel=1e-10)


def test_initial_curvature_causes_softening():
    d_straight = curved_vk_beam(0.0, 0.0, h=0.01, length=1.0, n_elem=10)
    d_curved = curved_vk_beam(0.006, 0.0, h=0.01, length=1.0, n_elem=10)
    g_straight = ssm_from_system(d_straight["system"], ref_dof=d_straight["ref"]).gamma.imag
    g_curved = ssm_from_system(d_curved["system"], ref_dof=d_curved["ref"]).gamma.imag
    assert g_straight > 0            # hardening
    assert g_curved < 0             # softening


def test_backbone_target_optimization():
    """Find an initial curvature A1 that softens the backbone to a target point.

    Fixed h, L; single design variable A1; target ``Omega(RMS v = a*) = k omega1``.
    Uses only the SSM backbone (fast).  This is the essential loop of the Fig. 3
    replication (there driven by SLSQP over A1, A2, h, L).
    """
    h, L = 0.006, 1.0
    r = curved_vk_beam(1e-4, 0.0, h, L, n_elem=10)["r_gyr"]
    a_star = 1.0 * r           # target RMS amplitude (~1 gyration radius)
    k_target = 0.99            # 1% softening at a_star

    def softening_gap(A1):
        d = curved_vk_beam(A1, 0.0, h, L, n_elem=10)
        ssm = ssm_from_system(d["system"], mode=0, ref_dof=d["ref"])
        omega, _ = ssm.omega_at_amplitude(a_star, kind="rms")
        return omega / ssm.omega0 - k_target

    # straight beam hardens (gap > 0); a modest curvature softens past the target
    # (softening is non-monotone in A1 -- deep arches re-harden -- so we bracket
    # within the initial monotone softening range)
    assert softening_gap(1e-4) > 0
    assert softening_gap(0.005) < 0
    A1_star = brentq(softening_gap, 1e-4, 0.005, xtol=1e-9)

    d = curved_vk_beam(A1_star, 0.0, h, L, n_elem=10)
    ssm = ssm_from_system(d["system"], mode=0, ref_dof=d["ref"])
    omega, _ = ssm.omega_at_amplitude(a_star, kind="rms")
    assert omega / ssm.omega0 == pytest.approx(k_target, abs=1e-4)
    assert 0.0 < A1_star < 0.005
