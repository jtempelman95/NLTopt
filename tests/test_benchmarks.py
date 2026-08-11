"""Coupled-oscillator benchmark + backbone-sensitivity validation (Pozzi 2025, Fig. 2)."""

import numpy as np
import pytest

from nltopt import ssm_from_system
from nltopt.benchmarks import coupled_oscillators

DOF = 1  # track x2


def _ssm(**params):
    return ssm_from_system(coupled_oscillators(**params), mode=0, ref_dof=DOF)


def test_coupled_oscillator_matches_paper_eigenvalue():
    ssm = _ssm()
    # paper: lambda = -0.0191 +/- i 0.6177, softening backbone
    assert ssm.lambda1.real == pytest.approx(-0.0191, abs=5e-4)
    assert ssm.lambda1.imag == pytest.approx(0.6177, abs=5e-4)
    assert ssm.gamma.imag < 0.0                      # softening


def test_backbone_sensitivity_predicts_perturbation():
    """dOmega/dmu . dmu predicts the perturbed backbone (crosses land on lines)."""
    levels = np.linspace(0.1, 0.5, 6)
    nominal = dict(m=1.0, k=1.0, k2=0.5, k3=0.2)

    def omega_at(params):
        ssm = ssm_from_system(coupled_oscillators(**params), ref_dof=DOF)
        return np.array([ssm.omega_at_amplitude(a, dof=DOF, kind="rms")[0]
                         for a in levels])

    om_nom = omega_at(nominal)
    for param, frac in [("m", 0.01), ("k", 0.01), ("k2", 0.03), ("k3", 0.03)]:
        v = nominal[param]
        eps = 1e-4 * v
        sens = (omega_at(dict(nominal, **{param: v + eps}))
                - omega_at(dict(nominal, **{param: v - eps}))) / (2 * eps)
        dmu = frac * v
        om_pred = om_nom + sens * dmu
        om_act = omega_at(dict(nominal, **{param: v + dmu}))
        # first-order prediction matches the finite perturbation to high accuracy
        assert np.max(np.abs(om_pred - om_act)) < 1e-3
