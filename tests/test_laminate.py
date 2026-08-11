"""Classical laminate theory: the bending-extension coupling B is the design lever."""

import numpy as np
import pytest

from nltopt.laminate import laminate_ABD, beam_abd, Qbar, lamina_Q, E1, E2

d2r = np.deg2rad


def test_symmetric_layup_has_zero_coupling():
    for lay in ([0, 45, 45, 0], [30, -30, -30, 30], [0, 90, 90, 0]):
        A, B, D = laminate_ABD(d2r(lay), 0.002)
        assert np.max(np.abs(B)) < 1e-6 * np.max(np.abs(A))   # B == 0


def test_single_ply_has_zero_coupling():
    A, B, D = laminate_ABD([d2r(30)], 0.002)
    assert np.max(np.abs(B)) == 0.0


def test_unsymmetric_layup_has_coupling():
    A, B, D = laminate_ABD(d2r([0, 90]), 0.002)
    assert abs(B[0, 0]) > 0.0


def test_on_axis_modulus():
    nu21 = 0.27 * E2 / E1
    assert Qbar(0.0)[0, 0] == pytest.approx(E1 / (1 - 0.27 * nu21), rel=1e-6)
    assert Qbar(d2r(90))[0, 0] == pytest.approx(E2 / (1 - 0.27 * nu21), rel=1e-6)


def test_steering_lever_monotone():
    """[theta/0]: steering the top ply raises the dimensionless coupling 0 -> ~0.75."""
    coup = []
    for thd in [0, 30, 60, 90]:
        A11, B11, D11, _ = beam_abd(d2r([thd, 0.0]), 0.002)
        coup.append(B11 / np.sqrt(A11 * D11))
    assert coup[0] == pytest.approx(0.0, abs=1e-9)   # theta=0 -> symmetric, no coupling
    assert np.all(np.diff(coup) > 0)                 # monotone lever
    assert coup[-1] > 0.7                            # strong coupling available
