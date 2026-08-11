"""Reference systems from the literature, for replication and testing.

Currently: the two coupled nonlinear oscillators of Pozzi, Marconi, Jain, Li &
Braghin (2025), Eq. (33) / Fig. 2 -- the backbone-sensitivity validation case.
"""

from __future__ import annotations

import numpy as np

from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import GeneralNonlinearElement


class _CoupledOscForce(GeneralNonlinearElement):
    r"""Nonlinear force of the coupled-oscillator benchmark (Pozzi 2025, Eq. 33).

        f1 = k2 x1^2 + k3 x1^3 + k2 (x1-x2)^2 + k3 (x1-x2)^3
        f2 = k2 (x2-x1)^2 + k3 (x2-x1)^3
    """

    n = 2

    def __init__(self, k2, k3):
        self.k2 = float(k2)
        self.k3 = float(k3)

    def forces(self, q_time, u_time):
        k2, k3 = self.k2, self.k3
        x1, x2 = q_time[:, 0], q_time[:, 1]
        d = x1 - x2                                  # (x2-x1)^2=d^2, (x2-x1)^3=-d^3
        N = q_time.shape[0]
        f = np.zeros((N, 2))
        f[:, 0] = k2 * x1 ** 2 + k3 * x1 ** 3 + k2 * d ** 2 + k3 * d ** 3
        f[:, 1] = k2 * d ** 2 - k3 * d ** 3
        Kt = np.zeros((N, 2, 2))
        Kt[:, 0, 0] = 2 * k2 * x1 + 3 * k3 * x1 ** 2 + 2 * k2 * d + 3 * k3 * d ** 2
        Kt[:, 0, 1] = -(2 * k2 * d + 3 * k3 * d ** 2)
        Kt[:, 1, 0] = 2 * k2 * d - 3 * k3 * d ** 2
        Kt[:, 1, 1] = -(2 * k2 * d - 3 * k3 * d ** 2)
        return f, Kt, np.zeros((N, 2, 2))


def coupled_oscillators(m=1.0, k=1.0, k2=0.5, k3=0.2, alphaR=0.0, betaR=0.1):
    r"""Two coupled nonlinear oscillators (Pozzi et al. 2025, Eq. 33).

    First mass grounded; masses joined by nonlinear (quadratic + cubic) springs.
    Rayleigh damping ``C = alphaR M + betaR K``.  Defaults reproduce the paper:
    first-mode eigenvalue ``lambda = -0.0191 +/- i 0.6177`` and a softening
    backbone.  Returns a :class:`pyhb.MechanicalSystem` (track ``x2`` = dof 1).
    """
    M = m * np.eye(2)
    K = k * np.array([[2.0, -1.0], [-1.0, 1.0]])
    C = alphaR * M + betaR * K
    return MechanicalSystem(M, C, K, [_CoupledOscForce(k2, k3)])
