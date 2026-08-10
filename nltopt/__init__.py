"""NLTopt -- manufacturable design of *nonlinear* structural dynamics.

NLTopt couples two of our existing tools:

* **PyHB** (``../PyHB``) -- harmonic balance, continuation, shooting and EPMC
  nonlinear modal analysis for second-order mechanical systems, and
* **wavetopo** (``../wavetopo``) -- curvature-feasible fiber-orientation design
  (CS-RBF tow paths, tow-curvature manufacturability constraints, MMA), so far
  applied only to *linear* problems,

to open a new line: **manufacturable fiber-steering design of the nonlinear
vibration response of composite components** -- tailoring backbone and
forced-response curves under tow-curvature manufacturability constraints.

The reduction engine is the **spectral submanifold** (SSM), following
Pozzi/Marconi/Jain/Li/Braghin (2025) and Liang et al. (2025); see
``docs/literature``.  This first release provides the cubic-order 2-D SSM and
its physical backbone curve, validated against analytic results and against
PyHB's EPMC.
"""

from .ssm import SSM2D, compute_ssm_2d, ssm_from_system
from .nonlinearity import multilinear_forms, nonlinear_force

__version__ = "0.0.1"

__all__ = [
    "SSM2D",
    "compute_ssm_2d",
    "ssm_from_system",
    "multilinear_forms",
    "nonlinear_force",
]
