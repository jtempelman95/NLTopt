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

# The SSM / nonlinearity layer builds on the author's harmonic-balance package PyHB (not
# public; available from the author on request). It is imported lazily, so the relaxed
# evaluator (nltopt.relaxed_plate, nltopt.laminate) and the paper's number generation run
# without PyHB.
_LAZY = {"SSM2D": ".ssm", "compute_ssm_2d": ".ssm", "ssm_from_system": ".ssm",
         "multilinear_forms": ".nonlinearity", "nonlinear_force": ".nonlinearity"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(name)


__version__ = "0.0.1"

__all__ = [
    "SSM2D",
    "compute_ssm_2d",
    "ssm_from_system",
    "multilinear_forms",
    "nonlinear_force",
]
