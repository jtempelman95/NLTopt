r"""Single source of truth for the plate designs used by every confirmation figure.

``heldfreq_design()`` returns the CS-RBF coefficients of the steered single-ply
design that minimises the relaxed detuning J at EXACTLY the unidirectional plate's
(adapted-mode) frequency, with R_min = 20 mm (examples/levers_study.py,
``optimise_steered``).  It is computed once and cached in
results/design_heldfreq.npz.  The support grid (0.05 m spacing, r_s = 0.075 m) is
the same as plate_framework_prototype's, so the coefficients are interchangeable.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "results", "design_heldfreq.npz")


def heldfreq_design(R_min=0.020):
    if os.path.exists(_F):
        d = np.load(_F)
        if abs(float(d["R_min"]) - R_min) < 1e-12:
            return d["theta_hat"]
    import levers_study as LS
    from nltopt.relaxed_plate import SteeredPlate
    w_ud = SteeredPlate(LS.PLATE).forward(0.0, full=False)["omega1"]
    r = LS.optimise_steered(w_ud, R_min=R_min)
    np.savez(_F, theta_hat=r["theta_hat"], R_min=R_min, J=r["J"], omega1=r["omega1"])
    return r["theta_hat"]
