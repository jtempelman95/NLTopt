r"""Numerical confirmation of the manufacturable-design result.

The prototype optimizes a single-mode surrogate (J = knl/K_b).  Here we take the
resulting curl-feasible tow layout, transfer it onto the *real* multi-mode von
Karman plate WITH in-plane relaxation, and confirm the backbone two ways for both
the unidirectional baseline and the optimized design:

  * NLTopt SSM  (the reduction),  vs
  * full-order EPMC nonlinear-mode continuation  (ground truth).

If the SSM and EPMC agree and the optimized design's amplitude detuning is lower
than the baseline's under EPMC, the result is confirmed -- and we get the honest
detuning reduction on the real physics (which the single-mode surrogate only
approximates).

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_confirm.py
"""
import sys
sys.path.insert(0, "/home/jrt/wavetopo")

import numpy as np
from wavetopo.cfrp import CSRBFMapping

import plate_framework_prototype as P          # reuse its mesh/rbf/optimizer
from nltopt.plate import VonKarmanPlate
from nltopt import ssm_from_system
from pyhb.nonlinear_modes import nonlinear_mode

A, B, H = P.A, P.B, P.H


def _epmc_detuning(sys, ref, w1, amax):
    """Full-order relative detuning coefficient d(Omega/omega1)/d(a^2) via EPMC."""
    ssm = ssm_from_system(sys, mode=0, ref_dof=ref, order=3)
    a_end = np.log10(1.15 * amax / abs(ssm.phi[ref]))
    nm = nonlinear_mode(sys, H=5, mode_index=0, ref_dof=ref,
                        log_a_start=a_end - 1.6, log_a_end=a_end,
                        ds=0.1, ds_max=0.15, max_steps=300)
    Ae, Oe = nm["phys_amplitude"], nm["omega"]
    o = np.argsort(Ae); Ae, Oe = Ae[o], Oe[o]
    sel = Ae <= amax
    return np.polyfit(Ae[sel] ** 2, Oe[sel] / w1, 1)[0]     # d(Om/w1)/da^2


def theta_on_plate(plate, theta_hat):
    """Evaluate the CS-RBF orientation field (support values theta_hat) at the
    plate's quadrature points."""
    class _DuckMesh:                                        # CSRBFMapping needs N, cx, cy
        pass
    dm = _DuckMesh(); dm.cx = plate.qx; dm.cy = plate.qy; dm.N = plate.qx.size
    rbf = CSRBFMapping(dm, (P.rbf.sx, P.rbf.sy), r_s=P.rbf.r_s)
    return rbf.Phi @ theta_hat


def main():
    # 1) get a curl-feasible optimized tow layout from the prototype (R = 20 mm)
    R_mm = 20.0
    J0, best = P.run(1.0 / (R_mm / 1000.0), verbose=False)
    th_opt = best[1]
    print(f"prototype (single-mode): baseline J0={J0:.3e} -> optimized {best[0]:.3e} "
          f"({100*(1-best[0]/J0):.0f}% reduction), curl-feasible R>={R_mm:.0f}mm")

    # 2) build the REAL relaxed plate for baseline and optimized layouts
    amax = 0.03 * H
    print("\n           relative amplitude detuning  d(Om/w1)/da^2  [1/m^2]")
    print(f"{'design':>14} {'f1(Hz)':>8} {'SSM':>12} {'EPMC':>12} {'SSM vs EPMC':>12}")
    results = {}
    for name, th in [("unidirectional", np.zeros(P.M)), ("optimized tow", th_opt)]:
        pl = VonKarmanPlate(A, B, H, n_modes=3, n_in=3, nq=12)
        pl.set_theta(theta_on_plate(pl, th))
        ref = pl.midspan_mode(); sys = pl.to_system()
        ssm = ssm_from_system(sys, mode=0, ref_dof=ref, order=3)
        w1 = ssm.omega0
        det_ssm = ssm.backbone_coeff() / w1
        det_epmc = _epmc_detuning(sys, ref, w1, amax)
        results[name] = (det_ssm, det_epmc, w1)
        print(f"{name:>14} {w1/2/np.pi:8.1f} {det_ssm:12.4e} {det_epmc:12.4e} "
              f"{abs(det_ssm-det_epmc)/abs(det_epmc):11.1%}")

    du_s, du_e, _ = results["unidirectional"]
    do_s, do_e, _ = results["optimized tow"]
    print(f"\nCONFIRMED on the real relaxed plate (full-order EPMC):")
    print(f"  detuning reduction  SSM = {100*(1-do_s/du_s):.0f}%   EPMC = {100*(1-do_e/du_e):.0f}%")
    print(f"  (single-mode surrogate predicted {100*(1-best[0]/J0):.0f}%)")


if __name__ == "__main__":
    main()
