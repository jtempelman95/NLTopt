r"""How much of the steered-plate gain is limited by the PARAMETRISATION and how much
by PHYSICS?

(a) CS-RBF resolution: support spacing dx_s in {50, 33, 25} mm at R_min = 20 mm and
    with no curvature limit.
(b) Physics ceiling for single-ply steering: theta free at EVERY quadrature point
    (3072 unknowns, no smoothness, no curvature limit), frequency held by an
    augmented Lagrangian, L-BFGS-B with the analytic gradient.  No manufacturable
    design can beat this; it bounds what in-plane steering of this laminate can do.

    python examples/ceiling_study.py     # -> results/ceiling.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, (_ROOT + "/examples"))
import levers_study as LS
from nltopt.relaxed_plate import SteeredPlate

P = LS.PLATE


def resolution(dx_s, R_min):
    LS.PHI, LS.PHIX, LS.PHIY = LS.csrbf(P, dx_s=dx_s, r_s=1.5 * dx_s)
    LS.NS = LS.PHI.shape[1]
    w_ud = SteeredPlate(P).forward(0.0, full=False)["omega1"]
    r = LS.optimise_steered(w_ud, R_min=R_min, maxiter=300)
    return r, LS.NS


def ceiling():
    sp = SteeredPlate(P)
    UD = sp.forward(0.0, full=False)
    w_ud, J_ud = UD["omega1"], UD["J"]
    lam, mu = 0.0, 50.0
    th = np.full(P.Nq, np.deg2rad(20.0))

    def f(x):
        st = sp.forward(x); g = sp.grad(st)
        c = st["omega1"] / w_ud - 1.0
        val = st["J"] / J_ud + lam * c + 0.5 * mu * c * c
        grad = g["J"] / J_ud + (lam + mu * c) * g["omega1"] / w_ud
        return val, grad

    for k in range(12):
        r = minimize(f, th, jac=True, method="L-BFGS-B", options=dict(maxiter=400))
        th = r.x
        st = sp.forward(th, full=False)
        c = st["omega1"] / w_ud - 1.0
        lam += mu * c; mu *= 2.0
        print(f"  AL{k}: J/J_UD={st['J']/J_ud:.4f}  f/f_UD-1={c:+.2e}")
        if abs(c) < 1e-5 and k > 2:
            break
    return th, st["J"] / J_ud


def main():
    out = {}
    for dx in (0.05, 0.0333, 0.025):
        for R in (0.020, np.inf):
            t0 = time.time()
            r, ns = resolution(dx, R)
            Jr = r["J"] / SteeredPlate(P).forward(0.0, full=False)["J"]
            print(f"[res] dx_s={1000*dx:.0f}mm (M={ns})  R_min={'inf' if not np.isfinite(R) else f'{1000*R:.0f}mm'}"
                  f"  J/J_UD={Jr:.3f}  R_used={1000*r['Rmin_used']:.0f}mm  [{time.time()-t0:.0f}s]")
            out[f"res_{int(1000*dx)}_{'inf' if not np.isfinite(R) else int(1000*R)}"] = Jr
            out[f"th_{int(1000*dx)}_{'inf' if not np.isfinite(R) else int(1000*R)}"] = r["theta_hat"]
    th, Jc = ceiling()
    print(f"[ceiling] pointwise single-ply steering at held frequency: J/J_UD = {Jc:.3f}")
    out.update(ceiling_theta=th, ceiling_J=Jc)
    np.savez("results/ceiling.npz", **out)
    print("wrote results/ceiling.npz")


if __name__ == "__main__":
    main()
