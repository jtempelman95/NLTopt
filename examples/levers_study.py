r"""Which laminate freedoms decouple the hardening from the linear frequency?

The achievable set of (omega_1, J) at FIXED mass for four design classes, all
evaluated by the same adapted-mode, in-plane-relaxed model
(:class:`nltopt.relaxed_plate.RelaxedPlate`, 25 transverse + 32 in-plane modes):

  U  uniform single-ply angle theta           (1 parameter)
  S  uniform symmetric stack [t1/t2/t3/t4]_s  (through-thickness freedom, 15 deg grid)
  F  steered single-ply field theta(x)        (in-plane freedom, CS-RBF, curl-limited)
  SF steered stack: the best stack at f*, rotated pointwise by theta(x)

For F and SF the optimiser is SLSQP on the CS-RBF coefficients with an EQUALITY
on the real (adapted-mode) frequency, omega_1 = omega*, and the pointwise tow-
curvature inequality |(n.grad)theta| <= 1/R_min at every quadrature point --
constraints satisfied exactly, not penalised.  Gradients are analytic
(``RelaxedPlate.gradient``), FD-verified in tests/test_relaxed_plate.py.

    python examples/levers_study.py            # writes results/levers.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

sys.path.insert(0, (_WAVETOPO + ""))
from wavetopo.cfrp import CSRBFMapping

from nltopt.relaxed_plate import RelaxedPlate, SteeredPlate, stack_AD, rotate
from nltopt.laminate import lamina_Q

Q = lamina_Q()
PLATE = RelaxedPlate()
H = PLATE.h
R_MIN = 0.020                                   # 20 mm minimum tow radius (paper value)


# ---------------------------------------------------------------- CS-RBF map
class _Duck:
    pass


def csrbf(plate, dx_s=0.05, r_s=0.075):
    xs = np.linspace(0, plate.a, int(round(plate.a / dx_s)) + 1)
    ys = np.linspace(0, plate.b, int(round(plate.b / dx_s)) + 1)
    SX, SY = np.meshgrid(xs, ys)
    m = _Duck(); m.cx, m.cy, m.N = plate.qx, plate.qy, plate.Nq
    r = CSRBFMapping(m, (SX.ravel(), SY.ravel()), r_s=r_s)
    return r.Phi.toarray(), r.Phi_x.toarray(), r.Phi_y.toarray()


PHI, PHIX, PHIY = csrbf(PLATE)
NS = PHI.shape[1]


def curl(th_hat):
    th = PHI @ th_hat
    return np.cos(th) * (PHIX @ th_hat) + np.sin(th) * (PHIY @ th_hat), th


def curl_jac(th_hat):
    th = PHI @ th_hat
    a, b = PHIX @ th_hat, PHIY @ th_hat
    return (diags(-np.sin(th) * a + np.cos(th) * b) @ PHI
            + diags(np.cos(th)) @ PHIX + diags(np.sin(th)) @ PHIY)


# ---------------------------------------------------------------- classes U, S
def uniform_angle_sweep(n=181):
    sp = SteeredPlate(PLATE)
    th = np.linspace(0, np.pi / 2, n)
    f = np.empty(n); J = np.empty(n); gam = np.empty(n)
    for i, t in enumerate(th):
        st = sp.forward(t, full=False)
        f[i], J[i], gam[i] = st["omega1"], st["J"], st["gamma"]
    return th, f, J, gam


def stack_enumeration(step_deg=15):
    angs = np.deg2rad(np.arange(-75, 91, step_deg))
    out = []
    t0 = time.time()
    for a1 in angs:
        for a2 in angs:
            for a3 in angs:
                for a4 in angs:
                    A, D = stack_AD([a1, a2, a3, a4], H, Q)   # mid-plane -> outer
                    st = PLATE.evaluate(D, A, full=False)
                    out.append((a1, a2, a3, a4, st["omega1"], st["J"], st["gamma"]))
    print(f"  stacks: {len(out)} in {time.time()-t0:.0f}s")
    return np.array(out)


# ---------------------------------------------------------------- classes F, SF
def optimise_steered(omega_target, D0=None, A0=None, R_min=R_MIN, seeds=None, maxiter=200):
    """min J(theta) s.t. omega_1(theta) = omega_target, |zeta| <= 1/R_min."""
    sp = SteeredPlate(PLATE, D0=D0, A0=A0)
    zall = 1.0 / R_min
    cache = {}

    def state(x):
        k = x.tobytes()
        if k not in cache:
            st = sp.forward(PHI @ x)
            cache.clear(); cache[k] = (st, sp.grad(st))
        return cache[k]

    J_ref = state(np.zeros(NS))[0]["J"]
    obj = lambda x: state(x)[0]["J"] / J_ref
    obj_g = lambda x: PHI.T @ state(x)[1]["J"] / J_ref
    feq = lambda x: state(x)[0]["omega1"] / omega_target - 1.0
    feq_g = lambda x: PHI.T @ state(x)[1]["omega1"] / omega_target

    def cin(x):
        z, _ = curl(x)
        return 1.0 - (z / zall) ** 2

    def cin_g(x):
        z, _ = curl(x)
        return -(2 * z / zall ** 2)[:, None] * np.asarray(curl_jac(x))

    cons = [dict(type="eq", fun=feq, jac=feq_g)]
    if np.isfinite(R_min):                      # R_min = inf -> no tow-curvature limit
        cons.append(dict(type="ineq", fun=cin, jac=cin_g))
    if seeds is None:
        rng = np.random.default_rng(0)
        seeds = [np.full(NS, t) for t in np.deg2rad([0, 20, 45, 70])]
        seeds += [np.deg2rad(30) + 0.3 * rng.standard_normal(NS) for _ in range(2)]
    best = None
    for s0 in seeds:
        r = minimize(obj, s0, jac=obj_g, constraints=cons, method="SLSQP",
                     bounds=[(-np.pi, np.pi)] * NS,
                     options=dict(maxiter=maxiter, ftol=1e-10))
        ok = abs(feq(r.x)) < 1e-4 and (not np.isfinite(R_min) or cin(r.x).min() > -1e-4)
        if ok and (best is None or r.fun < best[0]):
            best = (r.fun, r.x.copy())
    if best is None:
        return None
    x = best[1]
    st = sp.forward(PHI @ x)
    z, _ = curl(x)
    return dict(theta_hat=x, J=st["J"], omega1=st["omega1"], gamma=st["gamma"],
                knl=st["knl"], relax=st["relax"], mid=st["mid"],
                Rmin_used=1.0 / max(np.abs(z).max(), 1e-12))


def main():
    sp = SteeredPlate(PLATE)
    UD = sp.forward(0.0, full=False)
    w_ud, J_ud = UD["omega1"], UD["J"]
    print(f"UD: f1={w_ud/2/np.pi:.1f} Hz  J={J_ud:.4e}  detuning@a/h=0.4 "
          f"{100*PLATE.detuning(UD, 0.4):.2f}%")

    th, fU, JU, gU = uniform_angle_sweep()
    print(f"uniform: f1 in [{fU.min()/2/np.pi:.0f},{fU.max()/2/np.pi:.0f}] Hz, "
          f"J/J_UD in [{(JU/J_ud).min():.3f},{(JU/J_ud).max():.3f}]")
    S = stack_enumeration()

    targets = w_ud * np.array([0.90, 1.00, 1.10, 1.20, 1.30])
    F = []
    for wt in targets:
        t0 = time.time()
        r = optimise_steered(wt)
        print(f"  steered @ {wt/w_ud:.2f} f_UD: "
              + ("infeasible" if r is None else
                 f"J/J_UD={r['J']/J_ud:.3f}  R_used={1000*r['Rmin_used']:.0f}mm") +
              f"  ({time.time()-t0:.0f}s)")
        F.append(r)

    # steered stack: rotate the best uniform stack at each target
    SF = []
    for wt in targets:
        sel = np.abs(S[:, 4] / wt - 1) < 0.01
        if not sel.any():
            SF.append(None); continue
        k = np.argmin(np.where(sel, S[:, 5], np.inf))
        A0, D0 = stack_AD(S[k, :4], H, Q)
        r = optimise_steered(wt, D0=D0, A0=A0,
                             seeds=[np.zeros(NS), np.full(NS, 0.2), np.full(NS, -0.2)])
        if r is not None:
            r["stack"] = S[k, :4]; r["J_stack"] = S[k, 5]
        print(f"  steered stack @ {wt/w_ud:.2f}: stack {np.rad2deg(S[k,:4]).round()} "
              f"J/J_UD {S[k,5]/J_ud:.3f} -> "
              + ("infeasible" if r is None else f"{r['J']/J_ud:.3f}"))
        SF.append(r)

    np.savez("results/levers.npz", w_ud=w_ud, J_ud=J_ud,
             U_theta=th, U_f=fU, U_J=JU, U_gamma=gU, S=S, targets=targets,
             F=np.array(F, dtype=object), SF=np.array(SF, dtype=object),
             PHI=PHI)
    print("wrote results/levers.npz")


if __name__ == "__main__":
    main()
