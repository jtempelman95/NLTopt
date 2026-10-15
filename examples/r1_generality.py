r"""Does the held-frequency lever survive other plates?  (Reviewer R1, M3.)

The paper's plate is simply supported with immovable in-plane edges and aspect ratio
a/b = 4/3.  Here the SAME design problem -- minimise J at exactly the UD frequency, R_min
= 20 mm, 50 mm CS-RBF support grid, six starts -- is solved for

    ss_43     the paper's plate (reference, re-solved here with the same code)
    ss_11     square, 200 x 200 mm
    ss_21     2:1, 200 x 100 mm
    cl_43     clamped edges (w = dw/dn = 0), immovable in-plane, 200 x 150 mm
    mv_43     simply supported, in-plane TRACTION-FREE edges, 200 x 150 mm

and the steered optimum is compared with the strongest UNIFORM laminate at the same
frequency: every single angle (sweep) and a continuous optimisation of the four ply
angles of [t4/t3/t2/t1]_s (SLSQP, frequency equality, 20 starts).  If the UD plate is
the lowest-frequency uniform laminate (Corollary 1), the uniform optimum at omega_UD is
UD itself.

    python examples/r1_generality.py [cfg ...]   -> results/r1_generality_<cfg>.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import levers_study as LS
from nltopt.relaxed_plate import RelaxedPlate, SteeredPlate, stack_AD
from nltopt.laminate import lamina_Q

Q = lamina_Q()
CFGS = {
    "ss_43": dict(a=0.20, b=0.15),
    "ss_11": dict(a=0.20, b=0.20),
    "ss_21": dict(a=0.20, b=0.10),
    "cl_43": dict(a=0.20, b=0.15, bc="clamped"),
    "mv_43": dict(a=0.20, b=0.15, inplane="movable", n_in=10),
}


def use_plate(P):
    """Point levers_study's optimiser at plate P (it reads these module globals)."""
    LS.PLATE = P
    LS.PHI, LS.PHIX, LS.PHIY = LS.csrbf(P)
    LS.NS = LS.PHI.shape[1]


def best_uniform_at(P, w_t, w_ud, rng):
    """Strongest uniform laminate at omega_1 = w_t: single angle and continuous stack."""
    H = P.h
    th = np.linspace(-np.pi / 2, np.pi / 2, 721)
    fU = np.empty(th.size); JU = np.empty(th.size)
    for i, t in enumerate(th):
        A, D = stack_AD([t], H, Q)
        st = P.evaluate(D, A, full=False); fU[i], JU[i] = st["omega1"], st["J"]
    single = []
    for i in range(th.size - 1):
        a_, b_ = fU[i] - w_t, fU[i + 1] - w_t
        if a_ * b_ <= 0 and a_ != b_:
            s = a_ / (a_ - b_); single.append(JU[i] + s * (JU[i + 1] - JU[i]))
    J_single = min(single) if single else np.nan

    def ev(x):
        A, D = stack_AD(list(x), H, Q); st = P.evaluate(D, A, full=False)
        return st["omega1"], st["J"]

    best = np.inf
    seeds = [rng.uniform(-np.pi / 2, np.pi / 2, 4) for _ in range(16)]
    seeds += [np.zeros(4), np.full(4, np.pi / 2), np.full(4, np.pi / 4), np.array([0.3, -0.3, 0.3, -0.3])]
    for x0 in seeds:
        cache = {}

        def f(x):
            k = tuple(np.round(x, 12))
            if k not in cache:
                cache[k] = ev(x)
            return cache[k]
        try:
            r = minimize(lambda x: f(x)[1], x0, method="SLSQP",
                         constraints=[{"type": "eq", "fun": lambda x: f(x)[0] / w_t - 1}],
                         bounds=[(-np.pi / 2, np.pi / 2)] * 4,
                         options=dict(maxiter=200, ftol=1e-12, eps=1e-6))
        except Exception:
            continue
        fx, Jx = ev(r.x)
        if abs(fx / w_t - 1) < 1e-4 and Jx < best:
            best = Jx
    return th, fU, JU, J_single, best


def run(name):
    kw = dict(CFGS[name])
    P = RelaxedPlate(**kw)
    use_plate(P)
    sp = SteeredPlate(P)
    UD = sp.forward(0.0, full=False)
    w_ud, J_ud = UD["omega1"], UD["J"]
    t0 = time.time()
    th, fU, JU, J_single, J_stack = best_uniform_at(P, w_ud, w_ud, np.random.default_rng(1))
    print(f"[{name}] UD f1={w_ud/2/np.pi:.1f} Hz  J={J_ud:.4e}  min uniform f = {fU.min()/w_ud:.4f} f_UD "
          f"at {np.rad2deg(th[np.argmin(fU)]):.1f} deg; best uniform at f_UD: single {J_single/J_ud:.3f} "
          f"stack {J_stack/J_ud:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    r = LS.optimise_steered(w_ud)
    print(f"[{name}] steered @ f_UD: J/J_UD={r['J']/J_ud:.4f}  f/f_UD={r['omega1']/w_ud:.6f}  "
          f"R_used={1000*r['Rmin_used']:.1f} mm  relax share UD {UD['relax']/UD['knl']:.3f} -> "
          f"{r['relax']/r['knl']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(_os.path.join(_ROOT, "results", f"r1_generality_{name}.npz"), name=name,
             a=P.a, b=P.b, bc=kw.get("bc", "ss"), inplane=kw.get("inplane", "immovable"),
             w_ud=w_ud, J_ud=J_ud, relax_ud=UD["relax"] / UD["knl"],
             U_theta=th, U_f=fU, U_J=JU, J_single=J_single, J_stack=J_stack,
             J_steer=r["J"], f_steer=r["omega1"], relax_steer=r["relax"] / r["knl"],
             theta_hat=r["theta_hat"], Rmin_used=r["Rmin_used"])


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(CFGS)):
        run(c)
