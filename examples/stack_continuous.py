r"""Strongest uniform competitor at held frequency: continuous symmetric stacks.

The 15-degree enumeration of [t1/t2/t3/t4]_s in levers_study.py is a grid; between grid
points the continuous single angle can beat it (e.g. 0.75 vs 0.81 at 1.1 f_UD).  Here we
minimise J over CONTINUOUS ply angles subject to omega_1 = omega_target (SLSQP,
finite-difference gradients, 4 variables), multi-started from the best grid stacks near the
target, the continuous single-angle solution, and random stacks.

    python examples/stack_continuous.py   -> results/stack_cont.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, (_WAVETOPO + ""))
from nltopt.relaxed_plate import RelaxedPlate, stack_AD
from nltopt.laminate import lamina_Q

Q = lamina_Q()
PLATE = RelaxedPlate()
H = PLATE.h
L = np.load("results/levers.npz", allow_pickle=True)
w_ud, J_ud = float(L["w_ud"]), float(L["J_ud"])
S = L["S"]; fS, JS = S[:, 4] / w_ud, S[:, 5] / J_ud
fU, JU, thU = L["U_f"] / w_ud, L["U_J"] / J_ud, L["U_theta"]


def ev(a):
    A, D = stack_AD(list(a), H, Q)
    st = PLATE.evaluate(D, A, full=False)
    return st["omega1"] / w_ud, st["J"] / J_ud


def single_angle_at(t):
    out = []
    for i in range(len(fU) - 1):
        a, b = fU[i] - t, fU[i + 1] - t
        if a * b <= 0 and a != b:
            s = a / (a - b)
            out.append((JU[i] + s * (JU[i + 1] - JU[i]), thU[i] + s * (thU[i + 1] - thU[i])))
    return min(out) if out else (np.nan, np.nan)


def solve(t, x0):
    cache = {}

    def f(x):
        k = tuple(np.round(x, 12))
        if k not in cache:
            cache[k] = ev(x)
        return cache[k]

    r = minimize(lambda x: f(x)[1], x0, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda x: f(x)[0] - t}],
                 bounds=[(-np.pi / 2, np.pi / 2)] * 4, options=dict(maxiter=200, ftol=1e-10, eps=1e-6))
    fx, Jx = ev(r.x)
    return (Jx, r.x) if abs(fx - t) < 1e-4 else (np.inf, r.x)


rng = np.random.default_rng(0)
res = {}
for t in (1.0, 1.1, 1.2, 1.3):
    t0 = time.time()
    near = np.where(np.abs(fS - t) < 0.03)[0]
    seeds = [S[k, :4] for k in near[np.argsort(JS[near])][:12]]
    Js, ths = single_angle_at(t)
    if np.isfinite(Js):
        seeds.append(np.full(4, ths)); seeds.append(np.array([ths, -ths, ths, -ths]))
    seeds += [rng.uniform(-np.pi / 2, np.pi / 2, 4) for _ in range(8)]
    best = (np.inf, None)
    for x0 in seeds:
        try:
            b = solve(t, np.asarray(x0, float))
        except Exception:
            continue
        if b[0] < best[0]:
            best = b
    grid = JS[np.abs(fS - t) < 0.01].min() if (np.abs(fS - t) < 0.01).any() else np.nan
    res[t] = (best[0], best[1], Js, grid)
    print(f"t={t:.1f}: continuous stack J/J_UD={best[0]:.4f} angles(deg)={np.rad2deg(best[1]).round(1)}  "
          f"single angle {Js:.4f}  grid stack {grid:.4f}  ({time.time()-t0:.0f}s, {len(seeds)} starts)")

np.savez("results/stack_cont.npz", targets=np.array(list(res)),
         J_cont=np.array([v[0] for v in res.values()]), angles=np.array([v[1] for v in res.values()]),
         J_single=np.array([v[2] for v in res.values()]), J_grid=np.array([v[3] for v in res.values()]))
print("wrote results/stack_cont.npz")
