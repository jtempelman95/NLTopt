r"""Auditable verification of the optimizer's ingredients (Reviewer R1, M4).

1  Gradient check at the published held-frequency design: the analytic derivatives of
   J, omega_1, gamma and of the tow-curvature constraint with respect to the CS-RBF
   coefficients, against central finite differences at three step sizes, along five
   random unit directions and the five largest-gradient coordinates.
2  Constraint residuals on an INDEPENDENT grid: the tow curvature of the final field is
   re-evaluated analytically on a 1 mm grid (40 000 points, none of them a quadrature
   point of the optimizer), and the frequency on a finer basis.
3  Multi-start diagnostics of the held-frequency solve: for each of the six starts, the
   final J/J_UD, frequency residual, worst curvature residual, SLSQP status and
   iterations.

    python examples/r1_verification.py       -> results/r1_verification.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, _os.path.join(_ROOT, "examples")); sys.path.insert(0, _WAVETOPO)
import levers_study as LS
from design_source import heldfreq_design
from nltopt.relaxed_plate import SteeredPlate
from wavetopo.cfrp import CSRBFMapping

th_hat = heldfreq_design()
sp = SteeredPlate(LS.PLATE)
UD = sp.forward(0.0, full=False)


def quantities(x):
    st = sp.forward(LS.PHI @ x, full=False)
    z, _ = LS.curl(x)
    return np.array([st["J"] / UD["J"], st["omega1"] / UD["omega1"], st["gamma"] / UD["gamma"]]), z


st = sp.forward(LS.PHI @ th_hat)
gr = sp.grad(st)
G = np.vstack([LS.PHI.T @ gr["J"] / UD["J"], LS.PHI.T @ gr["omega1"] / UD["omega1"],
               LS.PHI.T @ gr["gamma"] / UD["gamma"]])
Gc = np.asarray(LS.curl_jac(th_hat))                       # d kappa_tow / d theta_hat  (Nq, NS)
rng = np.random.default_rng(3)
dirs = [v / np.linalg.norm(v) for v in rng.standard_normal((5, LS.NS))]
top = np.argsort(-np.abs(G[0]))[:5]
for k in top:
    e = np.zeros(LS.NS); e[k] = 1.0; dirs.append(e)
rows = []                                                    # (dir, quantity, step, rel err)
names = ["J", "omega1", "gamma", "tow curvature"]
for i, dv in enumerate(dirs):
    an = np.r_[G @ dv, 0.0]
    zc = Gc @ dv
    for hstep in (1e-4, 1e-5, 1e-6):
        qp, zp = quantities(th_hat + hstep * dv)
        qm, zm = quantities(th_hat - hstep * dv)
        fd = (qp - qm) / (2 * hstep)
        fdz = (zp - zm) / (2 * hstep)
        for j in range(3):
            rows.append((i, j, hstep, abs(an[j] - fd[j]) / max(abs(fd[j]), 1e-14)))
        rows.append((i, 3, hstep, np.linalg.norm(zc - fdz) / max(np.linalg.norm(fdz), 1e-14)))
rows = np.array(rows)
best = {n: min(rows[(rows[:, 1] == j) & (rows[:, 0] == i), 3].min() for i in range(len(dirs)))
        for j, n in enumerate(names)}
worst = {n: max(rows[(rows[:, 1] == j) & (rows[:, 0] == i), 3].min() for i in range(len(dirs)))
         for j, n in enumerate(names)}
for n in names:
    print(f"[1] {n:14s}: best-step relative error over {len(dirs)} directions: "
          f"max {worst[n]:.1e} (median of best {best[n]:.1e})", flush=True)

# ---------------------------------------------------------------- 2 independent grid
xs = np.linspace(0.0005, LS.PLATE.a - 0.0005, 200)
ys = np.linspace(0.0005, LS.PLATE.b - 0.0005, 200)
GX, GY = np.meshgrid(xs, ys)


class _D: pass


m = _D(); m.cx, m.cy, m.N = GX.ravel(), GY.ravel(), GX.size
XS = np.linspace(0, LS.PLATE.a, 5); YS = np.linspace(0, LS.PLATE.b, 4)
SX, SY = np.meshgrid(XS, YS)
R = CSRBFMapping(m, (SX.ravel(), SY.ravel()), r_s=0.075)
th = R.Phi @ th_hat
kap = np.cos(th) * (R.Phi_x @ th_hat) + np.sin(th) * (R.Phi_y @ th_hat)
kmax_fine = np.abs(kap).max()
z_q, _ = LS.curl(th_hat)
print(f"[2] tow curvature: max on optimizer quadrature {np.abs(z_q).max():.3f} 1/m, on independent 1 mm grid "
      f"{kmax_fine:.3f} 1/m; limit {1/LS.R_MIN:.3f} 1/m -> R_min achieved {1000/kmax_fine:.2f} mm", flush=True)

# ---------------------------------------------------------------- 3 multistart
rng = np.random.default_rng(0)
seeds = [np.full(LS.NS, t) for t in np.deg2rad([0, 20, 45, 70])]
seeds += [np.deg2rad(30) + 0.3 * rng.standard_normal(LS.NS) for _ in range(2)]
labels = ["uniform 0", "uniform 20", "uniform 45", "uniform 70", "random 1", "random 2"]
zall = 1 / LS.R_MIN
cache = {}


def state(x):
    k = x.tobytes()
    if k not in cache:
        s = sp.forward(LS.PHI @ x); cache.clear(); cache[k] = (s, sp.grad(s))
    return cache[k]


obj = lambda x: state(x)[0]["J"] / UD["J"]
obj_g = lambda x: LS.PHI.T @ state(x)[1]["J"] / UD["J"]
feq = lambda x: state(x)[0]["omega1"] / UD["omega1"] - 1
feq_g = lambda x: LS.PHI.T @ state(x)[1]["omega1"] / UD["omega1"]
cin = lambda x: 1 - (LS.curl(x)[0] / zall) ** 2
cin_g = lambda x: -(2 * LS.curl(x)[0] / zall ** 2)[:, None] * np.asarray(LS.curl_jac(x))
ms = []
for lab, s0 in zip(labels, seeds):
    r = minimize(obj, s0, jac=obj_g, method="SLSQP", bounds=[(-np.pi, np.pi)] * LS.NS,
                 constraints=[dict(type="eq", fun=feq, jac=feq_g), dict(type="ineq", fun=cin, jac=cin_g)],
                 options=dict(maxiter=200, ftol=1e-10))
    ms.append((obj(r.x), feq(r.x), cin(r.x).min(), r.status, r.nit))
    print(f"[3] start {lab:10s}: J/J_UD={ms[-1][0]:.4f}  freq residual {ms[-1][1]:+.1e}  "
          f"min c_e {ms[-1][2]:+.1e}  status {r.status} ({r.message[:30]})  iterations {r.nit}", flush=True)
np.savez(_os.path.join(_ROOT, "results", "r1_verification.npz"), grad_rows=rows, names=names,
         ndirs=len(dirs), kmax_quad=np.abs(z_q).max(), kmax_fine=kmax_fine, zall=zall,
         ms=np.array(ms, float), ms_labels=labels)
print("wrote results/r1_verification.npz")
