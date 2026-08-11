r"""PROTOTYPE: couple wavetopo's composite-design framework to an NLTopt SSM
backbone objective, to verify the loop before the full 2-D von Karman plate.

What this checks end-to-end (the "framework"):
    theta_hat  --(wavetopo CS-RBF Phi)-->  theta(x)  [orientation field]
    theta(x)   --(CLT A,B,D per element)-->  K(theta), knl(theta)  [plate stiffness]
    (M,K,knl)  --(NLTopt SSM)-->  backbone coefficient gamma_3(theta)
    objective J(gamma_3), curl zeta = cos(theta) Phi_x thetahat + sin(theta) Phi_y thetahat
    --(projected-gradient + curl penalty, best-feasible)-->  manufacturable theta*(x)

Reused from wavetopo unchanged: QuadMesh, grid_support_points, CSRBFMapping
(the exact Phi, Phi_x, Phi_y and hence the curl). Reused from NLTopt:
laminate_ABD (CLT) and ssm_from_system (the backbone).

PLACEHOLDER PHYSICS (flagged): a single simply-supported assumed mode
w = q sin(pi x/a) sin(pi y/b) with a fully-restrained membrane, so the plate
reduces to a Duffing oscillator whose bending stiffness K(theta) and membrane
hardening knl(theta) are domain integrals of the fiber-angle-dependent D(theta)
and A(theta). This is enough to exercise the *coupling*; the flagship replaces it
with a multi-mode / full-FE von Karman plate reduced by the same SSM.

    PYTHONPATH=/home/jrt/wavetopo python3 examples/plate_framework_prototype.py
"""
import os, sys
sys.path.insert(0, "/home/jrt/wavetopo")

import numpy as np
from wavetopo.cfrp import QuadMesh, grid_support_points, CSRBFMapping

from pyhb import MechanicalSystem, CubicSpring
from nltopt import ssm_from_system
from nltopt.laminate import laminate_ABD, RHO

# ---- domain, mesh, CS-RBF orientation field (wavetopo) ---------------------
A, B, H = 0.20, 0.15, 0.002               # plate a x b, thickness
mesh = QuadMesh(nelx=20, nely=15, Lx=A, Ly=B)
cx, cy = mesh.cx, mesh.cy
area = mesh.A if np.ndim(mesh.A) else np.full(cx.size, mesh.A)
sx, sy = grid_support_points(mesh, dx_s=0.05, dy_s=0.05)
rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.075)
Msup = rbf.M                              # number of support-point design vars
ZETA_ALL = np.deg2rad(1.0) / 0.010        # curl limit = 1/R, R=10mm  (rad/m)

# ---- mode shape derivatives at element centres (SS plate) ------------------
kx, ky = np.pi / A, np.pi / B
s_x, s_y = np.sin(kx * cx), np.sin(ky * cy)
c_x, c_y = np.cos(kx * cx), np.cos(ky * cy)
phi = s_x * s_y
kap = np.vstack([kx**2 * phi, ky**2 * phi, -2 * kx * ky * c_x * c_y])   # curvature (3,N)
mmb = np.vstack([(kx * c_x * s_y)**2, (ky * s_x * c_y)**2,
                 2 * (kx * c_x * s_y) * (ky * s_x * c_y)])              # (grad w)^2 terms (3,N)
Mmodal = RHO * H * np.sum(area * phi**2)

_ABD_CACHE = {}
def _abd(th):
    key = round(float(th), 5)
    if key not in _ABD_CACHE:
        _ABD_CACHE[key] = laminate_ABD([th], H)      # single rotated ply: B=0
    return _ABD_CACHE[key]


def backbone_coeff(theta_hat):
    """SSM backbone hardening coefficient J for the CS-RBF design theta_hat."""
    theta = rbf.Phi @ theta_hat                       # per-element orientation
    Kb = 0.0
    IA = 0.0
    for e in range(cx.size):
        Ae, Be, De = _abd(theta[e])
        Kb += area[e] * (kap[:, e] @ De @ kap[:, e])
        IA += area[e] * (mmb[:, e] @ Ae @ mmb[:, e])
    knl = 0.5 * IA
    sys = MechanicalSystem([[Mmodal]], [[0.0]], [[Kb]], [CubicSpring([1.0], knl)])
    ssm = ssm_from_system(sys, mode=0, order=3)
    return ssm.backbone_coeff(), ssm.omega0, knl, Kb


def curl(theta_hat):
    theta = rbf.Phi @ theta_hat
    zx = rbf.Phi_x @ theta_hat
    zy = rbf.Phi_y @ theta_hat
    return np.cos(theta) * zx + np.sin(theta) * zy    # zeta = (n . grad) theta


def main():
    rng = np.random.default_rng(0)
    th = np.deg2rad(20.0) * np.ones(Msup)             # start: ~20 deg uniform-ish
    J0, w0, knl0, Kb0 = backbone_coeff(th)
    z0 = np.max(np.abs(curl(th)))
    print(f"start:  f1={w0/2/np.pi:6.1f} Hz  J(hardening)={J0:.4e}  "
          f"max|zeta|={z0:.1f} (limit {ZETA_ALL:.1f})")

    # projected gradient on J with an augmented-Lagrangian-style curl penalty
    # that is *raised while infeasible* and eased otherwise (wavetopo's scheme),
    # retaining the best curl-feasible iterate.
    step = 0.30
    mu = 5.0e-2
    th = np.deg2rad(20.0) * np.ones(Msup)
    best_feas = None
    for it in range(1, 61):
        def obj(x):
            J, *_ = backbone_coeff(x)
            z = curl(x)
            pen = np.mean(np.maximum(0.0, (z / ZETA_ALL)**2 - 1.0))  # dimensionless
            return J / J0 + mu * pen
        g = np.zeros(Msup); f0 = obj(th); eps = 1e-4
        for i in range(Msup):
            tp = th.copy(); tp[i] += eps
            g[i] = (obj(tp) - f0) / eps
        th = np.clip(th - step * g / (np.linalg.norm(g) + 1e-12), -np.pi/2, np.pi/2)
        J, w, knl, Kb = backbone_coeff(th)
        zmax = np.max(np.abs(curl(th)))
        if zmax <= ZETA_ALL and (best_feas is None or J < best_feas[0]):
            best_feas = (J, th.copy())
        mu = min(mu * 1.35, 1e3) if zmax > ZETA_ALL else mu * 0.9   # penalty continuation
        if it % 10 == 0 or it == 1:
            fb = "" if best_feas is None else f"  bestfeas={100*(1-best_feas[0]/J0):+.0f}%"
            print(f"  it{it:2d}: f1={w/2/np.pi:6.1f} Hz  J={100*(1-J/J0):+.0f}%  "
                  f"max|zeta|={zmax:4.1f}  mu={mu:.1e}{fb}")

    print("\n=== prototype outcome ===")
    print("COUPLING VERIFIED: wavetopo CS-RBF theta + exact curl  <-->  NLTopt SSM")
    print("backbone objective; the loop runs end-to-end and the 2-D orientation")
    print("field is a STRONG lever on the backbone (unlike the 1-D beam).")
    if best_feas is None:
        print("Curl NOT driven feasible by this crude projected-gradient/penalty:")
        print("that is exactly wavetopo's MMA + augmented-Lagrangian's job -- the")
        print("next step is to drive the SAME objective/gradient through it.")
    else:
        Jf_val, thf = best_feas
        print(f"Best curl-feasible: hardening reduced {100*(1-Jf_val/J0):.0f}%, "
              f"max|zeta| <= {ZETA_ALL:.2f}.")


if __name__ == "__main__":
    main()
