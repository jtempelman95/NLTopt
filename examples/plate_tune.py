r"""DESIGN TO A TARGET: dial the fiber toolpath so the panel's nonlinear backbone
hits a PRESCRIBED amplitude-detuning, not merely the minimum.

Useful when a set of composite resonators must carry distinct, specified
nonlinearities -- a graded/"rainbow" array of nonlinear absorbers, or matched
elements binned to a spec.  Same verified SSM adjoint as before; the objective
becomes a least-squares match to a target relative detuning tau = J/J_uni:

    min_theta ( gamma_eff/K_b / (gamma_eff/K_b)_unidirectional  -  tau )^2
    s.t.  curl (manufacturability) limit,  linear-frequency ceiling.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_tune.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plate_framework_prototype as P
from wavetopo.cfrp_optimizer import MMA

A, B, H = P.A, P.B, P.H


def run_target(tau, R_mm=20.0, freq_cap=1.05, iters=140):
    """Steer the tow field to hit relative detuning ``tau`` = J/J_unidirectional."""
    zeta_all = 1.0 / (R_mm / 1000.0)
    J_uni, Kb_ref, _, _ = P.objective(np.zeros(P.M))
    Kb_cap = freq_cap ** 2 * Kb_ref
    th = np.deg2rad(20.0) * np.ones(P.M)
    lam = np.full(P.N, 15.0); lam_f = 0.0; mu = 5.0
    mma = MMA(np.full(P.M, -np.pi/2), np.full(P.M, np.pi/2))
    best = None
    for k in range(iters):
        for _ in range(5):
            J, Kb, gam, theta = P.objective(th)
            r = J / J_uni - tau                             # target residual
            gth = 2.0 * r * (P.objective_grad(th, Kb, gam, theta) / J_uni)
            zeta, theta = P.curl(th); g = zeta**2 - zeta_all**2
            fac = (lam + mu*np.maximum(g, -lam/mu)) * (g > -lam/mu)
            gth += (1.0/P.N) * (P.curl_jac(th, theta).T @ (2*zeta*fac))
            gf = Kb/Kb_cap - 1.0
            facf = (lam_f + mu*max(gf, -lam_f/mu)) * (gf > -lam_f/mu)
            gth += facf * (P.kb_grad(th, theta) / Kb_cap)
            th = np.clip(mma.update(th, gth), -np.pi/2, np.pi/2)
        J, Kb, gam, theta = P.objective(th)
        zeta, theta = P.curl(th); zmax = float(np.abs(zeta).max()); gf = Kb/Kb_cap - 1.0
        feas = (zmax/zeta_all < 1.01) and (gf < 5e-3)
        err = abs(J/J_uni - tau)
        if feas and (best is None or err < best[0]):
            best = (err, th.copy(), J/J_uni, zmax)
        lam = np.maximum(lam + mu*np.maximum(g, -lam/mu), 0.0)
        lam_f = max(lam_f + mu*max(gf, -lam_f/mu), 0.0)
        if not feas:
            mu = min(1.25*mu, 1e7)
    return J_uni, best


def backbone(theta_hat, amax=0.45):
    _, Kb, gam, _ = P.objective(theta_hat)
    bc, w1 = P.backbone_coeff_ssm(Kb, gam)
    aa = np.linspace(0, amax*H, 120)
    return 1.0 + bc*aa**2/w1, aa/H


def toolpath(ax, theta_hat, title):
    nelx, nely = P.mesh.nelx, P.mesh.nely
    xs = (np.arange(nelx)+0.5)*P.mesh.dx; ys = (np.arange(nely)+0.5)*P.mesh.dy
    tg = (P.Phi @ theta_hat).reshape(nely, nelx)
    Rmm = np.clip(1000.0/np.maximum(np.abs(P.curl(theta_hat)[0]), 1e-6), 0, 200).reshape(nely, nelx)
    ax.streamplot(xs, ys, np.cos(tg), np.sin(tg), density=1.7, color=Rmm,
                  cmap="RdYlGn", norm=plt.Normalize(20, 120), linewidth=1.0, arrowsize=1e-6)
    ax.set_xlim(0, A); ax.set_ylim(0, B); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


def main():
    targets = [0.80, 0.65, 0.50]                            # prescribed relative detuning
    shades = ["#7cb342", "#2e7d32", "#12592a"]
    print(f"{'target tau':>10} {'achieved':>9} {'reduction':>10} {'R>=(mm)':>8}")
    designs = []
    for tau in targets:
        J_uni, best = run_target(tau)
        err, th, ach, zmax = best
        designs.append((tau, th, ach, zmax))
        print(f"{tau:10.2f} {ach:9.3f} {100*(1-ach):9.0f}% {1000/zmax:8.0f}")

    fig, ax = plt.subplots(2, 3, figsize=(ps.FULL, 4.2),
                           gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.38, wspace=0.2))
    gsB = ax[0, 0].get_gridspec()
    for a in ax[0, :]:
        a.remove()
    axb = fig.add_subplot(gsB[0, :])

    ob, ab = backbone(np.zeros(P.M))
    axb.plot(ob, ab, color=ps.GREY, lw=1.7, label=r"unidirectional $[0]$ (baseline)")
    for (tau, th, ach, zmax), col in zip(designs, shades):
        o, a = backbone(th)
        axb.plot(o, a, color=col, lw=1.6,
                 label=rf"target $\tau={tau:.2f}\to$ achieved {ach:.2f}")
    axb.axvline(1.0, color=ps.GRID, ls=":", lw=1)
    axb.set_xlabel(r"resonance frequency  $\Omega/\omega_1$")
    axb.set_ylabel(r"amplitude  $a/h$")
    axb.set_title("(a) dial-a-backbone: the toolpath sets the detuning to spec")
    ps.grid(axb); ps.despine(axb)
    axb.legend(frameon=False, loc="lower right")

    for (tau, th, ach, zmax), a in zip(designs, ax[1, :]):
        toolpath(a, th, rf"$\tau={tau:.2f}$  ($R\geq{1000/zmax:.0f}$ mm)")
    fig.tight_layout()
    ps.save(fig, (_ROOT + "/figs/plate_tune"))


if __name__ == "__main__":
    main()
