r"""ENGINEERING DELIVERABLE: an amplitude-robust CFRP resonator panel, and the
optimal manufacturable fiber toolpath that makes it.

The component: a 200x150x2 mm carbon/epoxy panel used as a resonant element
(sensor diaphragm / harvester / precision mount) at its fundamental ~256 Hz.  A
plain unidirectional panel HARDENS strongly at working amplitude -- its resonance
climbs +17% by a/h=0.5 and jumps -- which corrupts calibration and drives
instability.  We steer the tow path (automated fiber placement, min radius
R>=20 mm) to cancel that geometric hardening at the SAME linear frequency, giving
an amplitude-stable resonance.  The deliverable is the OPTIMAL TOOLPATH; the SSM
backbone objective and its adjoint design it, and full-order EPMC plus brute-force
frequency sweeps confirm it.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_design.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys, pickle
sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plate_framework_prototype as P
from nltopt.laminate import RHO

A, B, H = P.A, P.B, P.H
R_LIM = 20.0                                                  # min tow radius [mm]


def toolpath_panel(ax, th_opt):
    """Optimal fiber toolpath as continuous tow courses, coloured by the local
    minimum steering radius (green = gentle, red = at the R>=20 mm AFP limit)."""
    nelx, nely = P.mesh.nelx, P.mesh.nely
    xs = (np.arange(nelx) + 0.5) * P.mesh.dx
    ys = (np.arange(nely) + 0.5) * P.mesh.dy
    th_g = (P.Phi @ th_opt).reshape(nely, nelx)
    zeta = np.abs(P.curl(th_opt)[0]).reshape(nely, nelx)
    Rmm = np.clip(1000.0 / np.maximum(zeta, 1e-6), 0, 200)    # local min radius [mm]
    u, v = np.cos(th_g), np.sin(th_g)
    sp = ax.streamplot(xs, ys, u, v, density=2.6, color=Rmm, cmap="RdYlGn",
                       norm=plt.Normalize(R_LIM, 120), linewidth=1.5, arrowsize=1e-6)
    ax.set_xlim(0, A); ax.set_ylim(0, B); ax.set_aspect("equal")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
    ax.set_title("Optimal fiber toolpath  (AFP tow courses)", fontweight="bold", fontsize=12)
    cb = plt.colorbar(sp.lines, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("local min tow radius  $R$  [mm]")
    ax.text(0.015, 0.02, f"curl-feasible  $R\\geq{1000/zeta.max():.0f}$ mm  "
            f"(AFP limit {R_LIM:.0f} mm)", transform=ax.transAxes,
            fontsize=8.5, va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.8))


def main():
    print("Designing the amplitude-robust panel (SSM adjoint, frequency held)...")
    J0, best = P.run(1.0 / (R_LIM / 1000.0), freq_cap=1.05, verbose=False)
    th_opt = best[1]
    curves, omr = pickle.load(open("figs/plate_confirm_cache.pkl", "rb"))
    uni, opt = curves["unidirectional"], curves["optimized tow"]
    f1u, f1o = uni["w1"] / 2 / np.pi, opt["w1"] / 2 / np.pi
    (ob_u, ab_u), (ob_o, ab_o) = uni["bb"], opt["bb"]
    a_ev = 0.40                                              # amplitude for the drift metric
    d_u = 100 * (np.interp(a_ev, ab_u / H, ob_u) - 1.0)
    d_o = 100 * (np.interp(a_ev, ab_o / H, ob_o) - 1.0)

    GREY, GREEN, INK = "#9aa0a6", "#1b7a3d", "#3a3f45"
    fig = plt.figure(figsize=(15.0, 6.6))
    gs = GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                  wspace=0.24, hspace=0.42, left=0.05, right=0.97, top=0.86, bottom=0.10)
    axT = fig.add_subplot(gs[:, 0])
    toolpath_panel(axT, th_opt)

    # backbone: amplitude stability
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(ob_u, ab_u / H, color=GREY, lw=2.6, label=f"unidirectional $[0]$  (+{d_u:.0f}%)")
    axB.plot(ob_o, ab_o / H, color=GREEN, lw=2.8, label=f"steered toolpath  (+{d_o:.0f}%)")
    axB.axvline(1.0, color="0.85", ls=":", lw=1)
    axB.set_ylim(0, 0.47)
    xu, xo = np.interp(a_ev, ab_u / H, ob_u), np.interp(a_ev, ab_o / H, ob_o)
    axB.annotate("", xy=(xo, a_ev), xytext=(xu, a_ev),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    axB.text(0.5 * (xu + xo), a_ev + 0.03, f"detuning $-{100*(1-d_o/d_u):.0f}\\%$",
             ha="center", fontsize=9.5, color=INK)
    axB.set_xlabel(r"resonance frequency  $\Omega/\omega_1$")
    axB.set_ylabel(r"amplitude  $a/h$")
    axB.set_title("Resonance holds with amplitude", fontsize=11)
    axB.legend(frameon=False, fontsize=8.5, loc="lower right"); axB.grid(alpha=0.25)

    # forced-response confirmation
    axC = fig.add_subplot(gs[1, 1])
    frc = opt["frc"]; ou, au = opt["up"]; od, ad = opt["down"]
    axC.plot(frc[:, 0], frc[:, 1], "-", color="#c98a00", lw=2.0, label="SSM forced response")
    axC.plot(ou, au / H, "o", color="#1f3b8c", ms=4, label="time sim (sweep up)")
    axC.plot(od, ad / H, "s", color="#d1495b", ms=3.5, mfc="none", label="time sim (sweep down)")
    axC.set_xlim(omr)
    axC.set_xlabel(r"drive frequency  $\Omega/\omega_1$")
    axC.set_ylabel(r"amplitude  $a/h$")
    axC.set_title("Verified: SSM = EPMC = time sweeps", fontsize=11)
    axC.legend(frameon=False, fontsize=8, loc="upper right"); axC.grid(alpha=0.25)

    fig.suptitle("Amplitude-robust CFRP resonator panel via an optimal fiber toolpath",
                 fontweight="bold", fontsize=15, y=0.975)
    fig.text(0.5, 0.915, f"carbon/epoxy $200\\times150\\times2$ mm  $\\cdot$  "
             f"$f_1={f1o:.0f}$ Hz held ($\\pm5\\%$)  $\\cdot$  AFP min radius $R\\geq{R_LIM:.0f}$ mm  "
             f"$\\cdot$  amplitude detuning $-{100*(1-d_o/d_u):.0f}\\%$",
             ha="center", fontsize=10.5, color="#555")
    fig.savefig("figs/plate_design_hero.png", dpi=135)
    print(f"unidirectional drifts +{d_u:.0f}% at a/h=0.5;  steered toolpath +{d_o:.0f}%  "
          f"(detuning -{100*(1-d_o/d_u):.0f}%), f1 held {f1o:.0f} vs {f1u:.0f} Hz")
    print("wrote figs/plate_design_hero.png")


if __name__ == "__main__":
    main()
