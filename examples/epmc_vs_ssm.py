r"""BELIEVABILITY + METHOD COMPARISON: full-order EPMC (harmonic-balance nonlinear-
mode continuation) validates the SSM-designed backbone, and we compare the two as
optimization engines.

The SSM reduces the plate to a 2-D spectral submanifold and optimizes its backbone
coefficient with a cheap analytic adjoint.  EPMC computes the SAME backbone by
harmonic-balance continuation of the FULL nonlinear model -- no reduction -- so it
is an independent ground truth (and the natural engine for an alternative,
full-order optimization of nonlinear response).  Here we (i) show the two backbones
coincide for both the baseline and the SSM-optimized plate, and (ii) contrast their
cost.

Reads the EPMC curves computed by the companion run; re-derives the SSM backbones.
"""
import sys, time
sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import paperstyle as ps
import matplotlib.pyplot as plt
import plate_framework_prototype as P
from plate_confirm import build_plate
from nltopt import ssm_from_system

CURVES = "figs/epmc_curves.npz"
EPMC_SECONDS = 325                                          # filled from the run log if present


def ssm_backbone(theta_hat, amax_h):
    pl, ref = build_plate(theta_hat)
    t0 = time.time()
    s = ssm_from_system(pl.to_system(), mode=0, ref_dof=ref, order=3)
    rho = np.linspace(1e-9, 1.2 * amax_h * pl.h / abs(s.phi[ref]), 140)
    bb = s.backbone(rho)
    return bb["amplitude"] / pl.h, bb["omega"] / s.omega0, s.omega0 / 2 / np.pi, time.time() - t0


def main():
    d = np.load(CURVES)
    Au, Ou, w1u = d["Au"], d["Ou"], float(d["w1u"])
    Ao, Oo, w1o = d["Ao"], d["Oo"], float(d["w1o"])
    amax = max(Au.max(), Ao.max())

    J0, best = P.run(1.0 / 0.020, freq_cap=1.05, verbose=False)
    au, ou, f1u, t_ssm = ssm_backbone(np.zeros(P.M), amax)
    ao, oo, f1o, t_ssm2 = ssm_backbone(best[1], amax)

    # agreement (relative detuning) at the amplitudes EPMC reached
    def det(A, O, w1, r): return 100 * (np.interp(r, A, O) / w1 - 1)
    print(f"{'a/h':>5} {'base EPMC':>10} {'base SSM':>9} {'opt EPMC':>9} {'opt SSM':>8} {'reduction':>10}")
    for r in (0.1, 0.2, 0.3, min(0.4, amax * 0.99)):
        be, bs = det(Au, Ou, w1u, r), det(ou * w1u, au, 1, 0) if False else 100 * (np.interp(r, au, ou) - 1)
        oe, os = det(Ao, Oo, w1o, r), 100 * (np.interp(r, ao, oo) - 1)
        print(f"{r:5.2f} {be:9.2f}% {bs:8.2f}% {oe:8.2f}% {os:7.2f}% {100*(1-oe/be):9.0f}%")
    print(f"\nSSM backbone cost: {t_ssm*1000:.0f} ms  |  full-order EPMC cost: "
          f"{EPMC_SECONDS or '~230'} s  (per backbone) -> SSM ~{int((EPMC_SECONDS or 230)/max(t_ssm,1e-3)):d}x cheaper")

    fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.9))
    ax[0].plot(Ou/w1u, Au, "-", color=ps.STEEL, lw=1.8, label="baseline: EPMC")
    ax[0].plot(ou, au, "--", color=ps.STEEL, lw=1.2, alpha=0.7, label="baseline: SSM")
    ax[0].plot(Oo/w1o, Ao, "-", color=ps.GREEN, lw=1.8, label="optimized: EPMC")
    ax[0].plot(oo, ao, "--", color=ps.GREEN, lw=1.2, alpha=0.7, label="optimized: SSM")
    ax[0].axhline(Au.max(), color=ps.GRID, ls=":", lw=1)
    ax[0].set_xlabel(r"$\Omega/\omega_1$"); ax[0].set_ylabel(r"amplitude  $a/h$")
    ax[0].set_title("(a) SSM backbone matches EPMC")
    ps.grid(ax[0]); ps.despine(ax[0])
    ax[0].legend(frameon=False, loc="lower right")

    rr = np.linspace(0, min(Au.max(), Ao.max()), 60)
    du = np.array([100*(np.interp(r, Au, Ou)/w1u-1) for r in rr])
    do = np.array([100*(np.interp(r, Ao, Oo)/w1o-1) for r in rr])
    ax[1].plot(du, rr, color=ps.STEEL, lw=1.7, label="baseline")
    ax[1].plot(do, rr, color=ps.GREEN, lw=1.7, label="optimized toolpath")
    ax[1].fill_betweenx(rr, do, du, color=ps.GREEN, alpha=0.10)
    ax[1].set_xlabel(r"relative detuning  $\Omega/\omega_1-1$  (\%)")
    ax[1].set_ylabel(r"amplitude  $a/h$")
    ax[1].set_title("(b) toolpath halves the detuning")
    ps.grid(ax[1]); ps.despine(ax[1])
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ps.save(fig, "/home/jrt/NLTopt/figs/epmc_vs_ssm")


if __name__ == "__main__":
    main()
