r"""DIRECT NUMERICAL CONFIRMATION of the jump result: brute-force time-domain
forced frequency sweeps (up and down, RK45, 2% Rayleigh damping, in-plane
statically condensed) of the FULL nonlinear plate, at a single drive level.

The unidirectional panel's resonance is multivalued: the up-sweep rides the upper
branch and JUMPS DOWN, the down-sweep rides the lower branch and JUMPS UP -- the two
sweeps disagree (hysteresis).  The steered panel is single-valued: up- and
down-sweeps coincide, no jump.  This is the time-domain ground truth behind
Fig.~plate_jump (which uses the SSM forced-response curve).

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_jump_numerics.py
"""
import sys
sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import paperstyle as ps
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import plate_framework_prototype as P
from plate_confirm import build_plate, ssm_frc, _sweep_task

ZETA = 0.02
OMR = (0.985, 1.075)


def main():
    J0, best = P.run(1.0 / 0.020, freq_cap=1.05, verbose=False)
    pl0, ref0 = build_plate(np.zeros(P.M))
    F = 0.0100 * pl0.h * pl0.K[ref0, ref0]                   # the drive from Fig. plate_jump
    designs = [("unidirectional", np.zeros(P.M)), ("steered tow", best[1])]
    print(f"time-domain forced sweeps at F={F:.2e}, zeta={ZETA} (up and down)...")

    cache = "figs/plate_jump_numerics.npz"
    import os
    if "--replot" in sys.argv and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); res = d["res"].item()
    else:
        with ProcessPoolExecutor(max_workers=4) as ex:
            fut = {}
            for name, th in designs:
                fut[(name, "up")] = ex.submit(_sweep_task, (th, F, OMR, ZETA, True))
                fut[(name, "down")] = ex.submit(_sweep_task, (th, F, OMR, ZETA, False))
            res = {k: f.result() for k, f in fut.items()}
        np.savez(cache, res=res)

    def hysteresis(ou, au, od, ad):
        def clean(o, a):
            m = ~np.isnan(a); o, a = o[m], a[m]; s = np.argsort(o); return o[s], a[s]
        ou, au = clean(ou, au); od, ad = clean(od, ad)
        g = np.linspace(max(ou.min(), od.min()), min(ou.max(), od.max()), 150)
        d = np.abs(np.interp(g, ou, au) - np.interp(g, od, ad))
        return g[np.argmax(d)], d.max()

    fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.85), sharey=True)
    for k, (name, th) in enumerate(designs):
        pl, ref = build_plate(th)
        frc = ssm_frc(pl, ref, F, ZETA)
        ou, au = res[(name, "up")]; od, ad = res[(name, "down")]
        Ojump, gap = hysteresis(ou, au / pl.h, od, ad / pl.h)
        jumped = gap > 0.05
        ax[k].plot(frc[:, 0], frc[:, 1], "-", color=ps.GREY, lw=1.4, zorder=1,
                   label="SSM forced response")
        ax[k].plot(ou, au / pl.h, "o", color=ps.STEEL, ms=3.5, label="time sim: up", zorder=3)
        ax[k].plot(od, ad / pl.h, "s", color=ps.RED, ms=3.2, mfc="none",
                   label="time sim: down", zorder=3)
        if jumped:
            a_up = np.interp(Ojump, *(lambda o, a: (o[np.argsort(o)], a[np.argsort(o)]))(ou, au/pl.h))
            a_dn = np.interp(Ojump, *(lambda o, a: (o[np.argsort(o)], a[np.argsort(o)]))(od, ad/pl.h))
            ax[k].annotate("", xy=(Ojump, min(a_up, a_dn) + 0.006),
                           xytext=(Ojump, max(a_up, a_dn) - 0.006),
                           arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=1.6))
            ax[k].text(Ojump + 0.002, 0.5*(a_up+a_dn), rf"jump $\Delta={gap:.2f}\,h$",
                       color=ps.RED, fontsize=7.5, fontweight="bold", va="center")
        ax[k].set_title(rf"({'ab'[k]}) {name}: " + ("jumps (hysteresis)" if jumped else "smooth (no jump)"))
        ax[k].set_xlabel(r"drive frequency  $\Omega/\omega_1$")
        ax[k].set_xlim(OMR)
        ps.grid(ax[k]); ps.despine(ax[k])
        ax[k].legend(frameon=False, loc="upper right")
        print(f"  {name:16}: max up-down gap = {gap:.3f} a/h -> {'JUMP' if jumped else 'no jump'}")
    ax[0].set_ylabel(r"steady amplitude  $a/h$")
    fig.tight_layout()
    ps.save(fig, "/home/jrt/NLTopt/figs/plate_jump_numerics")


if __name__ == "__main__":
    main()
