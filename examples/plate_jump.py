r"""THE ENGINEERING PAYOFF: manufacturable fiber steering extends the jump-free
operating range of a resonant composite panel.

At large amplitude the panel's resonance hardens; above a critical drive the
forced-response curve folds and the response JUMPS -- a sudden amplitude change
with hysteresis that corrupts a resonant sensor/harvester and can be unsafe. The
jump onset is a saddle-node of the forced response, at amplitude
``a_crit = sqrt(2 zeta w1 / (sqrt(3) kappa))`` (Duffing), so halving the hardening
``kappa`` pushes ``a_crit`` up by sqrt(2).  We show the steered panel stays
single-valued (no jump) at a drive where the baseline panel already jumps.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_jump.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import matplotlib.pyplot as plt

import plate_framework_prototype as P
from plate_confirm import build_plate, ssm_frc
from nltopt import ssm_from_system

ZETA = 0.02


def _multivalued(pl, ref, F):
    frc = ssm_frc(pl, ref, F, ZETA)
    m = (frc[:, 0] > 0.97) & (frc[:, 0] < 1.14) & (frc[:, 1] > 0.02)
    c = frc[m]
    nturn = int(np.sum(np.diff(np.sign(np.diff(c[:, 0]))) != 0))   # Omega turning points
    return nturn >= 2, c[:, 1].max()


def a_crit(pl, ref):
    """Jump-free amplitude = saddle-node onset of the FORCED response, found by
    bisecting the drive until the response-curve first becomes multivalued
    (numerical, not the leading-order Duffing formula)."""
    s = ssm_from_system(pl.to_system(), mode=0, ref_dof=ref, order=3)
    kF = pl.h * pl.K[ref, ref]
    lo, hi = 1e-4 * kF, 0.03 * kF
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if _multivalued(pl, ref, mid)[0]:
            hi = mid
        else:
            lo = mid
    return _multivalued(pl, ref, hi)[1], s.omega0 / 2 / np.pi, abs(s.backbone_coeff())


def main():
    from design_source import heldfreq_design
    best = (None, heldfreq_design())          # held-frequency optimum (single source)
    pl_u, ref_u = build_plate(np.zeros(P.M))
    pl_o, ref_o = build_plate(best[1])
    acu, f1u, ku = a_crit(pl_u, ref_u)
    aco, f1o, ko = a_crit(pl_o, ref_o)
    print(f"unidirectional: f1={f1u:.0f}Hz  jump-free a/h < {acu:.3f}")
    print(f"steered tow   : f1={f1o:.0f}Hz  jump-free a/h < {aco:.3f}")
    print(f"-> jump-free amplitude range x{aco/acu:.2f}, jump-free drive x~{(aco/acu)**1.5:.2f}")

    # a drive between the two thresholds: baseline folds, steered does not
    F = 0.0100 * pl_u.h * pl_u.K[ref_u, ref_u]
    frc_u = ssm_frc(pl_u, ref_u, F, ZETA)
    frc_o = ssm_frc(pl_o, ref_o, F, ZETA)

    def window(frc):                                        # keep the resonance, drop the tails
        m = (frc[:, 0] > 0.97) & (frc[:, 0] < 1.14) & (frc[:, 1] > 0.02)
        return frc[m]

    def crossings(frc, Om):                                 # amplitudes at drive frequency Om
        s = frc[:, 0] - Om; out = []
        for i in np.where(np.diff(np.sign(s)) != 0)[0]:
            t = -s[i] / (frc[i + 1, 0] - frc[i, 0])
            out.append(frc[i, 1] + t * (frc[i + 1, 1] - frc[i, 1]))
        return np.sort(out)

    cu, co = window(frc_u), window(frc_o)
    # unstable (middle) branch of the folded baseline = between its two turning points
    turns = np.where(np.diff(np.sign(np.diff(cu[:, 0]))) != 0)[0] + 1
    i1, i2 = (turns[0], turns[-1]) if turns.size >= 2 else (0, len(cu))
    Ostar = 0.5 * (cu[i1, 0] + cu[i2, 0])                   # a frequency inside the jump band
    au3 = crossings(cu, Ostar); ao1 = crossings(co, Ostar)

    fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax[0].axvspan(cu[i2, 0], cu[i1, 0], color=ps.RED, alpha=0.06)
    ax[0].plot(cu[:i1 + 1, 0], cu[:i1 + 1, 1], color=ps.RED, lw=1.7, label="unidirectional")
    ax[0].plot(cu[i1:i2 + 1, 0], cu[i1:i2 + 1, 1], color=ps.RED, lw=1.1, ls=(0, (4, 3)))  # unstable
    ax[0].plot(cu[i2:, 0], cu[i2:, 1], color=ps.RED, lw=1.7)
    ax[0].plot(co[:, 0], co[:, 1], color=ps.GREEN, lw=1.9, label="steered tow")
    ax[0].axvline(Ostar, color=ps.GREY, ls=":", lw=1.0)
    if len(au3) >= 3:
        top, bot = au3[-1], au3[0]
        ax[0].annotate("", xy=(Ostar, bot + 0.008), xytext=(Ostar, top - 0.008),
                       arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=1.6, shrinkA=0, shrinkB=0))
        ax[0].text(Ostar + 0.004, 0.5 * (top + bot), "jump", color=ps.RED,
                   fontsize=8, fontweight="bold", va="center")
        ax[0].plot([Ostar] * 3, au3, "o", color=ps.RED, ms=4.5, zorder=6)
        ax[0].text(Ostar - 0.004, top + 0.004, "3 states\n(baseline)", color=ps.RED,
                   fontsize=6.8, ha="right", va="bottom")
    ax[0].plot([Ostar], ao1, "o", color=ps.GREEN, ms=5.5, zorder=6)
    ax[0].text(Ostar - 0.004, ao1[0], "1 state\n(steered)", color=ps.GREEN,
               fontsize=6.8, ha="right", va="center")
    ax[0].set_xlim(0.985, 1.085); ax[0].set_ylim(0, 0.30)
    ax[0].set_xlabel(r"drive frequency  $\Omega/\omega_1$")
    ax[0].set_ylabel(r"steady amplitude  $a/h$")
    ax[0].set_title("(a) baseline jumps; steered is smooth")
    ps.grid(ax[0]); ps.despine(ax[0])
    ax[0].legend(frameon=False, loc="upper right")

    ax[1].bar([0, 1], [acu, aco], color=[ps.GREY, ps.GREEN], width=0.55, zorder=3)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["unidirectional", "steered"])
    ax[1].set_ylabel(r"jump-free amplitude  $a/h$")
    ax[1].set_title(rf"(b) jump-free range $\times{aco/acu:.2f}$")
    for x, a in [(0, acu), (1, aco)]:
        ax[1].text(x, a + 0.004, f"{a:.3f}", ha="center", fontweight="bold")
    ax[1].set_ylim(0, max(acu, aco) * 1.25)
    ps.grid(ax[1], axis="y"); ps.despine(ax[1])

    fig.tight_layout()
    ps.save(fig, (_ROOT + "/figs/plate_jump"))


if __name__ == "__main__":
    main()
