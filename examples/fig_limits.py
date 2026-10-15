r"""Figure: what bounds the steering lever (P5).

(a) anisotropy dose-response: held-frequency steered J/J_UD against the lamina
    anisotropy E1/E2 of the family Q(s) (s = 0 isotropic ... 1.1);
(b) manufacturability front: held-frequency J/J_UD against the minimum tow radius,
    with the support-grid refinements and the pointwise (unmanufacturable) ceiling.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt

M = np.load("results/mechanism.npz")
C = np.load("results/ceiling.npz")

fig = plt.figure(figsize=(ps.FULL, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.42, left=0.075, right=0.985, bottom=0.19, top=0.9)
ax = [fig.add_subplot(gs[0, k]) for k in range(3)]
d = M["dose"]
ratio, Jopt = d[:, 1], d[:, 2]
ax[0].plot(np.r_[1.0, ratio], np.r_[1.0, Jopt], "o-", color=ps.GREEN, ms=4, lw=1.4)
ax[0].axhline(1.0, color=ps.GREY, lw=0.8, ls="--")
ax[0].annotate("isotropic:\nno lever", (1.0, 1.0), xytext=(1.05, 0.62), fontsize=7, color=ps.GREY,
               arrowprops=dict(arrowstyle="->", color=ps.GREY, lw=0.6))
ax[0].axvline(14.56, color=ps.GREY, lw=0.7, ls=":")
ax[0].text(14.56, 0.06, " CFRP", fontsize=7, color=ps.GREY)
ax[0].set_xscale("log")
from matplotlib.ticker import FixedLocator as _FL, FixedFormatter as _FF, NullLocator as _NL
ax[0].xaxis.set_major_locator(_FL([1, 2, 5, 10, 20, 30])); ax[0].xaxis.set_major_formatter(_FF(["1", "2", "5", "10", "20", "30"]))
ax[0].xaxis.set_minor_locator(_NL())
ax[0].set_xlabel(r"lamina anisotropy $E_1/E_2$")
ax[0].set_ylabel(r"$J/J_{\mathrm{UD}}$ at held $\omega_1$")
ax[0].set_ylim(0, 1.08)
ps.panel(ax[0], "a"); ps.grid(ax[0]); ps.despine(ax[0])

R, J = M["front_R"] * 1000, M["front_J"]
ax[1].plot(R, J, "o-", color=ps.GREEN, ms=4, lw=1.4, label=r"steered optimum")
ax[1].axhline(float(C["res_50_inf"]), color=ps.GREEN, lw=0.9, ls="--", label="no tow limit")
if "ceiling_J" in C.files:
    ax[1].axhline(float(C["ceiling_J"]), color=ps.RED, lw=1.0, ls=":", label="pointwise ceiling")
ax[1].set_xscale("log")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
rt = [10, 20, 50, 100, 200, 400]
ax[1].xaxis.set_major_locator(FixedLocator(rt)); ax[1].xaxis.set_major_formatter(FixedFormatter([str(r) for r in rt]))
ax[1].xaxis.set_minor_locator(NullLocator())
ax[1].set_xlabel(r"minimum tow radius $R_{\min}$ (mm)")
ax[1].set_ylabel(r"$J/J_{\mathrm{UD}}$ at held $\omega_1$")
ax[1].set_ylim(0, 1.0)
ax[1].legend(frameon=False, fontsize=7, loc="upper left")
ps.panel(ax[1], "b"); ps.grid(ax[1]); ps.despine(ax[1])

hs = np.array([50, 33, 25]); xk = np.arange(3)
j20 = [float(C[f"res_{h}_20"]) for h in hs]; jin = [float(C[f"res_{h}_inf"]) for h in hs]
ax[2].plot(xk - 0.07, j20, "o-", color=ps.GREEN, mfc="white", mew=1.2, ms=4.5, lw=1.1, label=r"$R_{\min}=20$ mm")
ax[2].plot(xk + 0.07, jin, "s--", color=ps.GREEN, ms=4, lw=1.1, label="no limit")
ax[2].axhline(float(C["ceiling_J"]), color=ps.RED, lw=1.0, ls=":", label="ceiling")
ax[2].set_xticks(xk); ax[2].set_xticklabels([f"{h}" for h in hs]); ax[2].set_xlim(-0.3, 2.3)
ax[2].set_xlabel(r"support spacing $h_s$ (mm)")
ax[2].set_ylim(0.325, 0.375); ax[2].set_yticks([0.33, 0.34, 0.35, 0.36, 0.37])
ax[2].set_ylabel(r"$J/J_{\mathrm{UD}}$ (zoom)")
ax[2].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.36), handlelength=1.8)
ps.panel(ax[2], "c"); ps.grid(ax[2]); ps.despine(ax[2])
ps.save(fig, (_ROOT + "/figs/fig_limits"))
