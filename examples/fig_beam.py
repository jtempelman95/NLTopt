r"""Figure: the 1-D test -- symmetric (B = 0) steering of a clamped beam.

(a) optimised fibre-angle fields theta(x) (mirror symmetric) at three tow limits;
(b) the axial modulus E(x)/E_UD they produce against the UD curvature weight
    phi''^2 that sets omega_1^2 (shaded): fibres turn compliant exactly where the
    beam does not bend (the inflection points, dotted);
(c) full FE + SSM backbones: UD, the symmetric steered designs, and the
    unsymmetric bending-extension (B) design of the earlier draft.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from nltopt.laminated_beam import LaminatedBeam, _shape_derivs
from nltopt.laminate import Qbar

R = np.load("results/beam_symmetric.npz")
Bl = np.load("results/results_steered_beam.npz")
L = float(R["L"]); xk = R["xk"]
x = np.linspace(0, L, 400)
field = lambda c: np.interp(np.where(x <= L / 2, x, L - x), xk, c)

# UD curvature weight phi''^2 at element centres
b = LaminatedBeam(120, L, 0.002, 0.02)
ix = np.ix_(b.free, b.free)
_, V = eigh(b.K_full[ix], b.M_full[ix], subset_by_index=[0, 0])
phi = np.zeros(b.ndof); phi[b.free] = V[:, 0]
rows = _shape_derivs(b.Le)
wpp = np.array([rows[1][2] @ phi[3 * e:3 * e + 6] for e in range(b.n_elem)])
wpp2 = wpp ** 2 / (wpp ** 2).max()

designs = [("k900_f98", r"$900^\circ$/m, $f\geq0.98$", ps.GREEN, "-"),
           ("k900_f90", r"$900^\circ$/m, $f\geq0.90$", ps.GREEN, "--"),
           ("k99999_f98", r"no tow limit, $f\geq0.98$", ps.STEEL, (0, (5, 1.5, 1, 1.5)))]

fig, ax = plt.subplots(1, 3, figsize=(ps.FULL, 2.75))
for tag, lab, col, ls in designs:
    ax[0].plot(x / L, np.rad2deg(field(R[tag + "_c"])), color=col, ls=ls, lw=1.4, label=lab)
    E = np.array([Qbar(t)[0, 0] for t in field(R[tag + "_c"])]) / Qbar(0.0)[0, 0]
    ax[1].plot(x / L, E, color=col, ls=ls, lw=1.4)
ax[1].fill_between(b.xe / L, 0, wpp2, color=ps.FILL_MUTE, zorder=0, label=r"UD ${\varphi''}^{2}$ (scaled)")
for a in ax[:2]:
    for xi in (0.224, 0.776):
        a.axvline(xi, color=ps.GREY, ls=":", lw=0.8)
    a.set_xlabel(r"$x/L$"); ps.grid(a); ps.despine(a)
ax[0].set_ylabel(r"fiber angle $\theta$ (deg)")
ax[1].set_ylabel(r"$E(x)/E_{\mathrm{UD}}$"); ax[1].set_ylim(0, 1.05)
ax[1].text(0.5, 0.12, r"UD ${\varphi''}^{2}$" + "\n(scaled)", ha="center", va="center", fontsize=7, color=ps.GREY)
ps.panel(ax[0], "a"); ps.panel(ax[1], "b")

o, a = R["base_curve"]
ax[2].plot(o, a, color=ps.GREY, lw=1.6, label="UD")
for tag, lab, col, ls in designs:
    o, a = R[tag + "_curve"]
    ax[2].plot(o, a, color=col, ls=ls, lw=1.4, label=lab)

ax[2].set_xlabel(r"$\Omega/\omega_1$"); ax[2].set_ylabel(r"mid-span amplitude $a/h$")

ps.panel(ax[2], "c"); ps.grid(ax[2]); ps.despine(ax[2])
h_, l_ = ax[2].get_legend_handles_labels()
h_.append(plt.Line2D([], [], color=ps.GREY, ls=":", lw=0.8)); l_.append("UD inflection points")
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.2, handlelength=2.6,
           columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=0.8, rect=(0, 0.15, 1, 1))
ps.save(fig, (_ROOT + "/figs/fig_beam"))
