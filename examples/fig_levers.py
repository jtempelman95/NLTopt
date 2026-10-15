r"""Figure: the achievable (omega_1, J) set of each design class at fixed mass.

(a) the frequency--detuning plane: uniform single-ply angle (curve), uniform
    symmetric stacks (cloud + lower envelope), steered single ply, steered stack;
(b) at the unidirectional plate's frequency: the best design of each class.

    python examples/fig_levers.py        (needs results/levers.npz, mechanism.npz, ceiling.npz)
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt

L = np.load("results/levers.npz", allow_pickle=True)
w_ud, J_ud = float(L["w_ud"]), float(L["J_ud"])
S = L["S"]
fS, JS = S[:, 4] / w_ud, S[:, 5] / J_ud
F = [d for d in L["F"]]
SF = [d for d in L["SF"]]
tg = L["targets"] / w_ud


def envelope(f, J, edges):
    c = 0.5 * (edges[1:] + edges[:-1]); e = np.full(c.size, np.nan)
    for i in range(c.size):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            e[i] = J[m].min()
    return c, e


fig = plt.figure(figsize=(ps.FULL, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.3)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])

ax.scatter(fS, JS, s=1.2, color="#c9ced4", lw=0, rasterized=True, zorder=1,
           label=rf"all {len(fS):,} uniform stacks".replace(",", "{,}"))
c, e = envelope(fS, JS, np.linspace(0.95, 1.60, 40))
ax.plot(c, e, color=ps.GREY, lw=1.1, ls="--", zorder=2, label=r"lower envelope, $15^\circ$ grid")
ax.plot(L["U_f"] / w_ud, L["U_J"] / J_ud, color=ps.INK, lw=1.4, zorder=3,
        label=r"uniform single angle $\theta$")
SC = np.load("results/stack_cont.npz")
kc = SC["targets"] > 1.01                      # at 1.0 the continuous optimum is UD (the star)
ax.plot(SC["targets"][kc], np.minimum(SC["J_cont"], SC["J_single"])[kc], "D", color=ps.GREY, ms=5, mfc="white", mew=1.2,
        zorder=7, label=r"best continuous stack $[\theta_4/\theta_3/\theta_2/\theta_1]_s$")
Fp = [(d["omega1"] / w_ud, d["J"] / J_ud) for d in F if d is not None]
SFp = [(d["omega1"] / w_ud, d["J"] / J_ud) for d in SF if d is not None]
ax.plot(*zip(*Fp), "o", color=ps.GREEN, ms=3.8, zorder=9, label="steered single ply")
ax.plot(*zip(*SFp), "s", color=ps.STEEL, ms=7.0, mfc="white", mew=1.2, zorder=8,
        label="best grid stack, then steered")
ax.plot(1.0, 1.0, "*", color=ps.INK, ms=8, zorder=10)
ax.annotate("UD", (1.0, 1.0), xytext=(6, 2), textcoords="offset points", fontsize=7.5)
ax.axvline(1.0, color=ps.GREY, lw=0.6, ls=":", zorder=0)
ax.set_xlim(0.85, 1.58); ax.set_ylim(0.0, 1.2)
ax.set_xlabel(r"linear frequency $\omega_1/\omega_{1,\mathrm{UD}}$")
ax.set_ylabel(r"detuning $J/J_{\mathrm{UD}}$")
ps.panel(ax, "a"); ps.grid(ax); ps.despine(ax)
from matplotlib.lines import Line2D
h_, l_ = ax.get_legend_handles_labels()
h_[[i for i, l in enumerate(l_) if "all " in l][0]] = Line2D([], [], ls="", marker="o", ms=2.5, color="#c9ced4")
fig.legend(h_, l_, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.6, columnspacing=1.2)

# (b) held frequency comparison
M = np.load("results/mechanism.npz")
Cc = np.load("results/ceiling.npz")
held = np.abs(fS - 1.0) < 0.01
best_stack = float(np.minimum(SC["J_cont"], SC["J_single"])[np.argmin(np.abs(SC["targets"] - 1.0))])
i1 = int(np.argmin(np.abs(tg - 1.0)))
labels = ["UD\n(= best uniform)", "steered", "pointwise\nceiling"]
vals = [1.0, F[i1]["J"] / J_ud if F[i1] is not None else np.nan, float(Cc["ceiling_J"])]
cols = [ps.GREY, ps.GREEN, "#9aa5ae"]
bars = bx.bar(range(3), vals, color=cols, width=0.55)
bars[2].set_hatch("////"); bars[2].set_facecolor("white"); bars[2].set_edgecolor(ps.GREY)
for i, v in enumerate(vals):
    bx.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
bx.set_xticks(range(3)); bx.set_xticklabels(labels, fontsize=7)
bx.set_ylabel(r"$J/J_{\mathrm{UD}}$ at $\omega_1=\omega_{1,\mathrm{UD}}$")
bx.set_ylim(0, 1.15)
ps.panel(bx, "b"); ps.grid(bx, axis="y"); ps.despine(bx)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.80)
ps.save(fig, (_ROOT + "/figs/fig_levers"))
print(f"best uniform at held f: {best_stack:.3f}; steered {vals[1]:.3f}; ceiling {vals[2]:.3f}")
