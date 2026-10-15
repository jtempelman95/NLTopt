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


fig = plt.figure(figsize=(ps.FULL, 3.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 0.80], wspace=0.62)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])

# proved region: below omega_UD no uniform laminate exists (Corollary 1)
ax.axvspan(0.85, 1.0, color="#eef0f3", zorder=0, lw=0)
ax.text(0.925, 1.13, "no uniform\nlaminate\n(proved)", ha="center", va="top", fontsize=6.8, color=ps.GREY)
ax.scatter(fS, JS, s=1.2, color="#c9ced4", lw=0, rasterized=True, zorder=1,
           label=rf"uniform stacks, $15^\circ$ grid (all {len(fS):,})".replace(",", "{,}"))
c, e = envelope(fS, JS, np.linspace(0.95, 1.60, 40))
ax.plot(c, e, color="#8f98a1", lw=0.8, ls=(0, (2.5, 1.5)), zorder=1.5, label=r"lower envelope of the grid stacks")
ax.plot(L["U_f"] / w_ud, L["U_J"] / J_ud, color=ps.INK, lw=1.7, zorder=3,
        label=r"uniform single angle (all $\theta$)")
SC = np.load("results/stack_cont.npz")
kc = SC["targets"] > 1.01                      # at 1.0 the continuous optimum is UD (the star)
ax.plot(SC["targets"][kc], np.minimum(SC["J_cont"], SC["J_single"])[kc], "D", color=ps.INK, ms=5.5, mfc="white", mew=1.2,
        zorder=7, label=r"best continuous stack (local optimum)")
Fp = [(d["omega1"] / w_ud, d["J"] / J_ud) for d in F if d is not None]
SFp = [(d["omega1"] / w_ud, d["J"] / J_ud) for d in SF if d is not None]
ax.plot(*zip(*Fp), "-o", color=ps.GREEN, ms=5.0, lw=0.9, zorder=9, label="steered single ply (local optimum)")
ax.plot(*zip(*SFp), "s", color=ps.STEEL, ms=8.0, mfc="none", mew=1.2, zorder=8,
        label="best grid stack, then steered")
ax.plot(1.0, 1.0, "*", color=ps.INK, ms=10, zorder=10)
ax.annotate("UD", (1.0, 1.0), xytext=(6, 2), textcoords="offset points", fontsize=7.5)
i1 = int(np.argmin(np.abs(tg - 1.0)))
ax.annotate("", xy=(1.0, F[i1]["J"] / J_ud + 0.03), xytext=(1.0, 0.96),
            arrowprops=dict(arrowstyle="-|>", color=ps.GREEN, lw=1.1), zorder=11)
ax.text(1.012, 0.66, rf"$-{100*(1-F[i1]['J']/J_ud):.0f}\%$" + "\nat held $\\omega_1$", fontsize=7, color=ps.GREEN,
        ha="left", va="center")
ax.set_xlim(0.85, 1.58); ax.set_ylim(0.0, 1.2)
ax.set_xlabel(r"linear frequency $\omega_1/\omega_{1,\mathrm{UD}}$")
ax.set_ylabel(r"detuning $J/J_{\mathrm{UD}}$")
ps.panel(ax, "a"); ps.grid(ax); ps.despine(ax)
from matplotlib.lines import Line2D
h_, l_ = ax.get_legend_handles_labels()
h_[[i for i, l in enumerate(l_) if "grid (all" in l][0]] = Line2D([], [], ls="", marker="o", ms=3, color="#c9ced4")
order = [2, 0, 1, 3, 4, 5]
fig.legend([h_[i] for i in order], [l_[i] for i in order], frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.8, columnspacing=1.0)

# (b) the held-frequency result in every model
Cc = np.load("results/ceiling.npz")
R1 = np.load("results/r1_matched.npz")
best_stack = float(np.minimum(SC["J_cont"], SC["J_single"])[np.argmin(np.abs(SC["targets"] - 1.0))])
items = [("UD (best uniform)", 1.0, 1.0, ps.GREY, None),
         ("design model", F[i1]["J"] / J_ud, F[i1]["omega1"] / w_ud, ps.GREEN, None),
         ("refined basis, re-held", float(R1["ref_held_J"]), float(R1["ref_held_f"]), ps.GREEN, "...."),
         ("FE model, as designed", float(R1["fe_asis_J"]), float(R1["fe_asis_f"]), "#9cc9b3", None),
         ("FE model, re-matched", float(R1["fe_matched_J"]), float(R1["fe_matched_f"]), ps.GREEN, "////"),
         ("pointwise ceiling", float(Cc["ceiling_J"]), np.nan, "white", "xxxx")]
import os
if os.path.exists("results/r1_full_matched.npz"):
    FM = np.load("results/r1_full_matched.npz")
    items.insert(5, ("full plate (SSM), as designed", float(FM["ssm_ratio_asis"]), float(FM["f_asis"]), "#9cc9b3", None))
    items.insert(6, ("full plate (SSM), re-matched", float(FM["ssm_ratio_matched"]), float(FM["f_matched"]), ps.GREEN, "xx"))
yy = np.arange(len(items))[::-1]
for y, (lab, v, f, col, hat) in zip(yy, items):
    bx.barh(y, v, color=col, height=0.62, edgecolor=ps.GREY if col == "white" else "white", hatch=hat, lw=0.6)
    bx.text(v + 0.02, y, f"{v:.3f}", ha="left", va="center", fontsize=7)
labs = [lab + ("\n" + rf"$\omega_1={f:.3f}\,\omega_{{1,\mathrm{{UD}}}}$" if np.isfinite(f) else "\n(unmanufacturable; estimate)")
        for lab, v, f, col, hat in items]
bx.set_yticks(yy); bx.set_yticklabels(labs, fontsize=6.8, linespacing=1.0)
bx.set_xlabel(r"$J/J_{\mathrm{UD}}$ at held frequency")
bx.set_xlim(0, 1.18); bx.set_ylim(-0.6, len(items) - 0.4)
ps.panel(bx, "b"); ps.grid(bx, axis="x"); ps.despine(bx)
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.80)
ps.save(fig, (_ROOT + "/figs/fig_levers"))
print(f"best uniform at held f: {best_stack:.3f}; items", [(it[0].replace(chr(10),' '), round(it[1],3)) for it in items])
