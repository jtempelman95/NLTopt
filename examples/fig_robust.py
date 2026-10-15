r"""Figure: how robust is the held-frequency lever?  (Revision R1.)

(a) other plates: steered optimum vs the strongest uniform laminate at the same frequency;
(b) the lever follows the conflict chi between the two strain fields (P5);
(c) jump-free amplitude vs damping level and law (ratio steered/UD is damping-independent);
(d) as-laid tow density tau of the steered field (streamline spreading) with its tow paths;
(e) as-built laminates (tow cut/add with N tows per course; gaps or overlaps; 2 deg angle
    error) in the frequency--hardening plane, against the steered and uniform fronts.

    python examples/fig_robust.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RES = _os.path.join(_ROOT, "results")
names = ["ss_43", "ss_11", "ss_21", "cl_43", "mv_43"]
labels = ["SS\n4:3*", "SS\n1:1", "SS\n2:1", "CL\n4:3", "SM\n4:3"]
G = {n: np.load(_os.path.join(RES, f"r1_generality_{n}.npz")) for n in names}
C = np.load(_os.path.join(RES, "r1_conflict.npz"))
D = np.load(_os.path.join(RES, "r1_damping.npz"))["rows"]
A = np.load(_os.path.join(RES, "r1_asbuilt.npz"), allow_pickle=True)
L = np.load(_os.path.join(RES, "levers.npz"), allow_pickle=True)

fig = plt.figure(figsize=(ps.FULL, 4.75))
# ------------------------------------------------------------ (a) other plates
ax = fig.add_axes([0.075, 0.615, 0.33, 0.33])
x = np.arange(len(names))
uni = [min(float(G[n]["J_stack"] / G[n]["J_ud"]), float(G[n]["J_single"] / G[n]["J_ud"])) for n in names]
ste = [float(G[n]["J_steer"] / G[n]["J_ud"]) for n in names]
ax.bar(x - 0.19, uni, 0.36, color=ps.GREY, label=r"best uniform at $\omega_{1,\mathrm{UD}}$")
ax.bar(x + 0.19, ste, 0.36, color=ps.GREEN, label="steered, $R_{\\min}=20$ mm")
for i in range(len(names)):
    ax.text(x[i] + 0.19, ste[i] + 0.02, f"{ste[i]:.2f}", ha="center", fontsize=6.6)
    if uni[i] < 0.999:
        ax.text(x[i] - 0.19, uni[i] + 0.02, f"{uni[i]:.2f}", ha="center", fontsize=6.6)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6, linespacing=0.95)
ax.set_ylabel(r"$J/J_{\mathrm{UD}}$ at $\omega_{1,\mathrm{UD}}$"); ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1, handlelength=1.2, borderaxespad=0.1)
ax.set_title(r"(\textbf{a}) other plates, frequency held", loc="left", fontsize=8, pad=3)
ps.grid(ax, axis="y"); ps.despine(ax)

# ------------------------------------------------------------ (b) conflict
bx = fig.add_axes([0.49, 0.615, 0.20, 0.33])
mk = ["o", "o", "o", "s", "D"]
cols = [ps.GREEN, ps.GREEN, ps.GREEN, ps.STEEL, ps.RED]
chi, Js = C["rows"][:, 0], C["rows"][:, 1]
ss = np.array([0, 2, 1])                                         # SS immovable, ordered by chi
bx.plot(chi[[2, 0, 1]], Js[[2, 0, 1]], "-", color=ps.GREEN, lw=0.8, zorder=1)
short = ["SS 4:3", "SS 1:1", "SS 2:1", "CL", "SM"]
off = [(4, 3), (-4, -9), (4, 2), (-6, -9), (4, 2)]
ha = ["left", "right", "left", "right", "left"]
for i in range(len(names)):
    bx.plot(chi[i], Js[i], mk[i], color=cols[i], ms=5, mfc="white" if i > 2 else cols[i], mew=1.2, zorder=3)
    bx.annotate(short[i], (chi[i], Js[i]), xytext=off[i], textcoords="offset points", fontsize=6.4, ha=ha[i])
bx.set_xlabel(r"mean conflict $\chi$ (deg)"); bx.set_ylabel(r"steered $J/J_{\mathrm{UD}}$")
bx.set_xlim(15, 55); bx.set_ylim(0, 0.9)
bx.set_title(r"(\textbf{b}) lever vs.\ conflict", loc="left", fontsize=8, pad=3)
ps.grid(bx); ps.despine(bx)

# ------------------------------------------------------------ (c) damping
cx = fig.add_axes([0.785, 0.615, 0.20, 0.33])
m = D[:, 1] == 0
cx.loglog(100 * D[m, 0], D[m, 2], "-s", color=ps.GREY, ms=3.6, mfc="white", mew=1.0, label="UD")
cx.loglog(100 * D[m, 0], D[m, 3], "-o", color=ps.GREEN, ms=3.6, mfc="white", mew=1.0, label="steered")
s = D[:, 1] == 1
cx.loglog(100 * D[s, 0] * 1.18, D[s, 2], "x", color=ps.GREY, ms=5, mew=1.2)
cx.loglog(100 * D[s, 0] * 1.18, D[s, 3], "x", color=ps.GREEN, ms=5, mew=1.2, label=r"$\boldsymbol C\propto\boldsymbol K$")
cx.set_xlabel(r"damping ratio $\zeta$ (\%)"); cx.set_ylabel(r"jump-free amplitude $a_c/h$")
cx.set_xticks([0.5, 1, 2, 4]); cx.set_xticklabels(["0.5", "1", "2", "4"])
cx.set_yticks([0.1, 0.2, 0.3, 0.5]); cx.set_yticklabels(["0.1", "0.2", "0.3", "0.5"])
cx.minorticks_off()
r = D[m, 4]
cx.text(0.52, 0.44, rf"ratio ${r.min():.2f}$" + "\n" + r"at every $\zeta$", fontsize=6.8, color=ps.GREEN, ha="left", va="top")
cx.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.4, borderaxespad=0.1)
cx.set_title(r"(\textbf{c}) damping level and law", loc="left", fontsize=8, pad=3)
ps.grid(cx); ps.despine(cx)

# ------------------------------------------------------------ (d) tow density
nx, ny = int(A["nx"]), int(A["ny"])
X = A["qx"].reshape(ny, nx) * 1000; Y = A["qy"].reshape(ny, nx) * 1000
tau = A["tau"].reshape(ny, nx)
HD = 0.335 * 6.5 * 0.75 / 4.75
dx = fig.add_axes([0.075, 0.085, 0.335, HD])
im = dx.pcolormesh(X, Y, np.log2(tau), cmap="PuOr_r", norm=TwoSlopeNorm(0, -4, 4), shading="auto", rasterized=True)
sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import levers_study as LS
from design_source import heldfreq_design
th = (LS.PHI @ heldfreq_design()).reshape(ny, nx)
dx.streamplot(X[0], Y[:, 0], np.cos(th), np.sin(th), density=0.7, color=ps.INK, linewidth=0.5, arrowsize=0)
dx.set_aspect("equal"); dx.set_xlim(0, 200); dx.set_ylim(0, 150)
dx.set_xticks([0, 100, 200]); dx.set_yticks([0, 75, 150])
dx.set_xlabel(r"$x$ (mm)"); dx.set_ylabel(r"$y$ (mm)")
cax = fig.add_axes([0.42, 0.085, 0.012, HD])
cb = fig.colorbar(im, cax=cax, ticks=[-4, -2, 0, 2, 4])
cb.ax.set_yticklabels(["1/16", "1/4", "1", "4", "16"], fontsize=6.6)
cb.set_label(r"uncut tow density $\tau$", fontsize=7)
dx.set_title(r"(\textbf{d}) tow spreading along the tow paths", loc="left", fontsize=8, pad=3)

# ------------------------------------------------------------ (e) as-built plane
ex = fig.add_axes([0.585, 0.085, 0.40, HD])
w_ud, J_ud = float(L["w_ud"]), float(L["J_ud"])
Fp = np.array([(d["omega1"] / w_ud, d["J"] / J_ud) for d in L["F"] if d is not None])
ex.plot(Fp[:, 0], Fp[:, 1], "-", color=ps.GREEN, lw=0.9, zorder=2, label="steered front (fixed mass)")
keys = list(A["res_keys"]); R = A["res"]
MC = A["MC"]
from matplotlib.patches import Ellipse
cov = np.cov(MC[:, 1], MC[:, 0]); ev, evec = np.linalg.eigh(cov)
ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
ex.add_patch(Ellipse((MC[:, 1].mean(), MC[:, 0].mean()), 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=ang,
                     fc="#d9dde2", ec=ps.GREY, lw=0.8, zorder=3, label=r"$2^\circ$ angle error ($\pm2\sigma$)"))
ex.plot(*R[keys.index("ideal")][[1, 0]], "o", color=ps.GREEN, ms=6, zorder=5, label="design (idealized)")
for N, ms in ((4, 6.5), (8, 5.5), (16, 4.5)):
    g, o = R[keys.index(f"GAP{N}")], R[keys.index(f"OVL{N}")]
    ex.plot(g[1], g[0], "v", color=ps.RED, ms=ms, mfc="white", mew=1.1, zorder=6,
            label="gaps, 0\\% coverage" if N == 4 else None)
    ex.plot(o[1], o[0], "^", color=ps.STEEL, ms=ms, mfc="white", mew=1.1, zorder=6,
            label="overlaps, 100\\% coverage" if N == 4 else None)
    ex.annotate(rf"$N={N}$: ${100*(o[2]-1):+.0f}\%$", (o[1], o[0]), xytext={8: (5, 5), 16: (-3, -9)}.get(N, (-6, -16)),
                textcoords="offset points", fontsize=6.3, color=ps.STEEL, ha={8: "left", 16: "right"}.get(N, "center"), va="bottom" if N == 8 else "top")
_g = [R[keys.index(f"GAP{N}")] for N in (4, 8, 16)]
ex.annotate(r"gaps, $N=4,8,16$:" + "\n" + r"mass " + ", ".join(rf"${100*(g_[2]-1):+.0f}$" for g_ in _g) + r"\%",
            (_g[0][1], _g[0][0]), xytext=(-2, 14), textcoords="offset points", fontsize=6.3, color=ps.RED, ha="center", va="bottom")
ex.set_xlim(0.94, 1.15); ex.set_ylim(0.22, 0.46)
ex.set_xlabel(r"$\omega_1/\omega_{1,\mathrm{UD}}$"); ex.set_ylabel(r"$J/J_{\mathrm{UD}}$")
ex.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.3, borderaxespad=0.1, ncol=1)
ex.set_title(r"(\textbf{e}) as built: tow cut/add, $N$ tows per course", loc="left", fontsize=8, pad=3)
ps.grid(ex); ps.despine(ex)
ps.save(fig, _os.path.join(_ROOT, "figs", "fig_robust"))
