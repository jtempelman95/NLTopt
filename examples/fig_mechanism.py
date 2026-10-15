r"""Figure: the in-plane separation mechanism on the plate (results/mechanism.npz).

(a) bending-energy density e_K of the UD fundamental (sets omega_1^2) with the
    fibre direction that maximises it locally (principal curvature, phi_K);
(b) relaxed-stretch density e_G (sets gamma) with the direction that minimises it
    (perpendicular to the principal relaxed stretch, phi_G + 90 deg);
(c) the angle between the two preferred directions -- where it is large, a single
    fibre direction cannot serve both, and steering has leverage;
(d) the pointwise first-order design theta*(x) predicted from the UD fields alone;
(e) the optimised, curvature-limited toolpath theta_opt(x);
(f) what changed: raw cubic, relaxation, gamma, omega_1^2 (normalised by UD).
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt
from nltopt.relaxed_plate import RelaxedPlate

M = np.load("results/mechanism.npz")
P = RelaxedPlate()
ny, nx = P.ny, P.nx
X, Y = P.X * 1000, P.Y * 1000                      # mm
G = lambda f: np.asarray(f).reshape(ny, nx)


def directors(ax, ang, amp, step=8, color=ps.INK, scale=7.0, width=0.006):
    a = G(ang)[step // 2::step, step // 2::step]
    s = G(amp)[step // 2::step, step // 2::step]
    s = s / s.max()
    xx, yy = X[step // 2::step, step // 2::step], Y[step // 2::step, step // 2::step]
    L = scale * (0.35 + 0.65 * s)
    ax.quiver(xx, yy, L * np.cos(a), L * np.sin(a), angles="xy", scale_units="xy", scale=1,
              headwidth=0, headlength=0, headaxislength=0, pivot="middle", color=color,
              width=width)


W, H = ps.FULL, 3.40
fig = plt.figure(figsize=(W, H))
mw, mh = 1.40, 1.05                                    # every map: same size (200 x 150 mm, equal aspect)
X0, Y0 = (0.40, 2.02, 4.42), (0.36, 2.02)


def box(col, row, w=mw, h=mh):
    return [X0[col] / W, Y0[row] / H, w / W, h / H]


def field(col, row, f, cmap, vmin=None, vmax=None, ylab=False):
    ax = fig.add_axes(box(col, row))
    im = ax.pcolormesh(X, Y, G(f), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlim(0, 200); ax.set_ylim(0, 150)
    ax.set_xticks([0, 100, 200]); ax.set_yticks([0, 75, 150]); ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel(r"$x$ (mm)", fontsize=7.5, labelpad=1)
    if ylab:
        ax.set_ylabel(r"$y$ (mm)", fontsize=7.5, labelpad=1)
    else:
        ax.set_yticklabels([])
    return ax, im


def cbar(im, x, row, label, ticks=None):
    cax = fig.add_axes([x / W, Y0[row] / H, 0.07 / W, mh / H])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(label, fontsize=7.5, labelpad=2)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def inset(ax, text):
    ax.text(4, 4, text, fontsize=6.5, ha="left", va="bottom", linespacing=1.15,
            bbox=dict(fc="white", ec=ps.GREY, lw=0.4, pad=1.2, alpha=0.92), zorder=10)


def title(ax, letter, text):
    ax.set_title(rf"(\textbf{{{letter}}}) " + text, loc="left", pad=3, fontsize=7.5)


eK, eG = M["ud_eK"], M["ud_eG"]
# held frequency: Lambda < 0, so the frequency constraint favors fibers ACROSS the principal
# curvature (phi_K + 90); the stretch favors fibers ACROSS the principal stretch (phi_G + 90).
a, im = field(0, 1, eK / eK.max(), "Greys", vmin=0, vmax=1, ylab=True)
directors(a, M["ud_phiK"] + np.pi / 2, M["ud_aK2"], color=ps.STEEL, scale=9.0, width=0.009)
title(a, "a", r"$e_K$, integrates to $\omega_1^2$")
a, im = field(1, 1, eG / eG.max(), "Greys", vmin=0, vmax=1)
directors(a, M["ud_phiG"] + np.pi / 2, M["ud_aG2"], color=ps.RED, scale=9.0, width=0.009)
title(a, "b", r"$e_G$, integrates to $\gamma$")
cbar(im, X0[1] + mw + 0.06, 1, r"density / its max", ticks=[0, 0.5, 1])
conflict = np.rad2deg(np.abs(np.angle(np.exp(2j * (M["ud_phiK"] - M["ud_phiG"])))) / 2)
wgt = (eK / eK.sum() + eG / eG.sum())
a, im = field(2, 1, conflict, "magma_r", vmin=0, vmax=90, ylab=True)
title(a, "c", r"conflict $\chi$")
cbar(im, X0[2] + mw + 0.06, 1, r"$\chi$ (deg)", ticks=[0, 30, 60, 90])

a, im = field(0, 0, np.rad2deg(np.angle(np.exp(2j * M["theta_star"])) / 2), "twilight_shifted",
              vmin=-90, vmax=90, ylab=True)
directors(a, M["theta_star"], np.ones(P.Nq), step=8, scale=8.0)
title(a, "d", r"pointwise design $\theta^\star$")
th = M["theta_opt"]
a, im = field(1, 0, np.rad2deg(np.angle(np.exp(2j * th)) / 2), "twilight_shifted", vmin=-90, vmax=90)
_sp = a.streamplot(X[0], Y[:, 0], np.cos(G(th)), np.sin(G(th)), density=0.9,
                   color=ps.INK, linewidth=0.6, arrowsize=0)
import matplotlib.patheffects as pe
_sp.lines.set_path_effects([pe.Stroke(linewidth=1.5, foreground="white", alpha=0.8), pe.Normal()])
a.set_xlim(0, 200); a.set_ylim(0, 150)
title(a, "e", r"optimized $\theta_{\mathrm{opt}}$, tow paths")
cbar(im, X0[1] + mw + 0.06, 0, r"fiber angle $\theta$ (deg)", ticks=[-90, -45, 0, 45, 90])

bx = fig.add_axes([(X0[2] + 0.30) / W, Y0[0] / H, (W - X0[2] - 0.38) / W, mh / H])
keys = [("knl", r"$k_{nl}$"), ("relax", r"$k_{r}$"),
        ("gamma", r"$\gamma$"), ("om2", r"$\omega_1^2$"), ("J", r"$J$")]
ud = [float(M[f"ud_{k}"]) for k, _ in keys]
op = [float(M[f"opt_{k}"]) for k, _ in keys]
x = np.arange(len(keys))
bx.bar(x, np.array(op) / np.array(ud), color=[ps.GREEN] * 3 + [ps.GREY, ps.GREEN], width=0.6)
bx.axhline(1.0, color=ps.GREY, lw=0.8, ls="--")
for i in range(len(keys)):
    bx.text(i, op[i] / ud[i] + 0.04, f"{op[i]/ud[i]:.2f}", ha="center", fontsize=7)
bx.set_xticks(x); bx.set_xticklabels([l for _, l in keys], fontsize=7.5)
bx.tick_params(axis="y", labelsize=7)
bx.set_ylabel("optimized / UD", fontsize=7.5); bx.set_ylim(0, 1.3)
bx.set_title(r"(\textbf{f}) terms of $\gamma=k_{nl}-k_r$", loc="left", pad=3, fontsize=7.5)
ps.grid(bx, axis="y"); ps.despine(bx)
ps.save(fig, (_ROOT + "/figs/fig_mechanism"))
print("energy-weighted mean conflict (deg):", float(np.sum(wgt * conflict) / wgt.sum()))
