r"""Physical schematic of the problem (paper Fig. 1).

(a) the simply supported CFRP plate (200 x 150 x 2 mm) deflected in its fundamental mode, with
    the tow paths of the actual held-frequency steered design drawn on it, the supported
    edges, and the harmonic center load;
(b) what one material point sees: the fiber direction theta, the principal curvature of the
    mode (which sets omega_1^2) and the principal relaxed stretch (which sets the hardening
    gamma) -- two directions that a single fiber angle must arbitrate;
(c) the consequence sketched on the backbone / forced response: at the same linear frequency,
    steering lowers the hardening, so the fold (jump) of the unidirectional panel disappears.

    python examples/fig_schematic.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import levers_study as LS
from design_source import heldfreq_design

P = LS.PLATE
th = (LS.csrbf(P)[0] @ heldfreq_design()).reshape(P.ny, P.nx)       # design angle on the grid
a, b = P.a * 1000, P.b * 1000
xs = (np.arange(P.nx) + 0.5) * a / P.nx
ys = (np.arange(P.ny) + 0.5) * b / P.ny
W = 28.0                                                             # visual deflection (mm)
w = lambda x, y: W * np.sin(np.pi * x / a) * np.sin(np.pi * y / b)


def theta_at(x, y):
    i = np.clip(np.searchsorted(ys, y), 0, P.ny - 1); j = np.clip(np.searchsorted(xs, x), 0, P.nx - 1)
    return th[i, j]


def tow(y0, ds=1.0):
    """Integrate a tow center-line through the director field from the left edge."""
    pts = [(0.0, y0)]
    x, y = 0.0, y0
    for _ in range(600):
        t = theta_at(x, y)
        x, y = x + ds * np.cos(t), y + ds * np.sin(t)
        if not (0 <= x <= a and 0 <= y <= b):
            break
        pts.append((x, y))
    return np.array(pts)


fig = plt.figure(figsize=(ps.FULL, 2.55))
# ---------------------------------------------------------------- (a) plate, oblique projection
W = 40.0
pr = lambda x, y, z: (x + 0.55 * y, 0.42 * y + z)                     # oblique view
ax = fig.add_axes([0.0, 0.02, 0.45, 0.9]); ax.set_aspect("equal"); ax.axis("off")
gx, gy = np.linspace(0, a, 81), np.linspace(0, b, 61)
GX, GY = np.meshgrid(gx, gy); GZ = w(GX, GY)
PX, PY = pr(GX, GY, GZ)
shade = 0.86 + 0.12 * np.cos(np.pi * GX / a) * 0.0 - 0.10 * (GZ / W)
ax.pcolormesh(PX, PY, GZ, cmap="Greys", vmin=-0.6 * W, vmax=3.2 * W, shading="gouraud", rasterized=True, zorder=1)
for xx in np.linspace(0, a, 9):                                       # faint mesh lines show the mode shape
    yy = np.linspace(0, b, 50); ax.plot(*pr(xx + 0 * yy, yy, w(xx + 0 * yy, yy)), color="white", lw=0.4, zorder=2)
for yy0 in np.linspace(0, b, 7):
    xx = np.linspace(0, a, 60); ax.plot(*pr(xx, yy0 + 0 * xx, w(xx, yy0 + 0 * xx)), color="white", lw=0.4, zorder=2)
_f, _a = plt.subplots()                                                # actual steered tow paths: streamlines
_sp = _a.streamplot(xs, ys, np.cos(th), np.sin(th), density=(1.6, 1.1), arrowsize=0, integration_direction="both")
plt.close(_f)
for seg in _sp.lines.get_segments():
    ax.plot(*pr(seg[:, 0], seg[:, 1], w(seg[:, 0], seg[:, 1]) + 0.2), color=ps.GREEN, lw=0.55, zorder=3,
            solid_capstyle="round")
edge = [(0, 0), (a, 0), (a, b), (0, b), (0, 0)]
ax.plot(*pr(np.array([e[0] for e in edge]), np.array([e[1] for e in edge]), np.zeros(5)), color=ps.INK, lw=1.1, zorder=4)
for t in np.linspace(0.08, 0.92, 6):                                  # simple supports
    for (xa, ya) in ((t * a, 0), (0, t * b), (a, t * b)):
        X0, Y0 = pr(xa, ya, 0); ax.plot([X0], [Y0 - 5], marker="^", color=ps.GREY, ms=3.2, zorder=4)
xc, yc = pr(a / 2, b / 2, W)
ax.annotate("", xy=(xc, yc + 2), xytext=(xc, yc + 48), arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=1.4), zorder=6)
ax.text(xc + 5, yc + 40, r"$F\cos\Omega t$", color=ps.RED, fontsize=8, zorder=6)
X0, Y0 = pr(a / 2, 0, 0); ax.text(X0, Y0 - 22, r"$a=200$ mm, all edges simply supported", fontsize=7.5, ha="center")
X0, Y0 = pr(a, b / 2, 0); ax.text(X0 + 16, Y0 - 10, r"$b=150$ mm", fontsize=7.5)
X0, Y0 = pr(0, b, 0); ax.text(X0 - 95, Y0 + 34, "$h=2$ mm CFRP ply;\nsteered tow paths\n(held-frequency design)", fontsize=7.5, color=ps.GREEN, va="bottom")
ax.set_xlim(-60, a + 0.55 * b + 40); ax.set_ylim(-32, 0.42 * b + W + 58)
fig.text(0.01, 0.95, r"(\textbf{a})", fontsize=9)

# ---------------------------------------------------------------- (b) one material point
bx = fig.add_axes([0.45, 0.12, 0.24, 0.76]); bx.set_aspect("equal"); bx.axis("off")
bx.add_patch(Rectangle((-1, -1), 2, 2, fc="#f5f7f8", ec=ps.GREY, lw=0.8))
tf = np.deg2rad(25)
for o in np.linspace(-1.6, 1.6, 11):                                  # fibers at angle theta
    c, s = np.cos(tf), np.sin(tf); p0 = np.array([-2 * c - o * s, -2 * s + o * c]); p1 = p0 + 4 * np.array([c, s])
    t = np.linspace(0, 1, 60); seg = p0[None] + t[:, None] * (p1 - p0)[None]
    seg = seg[(np.abs(seg[:, 0]) <= 1) & (np.abs(seg[:, 1]) <= 1)]
    if len(seg) > 1:
        bx.plot(seg[:, 0], seg[:, 1], color=ps.GREEN, lw=0.8)
bx.annotate("", xy=(0, 1.42), xytext=(0, -1.42), arrowprops=dict(arrowstyle="<->", color=ps.STEEL, lw=1.6))
bx.text(-0.12, 1.46, r"principal curvature $\kappa$", color=ps.STEEL, fontsize=7.5, va="bottom", ha="right")
bx.text(-0.12, 1.78, r"$\omega_1^2=\int\kappa^{\!\top}\!D(\theta)\,\kappa$", color=ps.STEEL, fontsize=7.5, va="bottom", ha="right")
bx.annotate("", xy=(1.42, 0), xytext=(-1.42, 0), arrowprops=dict(arrowstyle="<->", color=ps.RED, lw=1.6))
bx.text(1.48, -0.06, "relaxed\nstretch\n" + r"$\varepsilon_r$", color=ps.RED, fontsize=7.5, va="top")
bx.text(-1.0, -1.32, r"$\gamma=\tfrac12\int\varepsilon_r^{\!\top}\!A(\theta)\,\varepsilon_r$", color=ps.RED, fontsize=7.5, va="top")
bx.annotate("", xy=(0.75 * np.cos(tf), 0.75 * np.sin(tf)), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=ps.GREEN, lw=1.3))
bx.text(0.62, 0.12, r"$\theta$", color=ps.GREEN, fontsize=9)
bx.set_xlim(-1.6, 2.25); bx.set_ylim(-1.65, 2.05)
fig.text(0.45, 0.95, r"(\textbf{b})", fontsize=9)

# ---------------------------------------------------------------- (c) consequence
cx = fig.add_axes([0.765, 0.2, 0.225, 0.68])
amp = np.linspace(0, 1, 200)
for beta, col, lab in ((0.35, ps.GREY, "UD"), (0.12, ps.GREEN, "steered")):
    cx.plot(1 + beta * amp ** 2, amp, color=col, lw=1.4, label=lab)
mu = 0.035                                                            # forced response, Prop. 3 form:
sp = np.concatenate([-np.logspace(1, -4, 600), np.logspace(-4, 1, 600)])   # x = F^2/(s^2+mu^2), Om = 1 + beta x + s
for beta, col in ((0.35, ps.GREY), (0.12, ps.GREEN)):
    F2 = 0.40 * mu ** 2
    x = F2 / (sp ** 2 + mu ** 2)
    cx.plot(1 + beta * x + sp, np.sqrt(x), color=col, lw=0.9, ls=(0, (3, 1.5)))

cx.set_xlim(0.85, 1.42); cx.set_ylim(0, 0.95)
cx.set_xticks([1.0]); cx.set_xticklabels([r"$\omega_1$"]); cx.set_yticks([])
cx.set_xlabel(r"drive frequency $\Omega$", fontsize=8); cx.set_ylabel(r"amplitude $a$", fontsize=8)
cx.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.4)
ps.despine(cx); fig.text(0.735, 0.95, r"(\textbf{c})", fontsize=9)
ps.save(fig, (_ROOT + "/figs/fig_schematic"))
