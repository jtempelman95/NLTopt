r"""Flutter design/result figure (LaTeX-typeset, vector PDF):
 (a) the flutter boundary q_cr vs uniform ply angle -- the bend-twist 'flutter hole' near
     30-40 deg that craters a uniform layup at the required 300 Hz;
 (b) the manufacturable steered tow layout that recovers the boundary (9x, held f1 & mass).
Run: python plate_flutter_result_fig.py"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
import numpy as np
import paperstyle as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from wavetopo.cfrp import CSRBFMapping
import plate_bfs_flutter as BF
import plate_flutter as PF

mesh = BF.Mesh(16, 12)
front = BF.uniform_front(mesh, range(0, 91, 2))
deg = np.array([p[0] for p in front]); f1u = np.array([p[1] for p in front]); qu = np.array([p[2] for p in front]) / 1e6
near = np.abs(f1u - 300.0) < 15
bu = qu[near].max(); bu_ang = deg[near][np.argmax(qu[near])]
th_s = np.load((_ROOT + "/figs/bfs_steered_300.npy"))
basis = BF.build_basis(mesh, th_s, 60); qcs = BF.flutter(mesh, th_s, basis, grad=False)[0] / 1e6
print(f"best uniform near 300Hz: {bu_ang:.0f} deg, {bu:.3f} MPa | steered {qcs:.3f} MPa | {qcs/bu:.1f}x")

fig = ps.figure(width=ps.FULL, height=2.5)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.30,
                      left=0.085, right=0.915, top=0.90, bottom=0.17)
axa, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# ---------------- (a) the flutter hole ----------------
ps.despine(axa); ps.grid(axa, axis="y")
axa.axvspan(22, 42, color=ps.FILL_WARN, lw=0, zorder=0)
axa.fill_between(deg, qu, color=ps.FILL_MUTE, zorder=1)
axa.plot(deg, qu, color=ps.GREY, lw=1.5, zorder=2)
axa.scatter([bu_ang], [bu], s=26, color=ps.RED, ec="white", lw=0.8, zorder=5)
axa.annotate(r"best uniform at 300\,Hz" + "\n" + rf"${bu*1000:.0f}$ kPa",
             xy=(bu_ang, bu), xytext=(46, 2.4), fontsize=7.6, color=ps.INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=0.9, shrinkA=1, shrinkB=2,
                             connectionstyle="arc3,rad=-0.15"))
axa.text(32, 2.7, "flutter\nhole", ha="center", va="center", color=ps.RED, fontsize=7.8)
axa.set_xlim(0, 90); axa.set_ylim(0, 3.9); axa.set_xticks(range(0, 91, 30))
axa.set_xlabel(r"uniform ply angle (deg)")
axa.set_ylabel(r"flutter pressure $q_{\mathrm{cr}}$ (MPa)")
ps.panel(axa, "a")

# ---------------- (b) the steered tow layout ----------------
ps.despine(axc, keep=("left", "bottom"))
gx = np.linspace(0, BF.A, 60); gy = np.linspace(0, BF.B, 45); GX, GY = np.meshgrid(gx, gy)
class DM: pass
dm = DM(); dm.cx = GX.ravel(); dm.cy = GY.ravel(); dm.N = dm.cx.size
THg = np.rad2deg((CSRBFMapping(dm, (PF.P.rbf.sx, PF.P.rbf.sy), r_s=PF.P.rbf.r_s).Phi @ th_s).reshape(GX.shape))
sp = axc.streamplot(gx, gy, np.cos(np.deg2rad(THg)), np.sin(np.deg2rad(THg)), color=THg,
                    cmap="cividis", density=1.1, linewidth=1.0, arrowsize=0)
cb = fig.colorbar(sp.lines, ax=axc, fraction=0.05, pad=0.03,
                  format=mticker.FuncFormatter(lambda v, p: f"${int(round(v))}$"))
cb.set_label(r"tow angle $\theta$ (deg)", fontsize=8)
cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.6)
axc.annotate(r"flow $M=2$", xy=(0.05, 0.163), xytext=(0.004, 0.163), color=ps.RED, fontsize=7.8,
             va="center", arrowprops=dict(arrowstyle="-|>", color=ps.RED, lw=1.1))
axc.set_xlim(0, BF.A); axc.set_ylim(0, BF.B); axc.set_aspect("equal")
axc.set_xlabel(r"chordwise $x$ (m)"); axc.set_ylabel(r"spanwise $y$ (m)")
ps.panel(axc, "b")

ps.save(fig, (_ROOT + "/figs/plate_flutter"))
