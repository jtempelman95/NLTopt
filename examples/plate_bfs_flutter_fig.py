r"""Corrected panel-flutter figure (conforming BFS FE), styled:
 (a) the bend-twist 'flutter hole' a uniform layup falls into at 300 Hz
 (b) the two layups compared -- horizontal pill bars, steered vs best uniform
 (c) the steered tow layout."""
import sys; sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wavetopo.cfrp import CSRBFMapping
import plate_bfs_flutter as BF
import plate_flutter as PF

INK, MUT, FAINT = "#1f2933", "#7b8794", "#cbd2d9"
UNI, UNI_SOFT = "#8695a6", "#dce2e8"
STE, STE_SOFT = "#137a54", "#d5ede2"
HOT, HOT_SOFT = "#cf5252", "#f7e4e4"
GRID, PANEL = "#edf0f3", "#ffffff"
rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": FAINT, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 1.1,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "figure.facecolor": "white",
})


def panel_title(ax, letter, text, color):
    ax.set_title(f"       {text}", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=12)
    ax.text(0.028, 1.065, letter, transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.30", fc=color, ec="none"))


def clean(ax, spines=("left", "bottom")):
    for k in ("top", "right", "left", "bottom"):
        ax.spines[k].set_visible(k in spines)
    for k in spines:
        ax.spines[k].set_color(FAINT)


mesh = BF.Mesh(16, 12)
th_s = np.load("bfs_steered_300.npy")
front = BF.uniform_front(mesh, range(0, 91, 2))
deg = np.array([p[0] for p in front]); f1u = np.array([p[1] for p in front]); qu = np.array([p[2] for p in front])/1e6
basis = BF.build_basis(mesh, th_s, 60); qcs = BF.flutter(mesh, th_s, basis, grad=False)[0]/1e6
near = np.abs(f1u - 300.0) < 15
bu = qu[near].max(); bu_ang = deg[near][np.argmax(qu[near])]
ratio = qcs/bu

fig = plt.figure(figsize=(15.2, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.02], wspace=0.34,
                      left=0.055, right=0.965, top=0.80, bottom=0.16)
axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

# ================= (a) the flutter hole =================
clean(axa)
axa.set_facecolor(PANEL)
axa.grid(True, axis="y", color=GRID, lw=1.1); axa.set_axisbelow(True)
axa.axvspan(22, 42, color=HOT_SOFT, alpha=0.6, lw=0)
axa.fill_between(deg, qu, color=UNI_SOFT, alpha=0.7, zorder=1)
axa.plot(deg, qu, "-", color=UNI, lw=2.8, zorder=2, solid_capstyle="round")
axa.scatter([bu_ang], [bu], s=90, color=HOT, zorder=5, ec="white", lw=1.8)
axa.annotate(f"best uniform near\n300 Hz  =  {bu*1000:.0f} kPa", xy=(bu_ang, bu),
             xytext=(41, 2.3), fontsize=10.5, color=INK, fontweight="bold", ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=HOT, lw=1.4),
             arrowprops=dict(arrowstyle="-|>", color=HOT, lw=1.8, connectionstyle="arc3,rad=-0.15"))
axa.text(32, 0.12, "bend-twist flutter hole", ha="center", va="bottom",
         fontsize=10.5, color=HOT, fontweight="bold")
axa.set_xlabel("uniform ply angle   [deg]"); axa.set_ylabel("flutter pressure  $q_{cr}$   [MPa]")
axa.set_xlim(0, 90); axa.set_ylim(0, 3.9); axa.set_xticks(range(0, 91, 30))
panel_title(axa, "a", "coupling craters a uniform layup", HOT)

# ================= (b) the comparison =================
clean(axb, spines=())
axb.set_facecolor(PANEL)
axb.axvline(0, color=FAINT, lw=1.2, zorder=1)
for name, y, val, col, lab, labcol in [("uniform", 0.0, bu, UNI, "best\nuniform", MUT),
                                        ("steered", 1.0, qcs, STE, "steered\nfield", INK)]:
    axb.plot([0, val], [y, y], lw=24, color=col, solid_capstyle="round", zorder=3)
    axb.text(val + 0.30, y, f"{val:.2f} MPa", va="center", ha="left",
             fontsize=13, fontweight="bold", color=col)
    axb.text(-0.16, y, lab, va="center", ha="right", fontsize=11.5, color=labcol,
             fontweight="bold" if name == "steered" else "normal")
axb.text(2.5, 0.5, f"{ratio:.0f}$\\times$", ha="center", va="center", fontsize=42,
         fontweight="bold", color=STE)
axb.text(2.5, 0.17, "more flutter margin", ha="center", va="center", fontsize=10, color=MUT)
axb.set_xlim(-0.05, 4.95); axb.set_ylim(-0.7, 1.7)
axb.set_xticks([]); axb.set_yticks([])
axb.text(0.5, -0.42, "same 300 Hz fundamental  ·  same mass", transform=axb.transAxes,
         ha="center", fontsize=10.5, color=MUT, style="italic")
panel_title(axb, "b", "steering recovers the margin", STE)

# ================= (c) steered tow layout =================
clean(axc, spines=("left", "bottom"))
gx = np.linspace(0, BF.A, 62); gy = np.linspace(0, BF.B, 46); GX, GY = np.meshgrid(gx, gy)
class DM: pass
dm = DM(); dm.cx = GX.ravel(); dm.cy = GY.ravel(); dm.N = dm.cx.size
THg = np.rad2deg((CSRBFMapping(dm, (PF.P.rbf.sx, PF.P.rbf.sy), r_s=PF.P.rbf.r_s).Phi @ th_s).reshape(GX.shape))
sp = axc.streamplot(gx, gy, np.cos(np.deg2rad(THg)), np.sin(np.deg2rad(THg)), color=THg,
                    cmap="viridis", density=1.15, linewidth=1.7, arrowsize=0)
cb = fig.colorbar(sp.lines, ax=axc, fraction=0.045, pad=0.03)
cb.set_label(r"tow angle  $\theta$  [deg]", fontsize=10); cb.ax.tick_params(labelsize=9); cb.outline.set_edgecolor(FAINT)
axc.annotate("flow  $M{=}2$", xy=(0.062, 0.163), xytext=(0.004, 0.163), color="#b3760a", fontsize=10.5,
             va="center", fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="#b3760a", lw=2.2))
axc.set_xlim(0, BF.A); axc.set_ylim(0, BF.B); axc.set_aspect("equal")
axc.set_xlabel("chordwise  $x$   [m]"); axc.set_ylabel("spanwise  $y$   [m]")
panel_title(axc, "c", "the steered tow layout", "#b3760a")

fig.text(0.5, 0.95, "Same 300 Hz fundamental and mass — the steered tow field flutters at "
         f"{ratio:.0f}$\\times$ the pressure of the best uniform layup",
         ha="center", fontsize=13, fontweight="bold", color=INK)
fig.savefig("/home/jrt/NLTopt/figs/plate_flutter.png", dpi=160, facecolor="white")
print(f"wrote figs/plate_flutter.png; best-uniform {bu*1000:.0f} kPa, steered {qcs*1000:.0f} kPa -> {ratio:.1f}x")
