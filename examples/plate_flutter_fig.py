r"""Publication figure for the supersonic panel-flutter study (writes figs/plate_flutter.png):
 (a) steered tow layout   (b) flutter margin q_cr vs required f1
 (c) frequency-coalescence flutter diagram (baseline vs steered)
 (d) s-plane root locus of the critical mode (the Hopf crossing pushed out)

Run examples/plate_flutter.py first to produce figs/flutter_steered_30.npy.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_flutter_fig.py
"""
import os
import sys
sys.path.insert(0, "/home/jrt/wavetopo")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import eig
from wavetopo.cfrp import CSRBFMapping

import plate_flutter as PF

rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
                 "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
                 "axes.linewidth": 0.9, "font.family": "DejaVu Sans"})
UNI, STE, AMB, GRID, BAD = "#5b6470", "#12813f", "#b3760a", "#dcdfda", "#c44"
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")


def track(pl, G, qs, nmodes=4):
    """Continuation-track the lowest `nmodes` eigenvalue branches (Im>0) vs q."""
    IM = np.full((len(qs), nmodes), np.nan); RE = np.full((len(qs), nmodes), np.nan)
    prev = None
    for i, q in enumerate(qs):
        Amat, Bmat, _ = PF._pencil(pl, G, q)
        w = eig(Amat, Bmat, right=False)
        w = w[w.imag > 1.0]; w = w[np.argsort(w.imag)]
        if prev is None:
            cur = list(w[:nmodes])
        else:
            avail = list(w); cur = []
            for p in prev:
                j = int(np.argmin([abs(p - a) for a in avail])); cur.append(avail.pop(j))
        prev = cur
        for k, c in enumerate(cur):
            IM[i, k] = c.imag / (2 * np.pi); RE[i, k] = c.real
    return IM, RE


def main():
    th = np.load(os.path.join(FIGS, "flutter_steered_30.npy"))
    pls = PF.build(th); Gs = PF.aero_G(pls); qcs = PF.qcr(pls, Gs); f1s = PF.f1_of(pls)
    plu = PF.build(np.full(PF.P.M, np.deg2rad(30))); Gu = PF.aero_G(plu); qcu = PF.qcr(plu, Gu)
    qs = np.linspace(0, 1.12 * qcs, 90)
    IMs, REs = track(pls, Gs, qs); IMu, REu = track(plu, Gu, qs)
    nb = np.searchsorted(qs, 1.06 * qcu)

    fig, ax = plt.subplots(2, 2, figsize=(11.4, 8.6))
    for a in ax.ravel():
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_color("#9aa0a6")

    # (a) tow layout
    axa = ax[0, 0]
    xs = np.linspace(0, PF.A, 64); ys = np.linspace(0, PF.B, 48); XX, YY = np.meshgrid(xs, ys)
    class DM: pass
    dm = DM(); dm.cx = XX.ravel(); dm.cy = YY.ravel(); dm.N = dm.cx.size
    Phi = CSRBFMapping(dm, (PF.P.rbf.sx, PF.P.rbf.sy), r_s=PF.P.rbf.r_s).Phi
    TH = np.rad2deg((Phi @ th).reshape(XX.shape))
    sp = axa.streamplot(xs, ys, np.cos(np.deg2rad(TH)), np.sin(np.deg2rad(TH)),
                        color=TH, cmap="viridis", density=1.25, linewidth=1.5, arrowsize=0)
    cb = fig.colorbar(sp.lines, ax=axa, fraction=0.045, pad=0.02)
    cb.set_label(r"tow angle  $\theta$  [deg]", fontsize=9); cb.ax.tick_params(labelsize=8)
    axa.annotate(r"flow  $M{=}2$", xy=(0.075, 0.164), xytext=(0.004, 0.164), fontsize=9.5,
                 color=AMB, va="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=AMB, lw=2.4))
    axa.set_xlim(0, PF.A); axa.set_ylim(0, PF.B); axa.set_aspect("equal")
    axa.set_xlabel(r"chordwise  $x$  [m]"); axa.set_ylabel(r"spanwise  $y$  [m]")
    axa.set_title("(a)  optimized steered tow layout", loc="left", fontweight="bold")

    # (b) payoff
    axb = ax[0, 1]
    front = sorted(PF.uniform_pareto(range(0, 57, 2)), key=lambda p: p[1])
    ff = np.array([p[1] for p in front]); fq = np.array([p[2] for p in front])
    axb.plot(ff, fq / 1e3, "-", color=UNI, lw=2.4, marker="o", ms=3.5, label="best uniform layup")
    axb.plot(f1s, qcs / 1e3, "*", color=STE, ms=21, mec="white", mew=1.1, zorder=6, label=r"steered (held $f_1$)")
    axb.plot(f1s, qcu / 1e3, "o", color=UNI, ms=8, mfc="white", mec=UNI, mew=1.8, zorder=5)
    axb.annotate("", xy=(f1s, qcs / 1e3), xytext=(f1s, qcu / 1e3),
                 arrowprops=dict(arrowstyle="-|>", color=STE, lw=2.8))
    axb.text(f1s + 4, 0.52 * (qcs + qcu) / 1e3, rf"$\mathbf{{{qcs/qcu:.2f}\times}}$",
             color=STE, fontsize=15, fontweight="bold", va="center")
    axb.text(f1s + 4, 0.52 * (qcs + qcu) / 1e3 - 260, r"flutter margin" + "\n" + r"at same $f_1$, mass",
             color=STE, fontsize=8.5, va="top")
    axb.set_xlabel(r"required operating frequency  $f_1$  [Hz]")
    axb.set_ylabel(r"flutter dynamic pressure  $q_{cr}$  [kPa]")
    axb.set_title("(b)  more flutter margin at the same frequency", loc="left", fontweight="bold")
    axb.legend(loc="lower left", frameon=False); axb.set_xlim(250, 360); axb.set_ylim(750, 4400)

    # (c) frequency coalescence
    axc = ax[1, 0]
    for k in range(3):
        axc.plot(qs[:nb] / 1e3, IMu[:nb, k], color=UNI, lw=1.9, alpha=0.9)
        axc.plot(qs / 1e3, IMs[:, k], color=STE, lw=2.1)
    axc.plot([], [], color=UNI, lw=2.2, label="uniform layup")
    axc.plot([], [], color=STE, lw=2.2, label="steered layup")
    for qc, c, lab in [(qcu, UNI, "2121"), (qcs, STE, "4057")]:
        axc.axvline(qc / 1e3, color=c, ls="--", lw=1.5)
        axc.text(qc / 1e3 - 0.05, 320, rf"$q_{{cr}}$" + f"\n{lab} kPa", color=c, fontsize=8.5,
                 ha="right", va="bottom", fontweight="bold")
    axc.annotate("modes\ncoalesce", xy=(qcs / 1e3 - 0.12, 815), xytext=(2450, 470),
                 fontsize=9.5, color=STE, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=STE, lw=1.7))
    axc.set_xlabel(r"dynamic pressure  $q$  [kPa]")
    axc.set_ylabel(r"modal frequency  Im$(s)/2\pi$  [Hz]")
    axc.set_title("(c)  steering delays the modal coalescence", loc="left", fontweight="bold")
    axc.set_xlim(0, qs.max() / 1e3); axc.set_ylim(280, 1360); axc.legend(loc="upper left", frameon=False)

    # (d) root locus
    axd = ax[1, 1]
    kc_s = int(np.argmax(REs[-1])); kc_u = int(np.argmax(REu[np.searchsorted(qs, 1.05 * qcu) - 1]))
    axd.axvspan(0, 60, color=BAD, alpha=0.07); axd.axvline(0, color="#333", lw=1.1)
    axd.text(3, 1285, "unstable", color=BAD, fontsize=9, fontweight="bold")
    axd.plot(REu[:nb, kc_u], IMu[:nb, kc_u], "-", color=UNI, lw=2.3, label="uniform")
    axd.plot(REs[:, kc_s], IMs[:, kc_s], "-", color=STE, lw=2.3, label="steered")
    iu = np.argmin(np.abs(REu[:nb, kc_u])); iv = np.argmin(np.abs(REs[:, kc_s]))
    axd.plot(0, IMu[iu, kc_u], "o", color=UNI, ms=11, mec="white", mew=1.4, zorder=6)
    axd.plot(0, IMs[iv, kc_s], "*", color=STE, ms=18, mec="white", mew=1.0, zorder=6)
    axd.annotate(r"Hopf" + "\n" + r"$q_{cr}$=2121", xy=(0, IMu[iu, kc_u]),
                 xytext=(-33, IMu[iu, kc_u] + 70), color=UNI, fontsize=8.6, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=UNI, lw=1.4))
    axd.annotate(r"Hopf" + "\n" + r"$q_{cr}$=4057", xy=(0, IMs[iv, kc_s]),
                 xytext=(11, IMs[iv, kc_s] - 30), color=STE, fontsize=8.6, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=STE, lw=1.4))
    axd.plot(REu[0, kc_u], IMu[0, kc_u], "s", color=UNI, ms=6)
    axd.text(REu[0, kc_u] - 1, IMu[0, kc_u] - 45, r"$q$=0", color=UNI, fontsize=8, ha="center")
    axd.legend(loc="lower left", frameon=False); axd.set_xlim(-40, 40); axd.set_ylim(600, 1330)
    axd.set_xlabel(r"damping  Re$(s)$  [1/s]"); axd.set_ylabel(r"frequency  Im$(s)/2\pi$  [Hz]")
    axd.set_title("(d)  root locus: the critical mode crosses later", loc="left", fontweight="bold")

    fig.suptitle("Fiber steering pushes the supersonic panel-flutter Hopf out — "
                 rf"{qcs/qcu:.2f}$\times$ margin at held $f_1={f1s:.0f}$ Hz and mass",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(FIGS, "plate_flutter.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  (steered {qcs/1e3:.0f} vs uniform {qcu/1e3:.0f} kPa = {qcs/qcu:.2f}x)")


if __name__ == "__main__":
    main()
