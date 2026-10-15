r"""Mechanism figure for the panel-flutter section (LaTeX-typeset, vector PDF).
How fiber steering reshapes the aeroelastic response of the conforming BFS model,
cause -> onset -> aftermath:
 (a) modal coalescence (V-f): the steered layup's bending & torsion modes merge far out in q,
     while the uniform 'flutter-hole' layup destabilises early  -- REAL BFS pencil eigenvalues.
 (b) growth rate (V-g): max Re(s) crosses zero at q_cr; steering slides the onset ~9x -- REAL data.
 (c) post-flutter bifurcation: supercritical Hopf (Re gamma<0) -> a bounded limit cycle; the
     normal-form branch a ~ sqrt(q-q_cr), with the recovered-margin band between the two onsets.
Run: python plate_flutter_mechanism_fig.py"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
import numpy as np
from scipy.linalg import eigh, eig
import paperstyle as ps
import plate_bfs_flutter as BF

# ===================== real aeroelastic sweeps (conforming BFS pencil) =====================
mesh = BF.Mesh(16, 12)
BETA, C0 = BF.BETA, BF.C0

front = BF.uniform_front(mesh, range(0, 91, 2))
deg = np.array([p[0] for p in front]); f1u = np.array([p[1] for p in front]); qu = np.array([p[2] for p in front])
near = np.abs(f1u - 300.0) < 15
bu = qu[near].max(); bu_ang = deg[near][np.argmax(qu[near])]

th_s = np.load((_ROOT + "/figs/bfs_steered_300.npy"))
basis_s = BF.build_basis(mesh, th_s, 60)
qcs, f1s = BF.flutter(mesh, th_s, basis_s, grad=False)
ratio = qcs / bu
print(f"uniform {bu_ang:.0f}deg q_cr={bu/1e6:.3f} MPa | steered q_cr={qcs/1e6:.3f} MPa | {ratio:.1f}x")

te_u = mesh.theta_e(np.full(mesh.PhiD.shape[1], np.deg2rad(bu_ang)))
w2u, Phiu = eigh(mesh.K(te_u), mesh.M); Phiu = Phiu[:, :40]
Kru = np.diag(w2u[:40]); Gru = Phiu.T @ mesh.G @ Phiu
Phis = basis_s["Phi"]; nrs = Phis.shape[1]
Krs = Phis.T @ mesh.K(mesh.theta_e(th_s)) @ Phis; Grs = basis_s["Gr"]


def poles(Kr, Gr, q):
    nr = Kr.shape[0]; Cr = C0 * np.eye(nr); Z = np.zeros((nr, nr)); I = np.eye(nr)
    return eig(np.block([[Z, I], [-(Kr + BETA * q * Gr), -Cr]]), right=False)


def sweep(Kr, Gr, qgrid, nf=2):
    freqs = np.full((len(qgrid), nf), np.nan); grow = np.empty(len(qgrid))
    for i, q in enumerate(qgrid):
        s = poles(Kr, Gr, q); grow[i] = s.real.max()
        w = np.sort(np.abs(s.imag[s.imag > 1e-6])) / (2 * np.pi)
        freqs[i, :min(nf, w.size)] = w[:nf]
    return freqs, grow


qU = np.linspace(0, 0.90e6, 130)
qS = np.linspace(0, 3.62e6, 220)
FU, GU = sweep(Kru, Gru, qU); FS, GS = sweep(Krs, Grs, qS)
qcrU, qcrS = bu / 1e6, qcs / 1e6
qUm, qSm = qU / 1e6, qS / 1e6

# ===================== figure =====================
fig = ps.figure(width=ps.FULL, height=2.35)
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.075, right=0.985, top=0.90, bottom=0.20)
axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
XL = r"dynamic pressure $q$ (MPa)"

# ---------------- (a) coalescence (V-f) ----------------
ps.despine(axa); ps.grid(axa)
axa.axvspan(0, qcrU, color=ps.FILL_MUTE, lw=0, zorder=0)
axa.plot(qSm, FS[:, 0], color=ps.GREEN, lw=1.5, label="steered")
axa.plot(qSm, FS[:, 1], color=ps.GREEN, lw=1.5)
mU = qUm <= qcrU + 1e-6
axa.plot(qUm[mU], FU[mU, 0], color=ps.GREY, lw=1.5, label="uniform")
axa.axvline(qcrU, color=ps.GREY, lw=0.9, ls=(0, (2.5, 2.5)))
axa.axvline(qcrS, color=ps.GREEN, lw=0.9, ls=(0, (2.5, 2.5)))
fco = np.nanmean(FS[np.argmin(np.abs(qSm - qcrS))])
axa.scatter([qcrS], [fco], s=22, color=ps.GREEN, ec="white", lw=0.8, zorder=6)
axa.scatter([qcrU], [FU[np.argmin(np.abs(qUm - qcrU)), 0]], s=18, color=ps.GREY, ec="white", lw=0.8, zorder=6)
axa.text(0.62, 286, "bending", color=ps.GREEN, va="top", fontsize=7.5)
axa.text(1.15, 612, "torsion", color=ps.GREEN, va="bottom", fontsize=7.5)
axa.annotate("coalescence", xy=(qcrS, fco), xytext=(2.15, 545),
             color=ps.GREEN, ha="center", va="center", fontsize=7.8,
             arrowprops=dict(arrowstyle="-|>", color=ps.GREEN, lw=0.8, shrinkA=1, shrinkB=2,
                             connectionstyle="arc3,rad=0.2"))
axa.text(0.10, 660, "uniform\nflutter hole", color=ps.GREY, ha="left", va="center", fontsize=7.2)
axa.set_xlim(0, 3.62); axa.set_ylim(250, 800)
axa.set_xlabel(XL); axa.set_ylabel("modal frequency (Hz)")
axa.legend(loc="lower right", frameon=False, handlelength=1.3, borderaxespad=0.3)
ps.panel(axa, "a")

# ---------------- (b) growth rate (V-g) ----------------
ps.despine(axb); ps.grid(axb)
axb.axhspan(0, 45, color=ps.FILL_WARN, lw=0, zorder=0)
axb.axhline(0, color=ps.INK, lw=0.9, zorder=2)
mUg = qUm <= 0.62
axb.plot(qUm[mUg], GU[mUg], color=ps.GREY, lw=1.5)
mSg = qSm <= qcrS + 0.05
axb.plot(qSm[mSg], GS[mSg], color=ps.GREEN, lw=1.5)
axb.scatter([qcrU, qcrS], [0, 0], s=[18, 22], color=[ps.GREY, ps.GREEN], ec="white", lw=0.8, zorder=6)
axb.text(1.9, 33, "unstable", ha="center", va="center", color=ps.RED, fontsize=7.8)
axb.text(1.9, -13, "stable", ha="center", va="center", color=ps.GREY, fontsize=7.8)
axb.annotate("", xy=(qcrS, 7), xytext=(qcrU, 7),
             arrowprops=dict(arrowstyle="<|-|>", color=ps.INK, lw=1.0, shrinkA=0, shrinkB=0))
axb.text(0.5 * (qcrU + qcrS), 11, r"$9\times$ margin", ha="center", va="bottom", fontsize=8.2)
axb.text(qcrU, -21.5, "0.37", ha="center", va="top", color=ps.GREY, fontsize=7.2)
axb.text(qcrS, -21.5, "3.4 MPa", ha="center", va="top", color=ps.GREEN, fontsize=7.2)
axb.set_xlim(0, 3.62); axb.set_ylim(-24, 45)
axb.set_xlabel(XL); axb.set_ylabel(r"growth rate $\mathrm{Re}\,s$ (s$^{-1}$)")
ps.panel(axb, "b")

# ---------------- (c) post-flutter bifurcation (supercritical normal form) ----------------
ps.despine(axc); ps.grid(axc)
axc.axvspan(qcrU, qcrS, color=ps.FILL_GOOD, lw=0, zorder=0)
qg = np.linspace(0, 3.62, 400)
sc = 1.0 / np.sqrt(3.62 - qcrU)
aU = np.sqrt(np.clip(qg - qcrU, 0, None)) * sc
aS = np.sqrt(np.clip(qg - qcrS, 0, None)) * sc
axc.plot([0, qcrU], [0, 0], color=ps.GREY, lw=1.5)
axc.plot(qg[qg >= qcrU], aU[qg >= qcrU], color=ps.GREY, lw=1.5, label="uniform")
axc.plot(qg[qg >= qcrS], aS[qg >= qcrS], color=ps.GREEN, lw=1.5, label="steered")
axc.scatter([qcrU, qcrS], [0, 0], s=[18, 22], color=[ps.GREY, ps.GREEN], ec="white", lw=0.8, zorder=6)
axc.text(0.5 * (qcrU + qcrS), 0.06, "steered\nquiet", ha="center", va="bottom", color=ps.GREEN, fontsize=7.2)
axc.annotate(r"supercritical: $\mathrm{Re}\,\gamma<0$", xy=(2.6, aU[np.argmin(np.abs(qg - 2.6))]),
             xytext=(0.9, 0.9), fontsize=7.6, color=ps.INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="-|>", color=ps.GREY, lw=0.8, shrinkA=1, shrinkB=2,
                             connectionstyle="arc3,rad=-0.2"))
axc.set_xlim(0, 3.62); axc.set_ylim(-0.03, 1.02); axc.set_yticks([])
axc.set_xlabel(XL); axc.set_ylabel("LCO amplitude (norm. form)")
ps.panel(axc, "c")

ps.save(fig, (_ROOT + "/figs/plate_flutter_mechanism"))
