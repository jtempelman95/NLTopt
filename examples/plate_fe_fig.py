r"""FE-validation figure: assumed-modes vs FE backbone (Omega/omega1 vs a/h) for a
homogeneous and a strongly steered plate, plus Ritz convergence toward FE."""
import sys; sys.path.insert(0, "/home/jrt/wavetopo"); sys.path.insert(0, "/home/jrt/NLTopt/examples")
import numpy as np
import paperstyle as ps
import matplotlib.pyplot as plt
import plate_fe_backbone as BB

FE_C, UD, STE, AMB, GRID = ps.INK, ps.GREY, ps.GREEN, ps.STEEL, ps.GRID
A, B = BB.A, BB.B
ud = lambda x, y: 0.0
steer = lambda x, y: np.deg2rad(20 + 45*np.sin(np.pi*x/A)*np.sin(np.pi*y/B))

print("computing FE (UD, steered) + modal convergence...")
fe_ud = BB.fe_backbone(ud);      fe_st = BB.fe_backbone(steer)      # (f1, gamma, Kb, beta)
NM = [3, 6, 8, 12, 18, 24, 32]
mod_ud = [BB.modal_backbone(ud, nm) for nm in NM]
mod_st = [BB.modal_backbone(steer, nm) for nm in NM]

ah = np.linspace(0, 0.5, 100)
def bb(beta): return 1 + beta*ah**2

fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.9))
for a in ax: a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
for a in ax:
    for s in a.spines.values(): s.set_color("#9aa0a6")

# (a) backbone curves
axa = ax[0]
axa.plot(ah, bb(fe_ud[3]), "-", color=FE_C, lw=2.6, label="FE (ground truth)")
axa.plot(ah, bb(mod_ud[2][3]), "--", color=UD, lw=2.0, label="assumed-modes (8)")
axa.plot(ah, bb(fe_st[3]), "-", color=FE_C, lw=2.6)
axa.plot(ah, bb(mod_st[2][3]), ":", color=AMB, lw=2.2, label="steered, modes = 8")
axa.plot(ah, bb(mod_st[5][3]), "--", color=STE, lw=2.0, label="steered, modes = 24")
axa.annotate("homogeneous UD\n(FE = modal: validates FE model)", xy=(0.42, bb(fe_ud[3])[84]),
             xytext=(0.10, 1.028), fontsize=8.6, color=UD,
             arrowprops=dict(arrowstyle="-|>", color=UD, lw=1.2))
axa.annotate("strongly steered\n(modal $\\to$ FE as modes added)", xy=(0.46, bb(fe_st[3])[92]),
             xytext=(0.045, 1.0125), fontsize=8.6, color=STE,
             arrowprops=dict(arrowstyle="-|>", color=STE, lw=1.2))
axa.set_xlabel("amplitude  $a/h$"); axa.set_ylabel(r"backbone  $\Omega/\omega_1$")
axa.set_title("(a)  backbone: assumed-modes vs FE", loc="left", fontweight="bold")
axa.legend(loc="upper left", frameon=False); axa.set_xlim(0, 0.5); axa.set_ylim(1.0, 1.035)

# (b) convergence toward FE (steered)
axb = ax[1]
det_fe = 0.16*fe_st[3]*100
det_mod = np.array([0.16*m[3]*100 for m in mod_st])
f_fe = fe_st[0]; f_mod = np.array([m[0] for m in mod_st])
axb.axhline(0, color="#333", lw=0.9)
axb.plot(NM, 100*(det_mod-det_fe)/det_fe, "-o", color=STE, lw=2.2, ms=6, label="backbone detuning error")
axb.plot(NM, 100*(f_mod-f_fe)/f_fe, "-s", color=AMB, lw=2.2, ms=6, mfc="white", label="frequency $f_1$ error")
axb.fill_between([2, 34], -2, 2, color=STE, alpha=0.09)
axb.text(3.2, 1.15, "FE-accurate band $\\pm2\\%$", color=STE, fontsize=8.2, va="center")
axb.set_xlabel("number of assumed modes"); axb.set_ylabel(r"error vs FE  (\%)")
axb.set_title("(b)  Ritz convergence to FE (strongly steered)", loc="left")
axb.legend(loc="lower right", frameon=False); axb.set_xlim(2, 34); axb.set_ylim(-9, 9)
axb.annotate(rf"detuning: {100*(det_mod[5]-det_fe)/det_fe:+.0f}\% @24 modes" + "\n"
             + rf"freq: {100*(f_mod[5]-f_fe)/f_fe:+.0f}\% (Ritz upper bound)",
             xy=(24, 100*(f_mod[5]-f_fe)/f_fe), xytext=(9, 6.6), fontsize=8, color=ps.INK,
             arrowprops=dict(arrowstyle="-|>", color=AMB, lw=1.0))

fig.tight_layout()
ps.save(fig, "/home/jrt/NLTopt/figs/plate_fe_validation")
print(f"UD:      FE det {0.16*fe_ud[3]*100:.2f}%  modal-8 {0.16*mod_ud[2][3]*100:.2f}%")
print(f"steered: FE det {det_fe:.2f}%  modal-8 {det_mod[2]:.2f}%  modal-24 {det_mod[5]:.2f}%")
