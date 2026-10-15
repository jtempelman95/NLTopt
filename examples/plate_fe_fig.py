r"""FE-validation figure: assumed-modes vs FE backbone (Omega/omega1 vs a/h) for a
homogeneous and a strongly steered plate, plus Ritz convergence toward FE."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys; sys.path.insert(0, (_WAVETOPO + "")); sys.path.insert(0, (_ROOT + "/examples"))
import numpy as np
import paperstyle as ps
import matplotlib.pyplot as plt
import plate_fe_backbone as BB

FE_C, UD, STE, AMB, GRID = ps.INK, ps.GREY, ps.GREEN, ps.STEEL, ps.GRID
A, B = BB.A, BB.B
ud = lambda x, y: 0.0
steer = lambda x, y: np.deg2rad(20 + 45*np.sin(np.pi*x/A)*np.sin(np.pi*y/B))

import os
CACHE = (_ROOT + "/results/plate_fe_fig_cache.npz")
NM = [3, 6, 8, 12, 18, 24, 32]
if os.path.exists(CACHE):
    Z = np.load(CACHE); fe_ud, fe_st, mod_ud, mod_st = Z["fe_ud"], Z["fe_st"], Z["mod_ud"], Z["mod_st"]
else:
    print("computing FE (UD, steered) + modal convergence...")
    fe_ud = np.array(BB.fe_backbone(ud)[:4]); fe_st = np.array(BB.fe_backbone(steer)[:4])   # (f1, gamma, Kb, beta)
    mod_ud = np.array([BB.modal_backbone(ud, nm)[:4] for nm in NM])
    mod_st = np.array([BB.modal_backbone(steer, nm)[:4] for nm in NM])
    np.savez(CACHE, fe_ud=fe_ud, fe_st=fe_st, mod_ud=mod_ud, mod_st=mod_st)

ah = np.linspace(0, 0.33, 100)
def bb(beta): return 1 + beta*ah**2

fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.55))
# (a) backbone curves
axa = ax[0]
axa.plot(bb(fe_ud[3]), ah, "-", color=FE_C, lw=2.4, label="FE")
axa.plot(bb(mod_ud[2][3]), ah, "--", color=UD, lw=1.4, label="UD, 8 modes")
axa.plot(bb(fe_st[3]), ah, "-", color=FE_C, lw=2.4)
axa.plot(bb(mod_st[2][3]), ah, ":", color=AMB, lw=1.6, label="steered, 8 modes")
axa.plot(bb(mod_st[5][3]), ah, "--", color=STE, lw=1.4, label="steered, 24 modes")
k1, k2 = np.searchsorted(ah, 0.15), np.searchsorted(ah, 0.26)
axa.text(bb(fe_ud[3])[k1] + 0.0010, ah[k1] - 0.01, "UD", fontsize=7.5, color=UD, ha="left", va="top")
axa.text(bb(fe_st[3])[k1] - 0.0012, ah[k1] + 0.03, "steered", fontsize=7.5, color=STE, ha="right", va="bottom")
axa.set_ylabel(r"amplitude $a/h$"); axa.set_xlabel(r"backbone $\Omega/\omega_1$")
axa.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.0, borderaxespad=0.2)
axa.set_ylim(0, 0.33); axa.set_xlim(1.0, 1.035)
ps.panel(axa, "a"); ps.grid(axa); ps.despine(axa)

# (b) convergence toward FE (steered)
axb = ax[1]
det_fe = fe_st[3]; det_mod = mod_st[:, 3]
f_fe = fe_st[0]; f_mod = mod_st[:, 0]
ed, ef = 100*(det_mod-det_fe)/det_fe, 100*(f_mod-f_fe)/f_fe
axb.fill_between([0, 34], -2, 2, color=ps.FILL_GOOD, zorder=0, label=r"$\pm2\%$ band")
axb.axhline(0, color=ps.GREY, lw=0.7)
axb.plot(NM, ed, "-o", color=STE, lw=1.4, ms=4, label="backbone detuning $\\beta$")
axb.plot(NM, ef, "-s", color=AMB, lw=1.4, ms=4, mfc="white", mew=1.1, label="frequency $\\omega_1$ (Ritz upper bound)")
axb.set_xlabel("number of assumed modes"); axb.set_ylabel(r"error vs.\ FE (\%)")
axb.set_xlim(0, 34); axb.set_xticks([3, 8, 12, 18, 24, 32])
lo, hi = min(ed.min(), ef.min()), max(ed.max(), ef.max())
axb.set_ylim(np.floor(lo) - 1, np.ceil(hi) + 1)
axb.legend(loc="lower right", frameon=False, fontsize=7)
ps.panel(axb, "b"); ps.grid(axb); ps.despine(axb)
fig.tight_layout(w_pad=1.5)
ps.save(fig, (_ROOT + "/figs/plate_fe_validation"))
print("modes", NM); print("det err %", np.round(ed, 2)); print("freq err %", np.round(ef, 2))
