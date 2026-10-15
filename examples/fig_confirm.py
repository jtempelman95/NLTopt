r"""Figure: independent confirmation of the held-frequency design and its consequence.

(a) mid-plate backbones of the full multi-mode plate: cubic SSM (lines) and full-order
    EPMC (markers), UD vs steered, with the evaluator's parabola (dotted);
(b, c) direct time-integration up/down sweeps at ONE drive between the two jump
    thresholds, over the damped SSM forced response: UD folds and jumps, the steered
    panel at the same frequency and mass is single-valued.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
sys.path.insert(0, (_WAVETOPO + ""))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt
import confirm_heldfreq as C
from design_source import heldfreq_design

R = np.load("results/confirm.npz")
H = C.H
designs = [("ud", "UD", ps.GREY, np.zeros(20)), ("opt", "steered", ps.GREEN, heldfreq_design())]
MK = {"ud": "s", "opt": "o"}
HARM = 3                                                       # EPMC harmonics (epmc_check.py)

fig, ax = plt.subplots(1, 3, figsize=(ps.FULL, 2.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
for d, lab, col, th in designs:
    w1 = float(R[f"{d}_ssm_w1"])
    a, o = R[f"{d}_ssm_amp"] / H, R[f"{d}_ssm_om"] / w1
    k = a <= 0.45
    ax[0].plot(o[k], a[k], color=col, lw=1.5, label=f"{lab}: SSM")
    if f"{d}_epmc_amp" in R.files:
        ae, oe = R[f"{d}_epmc_amp"] / H, R[f"{d}_epmc_om"] / float(R[f"{d}_epmc_w1"])
        sel = (ae <= 0.45) & (ae > 0.02)
        idx = np.linspace(0, sel.sum() - 1, 12).astype(int)
        ax[0].plot(oe[sel][idx], ae[sel][idx], MK[d], color=col, ms=3.6, mfc="white", mew=1.1,
                   label=f"{lab}: EPMC")
    aa = np.linspace(0, 0.45, 50)
    ax[0].plot(1 + float(R[f"{d}_ssm_beta"]) * aa ** 2, aa, ":", color=col, lw=0.9)
ax[0].set_xlim(0.99, 1.24); ax[0].set_xticks([1.0, 1.1, 1.2]); ax[0].set_xlabel(r"$\Omega/\omega_1$ (full plate, 36+72 modes)"); ax[0].set_ylabel(r"mid-plate amplitude $a/h$")
from matplotlib.lines import Line2D
h_ = [Line2D([], [], color=ps.GREY, lw=1.5, marker="s", ms=3.6, mfc="white", mew=1.1),
      Line2D([], [], color=ps.GREEN, lw=1.5, marker="o", ms=3.6, mfc="white", mew=1.1),
      Line2D([], [], color=ps.INK, ls=":", lw=0.9)]
l_ = ["UD", "steered", "cubic parabola"]
ax[0].legend(h_, l_, frameon=False, fontsize=6.6, loc="lower right", handlelength=1.8, borderaxespad=0.0,
             bbox_to_anchor=(1.02, 0.0), title="SSM lines, EPMC marks ($H{=}%d$)" % HARM,
             title_fontsize=6.3, alignment="right")
ax[0].set_title(r"(\textbf{a}) free backbones", loc="left", fontsize=8, pad=4)
ps.grid(ax[0]); ps.despine(ax[0])

F = float(R["Fdrive"])
for j, (d, lab, col, th) in enumerate(designs):
    b = ax[1 + j]
    ssm, ref, M, K, elem = C.ssm_backbone(th, damped=True)
    pl, _, _, _, _, Sinv = C.build(th)
    cfull = np.zeros(pl.ndof)
    cfull[:pl.nw] = [np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2) for (m, n) in pl.w_modes]
    fr = ssm.forced_response(F * (Sinv.T @ cfull), dof=ref, n=800)
    ok = np.isfinite(fr["omega_lo"]); w1 = ssm.omega0
    b.plot(fr["omega_lo"][ok] / w1, fr["amplitude"][ok] / H, color=col, lw=1.4)
    b.plot(fr["omega_hi"][ok] / w1, fr["amplitude"][ok] / H, color=col, lw=1.4,
           label="SSM, full plate")
    ou, au = R[f"{d}_sweep_up_o"], R[f"{d}_sweep_up_a"]
    od, ad = R[f"{d}_sweep_down_o"], R[f"{d}_sweep_down_a"]
    b.plot(ou, au, "o", color=ps.INK, ms=2.8, label="time int.: up")
    b.plot(od, ad, "s", color=ps.RED, ms=3.0, mfc="none", mew=0.9, label="time int.: down")
    b.set_xlim(0.962, 1.128); b.set_ylim(0, 0.50); b.set_xticks([1.0, 1.05, 1.10])
    b.set_xlabel(r"drive frequency $\Omega/\omega_1$")
    b.set_ylabel(r"mid-plate amplitude $a/h$")
    b.legend(frameon=False, fontsize=6.6, loc="upper right", handlelength=1.4, borderaxespad=0.1)
    b.set_title(rf"(\textbf{{{'bc'[j]}}}) {lab}: " + ("folds and jumps" if d == "ud" else "single-valued"),
                loc="left", fontsize=8, pad=4)
    b.text(1.124, 0.335, "time integration on a\n6-mode modal model,\n300 periods per step", fontsize=6.4,
           color=ps.GREY, va="top", ha="right")
    ps.grid(b); ps.despine(b)
fig.tight_layout(w_pad=1.2)
ps.save(fig, (_ROOT + "/figs/fig_confirm"))
