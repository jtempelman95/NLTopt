r"""Is the held-frequency comparison airtight under refinement?  (Reviewer R1, M2.)

The design holds omega_1 = omega_UD EXACTLY in the design model (64 transverse + 72
in-plane modes).  On a finer transverse basis and on the independent FE model the steered
frequency comes out up to ~1.4 % BELOW the UD value, so the published ratio compares the
steered plate at a slightly lower frequency.  Two corrections:

A  refined design model.  Evaluate the design on m,n <= 12 (144 + 128 modes), then
   RE-SOLVE the held-frequency problem on that basis (seeded with the design) so the
   equality holds exactly there, and report J/J_UD.
B  independent FE model (ACM + Q4, 48 x 36).  Move the design along the evaluator's
   frequency-gradient direction, theta_hat + c d, and find c by secant iteration so that
   f_FE(steered) = f_FE(UD) to 1e-5; report beta_steered/beta_UD there, and the local
   slope d(J ratio)/d(f ratio) along the family (the sensitivity of the headline number to
   a frequency drift).

    python examples/r1_matched_frequency.py    -> results/r1_matched.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
import time
import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import levers_study as LS
from design_source import heldfreq_design
from nltopt.relaxed_plate import RelaxedPlate, SteeredPlate

th_hat = heldfreq_design()
out = {}

# ------------------------------------------------------------------ A: refined basis
sp0 = SteeredPlate(LS.PLATE)
st0 = sp0.forward(LS.PHI @ th_hat)
ud0 = sp0.forward(0.0, full=False)
out["base_J"], out["base_f"] = st0["J"] / ud0["J"], st0["omega1"] / ud0["omega1"]
P = RelaxedPlate(mmax=12, n_in=8, nx=144, ny=108)
LS.PLATE = P
LS.PHI, LS.PHIX, LS.PHIY = LS.csrbf(P)
LS.NS = LS.PHI.shape[1]
sp = SteeredPlate(P)
ud = sp.forward(0.0, full=False)
st = sp.forward(LS.PHI @ th_hat, full=False)
out["ref_eval_J"], out["ref_eval_f"] = st["J"] / ud["J"], st["omega1"] / ud["omega1"]
print(f"[A] design model: J/J_UD={out['base_J']:.4f} f/f_UD={out['base_f']:.6f}", flush=True)
print(f"[A] refined m,n<=12, n_in=8: design as is J/J_UD={out['ref_eval_J']:.4f} "
      f"f/f_UD={out['ref_eval_f']:.5f}", flush=True)
t0 = time.time()
r = LS.optimise_steered(ud["omega1"], seeds=[th_hat, np.zeros(LS.NS), np.full(LS.NS, np.deg2rad(20))])
out["ref_held_J"], out["ref_held_f"] = r["J"] / ud["J"], r["omega1"] / ud["omega1"]
out["ref_held_theta"] = r["theta_hat"]
print(f"[A] refined, re-held exactly: J/J_UD={out['ref_held_J']:.4f} f/f_UD={out['ref_held_f']:.6f} "
      f"({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ B: FE, matched
import fe_check_design as FC
LS.PLATE = RelaxedPlate()
LS.PHI, LS.PHIX, LS.PHIY = LS.csrbf(LS.PLATE)
sp = SteeredPlate(LS.PLATE)
g = LS.PHI.T @ sp.grad(sp.forward(LS.PHI @ th_hat))["omega1"]
d = g / np.linalg.norm(g)
NX, NY = 48, 36


def fe(x):
    return FC.fe_backbone_c(FC.theta_fun_from(x), nx=NX, ny=NY)


f_ud, b_ud = fe(np.zeros(20))
fam = []
c0, c1 = 0.0, 0.05
f0, b0 = fe(th_hat + c0 * d); fam.append((c0, f0, b0))
f1, b1 = fe(th_hat + c1 * d); fam.append((c1, f1, b1))
for it in range(8):
    c2 = c1 - (f1 - f_ud) * (c1 - c0) / (f1 - f0)
    f2, b2 = fe(th_hat + c2 * d); fam.append((c2, f2, b2))
    print(f"[B] c={c2:+.5f}  f/f_UD={f2/f_ud:.6f}  beta ratio={b2/b_ud:.4f}", flush=True)
    c0, f0, b0, c1, f1, b1 = c1, f1, b1, c2, f2, b2
    if abs(f2 / f_ud - 1) < 1e-5:
        break
fam = np.array(sorted(fam))
fr, br = fam[:, 1] / f_ud, fam[:, 2] / b_ud
slope = np.polyfit(fr, br, 1)[0]
out.update(fe_f_ud=f_ud, fe_b_ud=b_ud, fe_asis_f=fam[fam[:, 0] == 0, 1][0] / f_ud,
           fe_asis_J=fam[fam[:, 0] == 0, 2][0] / b_ud, fe_matched_f=f1 / f_ud, fe_matched_J=b1 / b_ud,
           fe_slope=slope, fe_family=fam, fe_c=c1)
print(f"[B] FE {NX}x{NY}: as designed f/f_UD={out['fe_asis_f']:.5f} ratio={out['fe_asis_J']:.4f};"
      f" matched f/f_UD={out['fe_matched_f']:.6f} ratio={out['fe_matched_J']:.4f};"
      f" d(ratio)/d(f ratio)={slope:.3f}", flush=True)
np.savez(_os.path.join(_ROOT, "results", "r1_matched.npz"), **out)
print("wrote results/r1_matched.npz")
