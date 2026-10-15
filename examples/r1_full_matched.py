r"""Matched-frequency confirmation on the FULL multi-mode plate (Reviewer R1 audit, blocker 1).

On the full plate of confirm_heldfreq.py (36 transverse + 72 in-plane modes) the published
steered design comes out at +0.8 % above the UD frequency (the opposite sign to the refined
design basis and the FE model).  Here the design is moved along the design model's
frequency-gradient direction, theta_hat + c d, until the FULL-plate linear frequency equals
the UD value (secant, to 1e-6), and the cubic SSM and full-order EPMC detuning coefficients
are recomputed there.

    python examples/r1_full_matched.py     -> results/r1_full_matched.npz   (needs PyHB)
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import confirm_heldfreq as CH
import epmc_check as EC
import levers_study as LS
from design_source import heldfreq_design
from nltopt.relaxed_plate import SteeredPlate

th = heldfreq_design()
sp = SteeredPlate(LS.PLATE)
g = LS.PHI.T @ sp.grad(sp.forward(LS.PHI @ th))["omega1"]
d = g / np.linalg.norm(g)


def f_full(x):
    pl, ref, M, K, elem, _ = CH.build(x)
    return np.sqrt(eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])[0])


w_ud = f_full(np.zeros(20))
c0, c1 = 0.0, -0.05
f0, f1 = f_full(th + c0 * d), f_full(th + c1 * d)
for _ in range(10):
    c2 = c1 - (f1 - w_ud) * (c1 - c0) / (f1 - f0)
    c0, f0, c1, f1 = c1, f1, c2, f_full(th + c2 * d)
    print(f"c={c1:+.5f}  f/f_UD={f1/w_ud:.7f}", flush=True)
    if abs(f1 / w_ud - 1) < 1e-6:
        break
thm = th + c1 * d
out = dict(c=c1, f_asis=f_full(th) / w_ud, f_matched=f1 / w_ud)
for name, x in (("ud", np.zeros(20)), ("asis", th), ("matched", thm)):
    s = CH.task_ssm(x)
    out[f"{name}_ssm_beta"] = s["beta"]
    print(f"{name}: SSM beta = {s['beta']:.4f}", flush=True)
for name, x in (("ud", np.zeros(20)), ("matched", thm)):
    out[f"{name}_epmc_beta"] = EC.run((name, x))[4]
out["ssm_ratio_asis"] = out["asis_ssm_beta"] / out["ud_ssm_beta"]
out["ssm_ratio_matched"] = out["matched_ssm_beta"] / out["ud_ssm_beta"]
if "matched_epmc_beta" in out:
    out["epmc_ratio_matched"] = out["matched_epmc_beta"] / out["ud_epmc_beta"]
print({k: (round(float(v), 4) if np.ndim(v) == 0 else v) for k, v in out.items()})
np.savez(_os.path.join(_ROOT, "results", "r1_full_matched.npz"), **out)
