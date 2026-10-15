r"""Does the size of the lever follow the conflict between the two strain fields?  (P5, R1.)

For each plate of r1_generality.py, the UD fundamental's curvature and relaxed-stretch
fields give, pointwise, the fiber directions that the frequency (at Lambda < 0: across the
principal curvature) and the hardening (across the principal stretch) prefer.  The
energy-weighted mean conflict chi = |phi_K - phi_G| (mod 90 deg) is compared with the
held-frequency steered optimum J/J_UD of that plate.  Also reported: the frozen-field multiplier
Lambda at the held frequency and the frequency that minimizing the stretch energy alone
(Lambda = 0) would select -- an alternative ordering variable.

    python examples/r1_conflict.py      -> results/r1_conflict.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "examples"))
from mechanism_study import fourier_fields, eval_fourier
from r1_generality import CFGS
from nltopt.relaxed_plate import RelaxedPlate, SteeredPlate
from nltopt.laminate import lamina_Q

Q = lamina_Q()
rows = []
for name, kw in CFGS.items():
    P = RelaxedPlate(**kw)
    st = SteeredPlate(P).forward(0.0)
    fk = fourier_fields(st["kv"], P.h ** 3 / 12 * Q)
    fg = fourier_fields(st["eps_r"], 0.5 * P.h * Q)
    chi = np.rad2deg(np.abs(np.angle(np.exp(2j * (fk["phi2"] - fg["phi2"])))) / 2)
    wgt = st["eK"] / st["eK"].sum() + st["eG"] / st["eG"].sum()
    chi_m = float(np.sum(wgt * chi) / wgt.sum())
    g = np.load(_os.path.join(_ROOT, "results", f"r1_generality_{name}.npz"))
    # frozen-field multiplier at the held frequency, and the stretch-only (Lambda = 0) frequency
    tg = np.linspace(-np.pi / 2, np.pi / 2, 361)
    eK = np.stack([eval_fourier(fk["c"], t) for t in tg]); eG = np.stack([eval_fourier(fg["c"], t) for t in tg])
    cols = np.arange(P.Nq)
    om2_of = lambda lam: np.sum(P.w * eK[np.argmin(eG / st["gamma"] - lam * eK / st["om2"], axis=0), cols])
    lo, hi = -5.0, 5.0
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        lo, hi = (lo, lam) if om2_of(lam) > st["om2"] else (lam, hi)
    free = np.sqrt(om2_of(0.0) / st["om2"])
    rows.append((chi_m, float(g["J_steer"] / g["J_ud"]), 0.5 * (lo + hi), free))
    print(f"{name}: energy-weighted conflict {chi_m:5.1f} deg   steered J/J_UD {rows[-1][1]:.3f}   "
          f"Lambda {rows[-1][2]:+.3f}   stretch-only frequency {free:.3f} omega_UD", flush=True)
np.savez(_os.path.join(_ROOT, "results", "r1_conflict.npz"), names=list(CFGS), rows=np.array(rows))
