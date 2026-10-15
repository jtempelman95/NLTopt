r"""Sign and size of the frequency multiplier of Proposition 2 at the held (UD) frequency.

theta*(x) = argmin [ e_G/gamma_0 - Lambda e_K/omega_0^2 ] with Lambda fixed by the frequency.
Lambda > 0 draws fibers toward the principal curvature (raises omega_1); Lambda < 0 pushes
them away from it.  Also reports the frequency that minimizing the stretch energy ALONE
(Lambda = 0) would select, in the same frozen-field model.

    python examples/mechanism_lambda.py  -> results/mechanism_lambda.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples"))
import mechanism_study as MS
from nltopt.relaxed_plate import SteeredPlate

UD = SteeredPlate(MS.P).forward(0.0)
fk = MS.fourier_fields(UD["kv"], MS.H ** 3 / 12 * MS.Q)
fg = MS.fourier_fields(UD["eps_r"], 0.5 * MS.H * MS.Q)
tg = np.linspace(-np.pi / 2, np.pi / 2, 361)
eK = np.stack([MS.eval_fourier(fk["c"], t) for t in tg])
eG = np.stack([MS.eval_fourier(fg["c"], t) for t in tg])
g0, w0, wt = UD["gamma"], UD["om2"], UD["omega1"]
cols = np.arange(MS.P.Nq)


def om2_of(lam):
    idx = np.argmin(eG / g0 - lam * eK / w0, axis=0)
    return np.sum(MS.P.w * eK[idx, cols])


lo, hi = -5.0, 5.0
for _ in range(80):
    lam = 0.5 * (lo + hi)
    lo, hi = (lo, lam) if om2_of(lam) > wt ** 2 else (lam, hi)
lam = 0.5 * (lo + hi)
free = np.sqrt(om2_of(0.0)) / wt
print(f"Lambda = {lam:+.4f}   stretch-only field frequency = {free:.3f} omega_UD")
np.savez("results/mechanism_lambda.npz", lam=lam, free_freq=free)
