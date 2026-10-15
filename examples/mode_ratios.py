r"""Linear spectra used in the text: the steered held-frequency design's lowest transverse
frequency ratios and nearest 3:1 commensurability, and the UD plate's lowest in-plane frequency
and rho = 4 omega_1^2 / omega_in^2 (full multi-mode plate of confirm_heldfreq.py).

    python examples/mode_ratios.py  -> results/mode_ratios.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0, (_ROOT + "/examples")); sys.path.insert(0, (_WAVETOPO + ""))
import confirm_heldfreq as C
from design_source import heldfreq_design

out = {}
for tag, th in (("ud", np.zeros(20)), ("opt", heldfreq_design())):
    pl, *_ = C.build(th)
    nw = pl.nw
    wt = np.sqrt(eigh(pl.K[:nw, :nw], pl.M[:nw, :nw], eigvals_only=True))
    wi = np.sqrt(eigh(pl.K[nw:, nw:], pl.M[nw:, nw:], eigvals_only=True))
    r = wt[:4] / wt[0]
    det = np.abs(r[1:] / 3 - 1)
    out[f"{tag}_ratios"] = r; out[f"{tag}_det31"] = det.min(); out[f"{tag}_w_in"] = wi[0]; out[f"{tag}_w1"] = wt[0]
    print(tag, "ratios", np.round(r, 3), "nearest 3:1 detuning %.3f" % det.min(),
          "f_in %.0f Hz" % (wi[0] / 2 / np.pi), "rho %.2e" % (4 * wt[0] ** 2 / wi[0] ** 2))
np.savez("results/mode_ratios.npz", **out)
