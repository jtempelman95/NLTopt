r"""Finer up/down time-integration sweep near the fold (41 steps over Omega/omega_1 in [1.00, 1.08], 300
periods per step) for the UD and held-frequency steered plates, at the drive of confirm_heldfreq.py.

    python examples/confirm_fine_sweep.py  -> results/confirm_fine_sweep.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys, numpy as np
sys.path.insert(0, (_ROOT + "/examples")); sys.path.insert(0, (_WAVETOPO + ""))
import confirm_heldfreq as C
from design_source import heldfreq_design
from concurrent.futures import ProcessPoolExecutor
R = np.load("results/confirm.npz"); F = float(R["Fdrive"]); H = C.H
th = {"ud": np.zeros(20), "opt": heldfreq_design()}
def job(a):
    k, up = a
    o, amp = C.sweep(C.modal_cubic_model(th[k], 6), F, (1.00, 1.08), up=up, n_om=41, periods=300)
    return k, up, o, amp / H
if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, [(k, up) for k in th for up in (True, False)]))
    out = {}
    for k, up, o, a in res:
        out[f"{k}_{'up' if up else 'down'}_o"] = o; out[f"{k}_{'up' if up else 'down'}_a"] = a
    for k in th:
        u, d = out[f"{k}_up_a"], out[f"{k}_down_a"]
        gap = np.abs(u - d); n = int(np.sum(gap > 0.01))
        print(f"{k}: max gap {np.nanmax(gap):.3f} h; frequency steps with gap > 0.01 h: {n} of {len(gap)}; NaN up/down {int(np.isnan(u).sum())}/{int(np.isnan(d).sum())}")
    np.savez("results/confirm_fine_sweep.npz", **out)
