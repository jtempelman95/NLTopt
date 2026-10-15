r"""Full-order EPMC backbone of the UD and held-frequency steered plates (same full model
and mid-plate coordinate as confirm_heldfreq.py), run per design in its own process
with progress output.  H harmonics (default 3: the cubic nonlinearity's leading
harmonic content), N = 4H+4 AFT samples (exact for the cubic force).  Merges beta
into results/confirm.npz.

    python examples/epmc_check.py [H]
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, (_ROOT + "/examples"))
import confirm_heldfreq as C
from design_source import heldfreq_design
from nltopt import ssm_from_system
from pyhb import MechanicalSystem
from pyhb.nonlinear_modes import nonlinear_mode

HARM = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def run(args):
    name, th = args
    t0 = time.time()
    pl, ref, M, K, elem, _ = C.build(th)
    sysm = MechanicalSystem(M, np.zeros_like(M), K, [elem])
    ssm = ssm_from_system(sysm, mode=0, ref_dof=ref, order=3)
    a_end = np.log10(1.1 * 0.35 * C.H / abs(ssm.phi[ref]))
    # the force is an exact cubic polynomial, so N = 4H+4 AFT samples are alias-free
    # (exact); PyHB's default of 128 is ~8x more expensive for no gain.
    nm = nonlinear_mode(sysm, H=HARM, N=4 * HARM + 4, mode_index=0, ref_dof=ref, log_a_start=a_end - 1.5,
                        log_a_end=a_end, ds=0.15, ds_max=0.25, max_steps=200)
    Ae, Oe = nm["phys_amplitude"], nm["omega"]
    o = np.argsort(Ae)
    beta = C.small_amp_beta(Ae[o], Oe[o], ssm.omega0, amax=0.15)
    print(f"[{name}] EPMC H={HARM}: {Ae.size} points, beta={beta:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return name, Ae[o], Oe[o], ssm.omega0, beta, time.time() - t0


def main():
    designs = [("ud", np.zeros(20)), ("opt", heldfreq_design())]
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(run, designs))
    d = dict(np.load("results/confirm.npz"))
    for name, A, O, w1, beta, sec in res:
        d.update({f"{name}_epmc_amp": A, f"{name}_epmc_om": O, f"{name}_epmc_w1": w1,
                  f"{name}_epmc_beta": beta, f"{name}_epmc_seconds": sec, f"{name}_epmc_H": HARM})
    np.savez("results/confirm.npz", **d)
    print("merged EPMC into results/confirm.npz")


if __name__ == "__main__":
    main()
