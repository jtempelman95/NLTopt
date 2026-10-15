r"""Does the jump-free gain depend on the damping level or law?  (Reviewer R1, M3 / m2.)

Proposition 3 gives a_c/h = (8 zeta / (3 sqrt3 beta))^{1/2}: the RATIO of the steered to
the UD jump-free amplitude, (beta_UD/beta_steered)^{1/2}, does not contain zeta, while the
absolute thresholds scale as zeta^{1/2}.  Here a_c is computed numerically (the drive at
which the full-order damped SSM forced response first folds, as in confirm_heldfreq.py)
for mass-proportional damping zeta = 0.5, 1, 2, 4 % and for stiffness-proportional
(C = 2 zeta K / omega_1) damping at 2 %.

    python examples/r1_damping.py       -> results/r1_damping.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
import numpy as np
from scipy.linalg import eigh
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, _os.path.join(_ROOT, "examples"))
import confirm_heldfreq as CH
from design_source import heldfreq_design
from pyhb import MechanicalSystem


def ssm_backbone_law(theta_hat, zeta, law):
    pl, ref, M, K, elem, _ = CH.build(theta_hat)
    w1 = np.sqrt(eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])[0])
    C = 2 * zeta * w1 * M if law == "mass" else 2 * zeta / w1 * K
    ssm = CH.ssm_from_system(MechanicalSystem(M, C, K, [elem]), mode=0, ref_dof=ref, order=3)
    return ssm, ref, M, K, elem


def task(args):
    name, th, zeta, law = args
    CH.ZETA = zeta
    CH.ssm_backbone = lambda t, damped=False: ssm_backbone_law(t, zeta, law)
    r = CH.jump_free_amplitude(th)
    return name, zeta, law, r["a_c"], r["F_c"]


if __name__ == "__main__":
    designs = {"ud": np.zeros(20), "opt": heldfreq_design()}
    jobs = [(k, v, z, "mass") for z in (0.005, 0.01, 0.02, 0.04) for k, v in designs.items()]
    jobs += [(k, v, 0.02, "stiff") for k, v in designs.items()]
    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(task, jobs))
    rows = []
    for z, law in [(0.005, "mass"), (0.01, "mass"), (0.02, "mass"), (0.04, "mass"), (0.02, "stiff")]:
        u = [r for r in res if r[0] == "ud" and r[1] == z and r[2] == law][0]
        o = [r for r in res if r[0] == "opt" and r[1] == z and r[2] == law][0]
        rows.append((z, 0 if law == "mass" else 1, u[3], o[3], o[3] / u[3], o[4] / u[4]))
        print(f"zeta={100*z:.1f}% {law:5s}: a_c/h UD {u[3]:.4f}  steered {o[3]:.4f}  ratio {o[3]/u[3]:.3f}  "
              f"drive ratio {o[4]/u[4]:.3f}", flush=True)
    np.savez(_os.path.join(_ROOT, "results", "r1_damping.npz"), rows=np.array(rows))
    print("wrote results/r1_damping.npz")
