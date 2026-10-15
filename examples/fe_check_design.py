r"""Discretization check of the HELD-FREQUENCY design on the independent FE model
(ACM Kirchhoff bending + Q4 membrane, static condensation; Appendix C): f1 and the
mid-plate detuning coefficient beta for UD and the optimized tow field, at two meshes.

    python examples/fe_check_design.py      # -> results/fe_design.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys, inspect
sys.path.insert(0, (_ROOT + "/examples")); sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
import plate_fe_backbone as FB
from wavetopo.cfrp import CSRBFMapping
from design_source import heldfreq_design

# FE backbone that also reports the mode's centre value (beta referred to the centre)
src = inspect.getsource(FB.fe_backbone)
src = src.replace("def fe_backbone(", "def fe_backbone_c(").replace(
    "    return om1/(2*np.pi), gamma, Kb_m, beta",
    "    wc = abs(phi[3*nid(nx//2, ny//2)])\n    return om1/(2*np.pi), beta / wc**2")
ns = dict(FB.__dict__); exec(src, ns); fe_backbone_c = ns["fe_backbone_c"]


class _D: pass


def theta_fun_from(theta_hat):
    xs = np.linspace(0, FB.A, 5); ys = np.linspace(0, FB.B, 4)
    SX, SY = np.meshgrid(xs, ys)
    def f(x, y):
        m = _D(); m.cx, m.cy, m.N = np.array([x]), np.array([y]), 1
        return float((CSRBFMapping(m, (SX.ravel(), SY.ravel()), r_s=0.075).Phi @ theta_hat)[0])
    return f


def main():
    th = heldfreq_design()
    out = {}
    for nx, ny in ((32, 24), (48, 36)):
        for name, t in (("ud", np.zeros(20)), ("opt", th)):
            f1, beta = fe_backbone_c(theta_fun_from(t), nx=nx, ny=ny)
            out[f"{name}_{nx}_f"], out[f"{name}_{nx}_beta"] = f1, beta
            print(f"FE {nx}x{ny} {name}: f1={f1:.1f} Hz  beta(centre)={beta:.4f}")
        print(f"   ratio steered/UD = {out[f'opt_{nx}_beta']/out[f'ud_{nx}_beta']:.3f}")
    np.savez("results/fe_design.npz", **out)


if __name__ == "__main__":
    main()
