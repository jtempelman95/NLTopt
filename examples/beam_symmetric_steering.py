r"""The 1-D test of the separation mechanism: a SYMMETRIC (B = 0) steered beam.

Clamped-clamped single-ply-like laminated beam, A = b h E(x), D = b h^3/12 E(x),
E(x) = Qbar_11(theta(x)).  With immovable ends the condensed axial force is
uniform and, for a fixed mode shape phi,

    J  =  gamma / (omega_1^2 phi_mid^2)
       ~  (int phi'^2)^2 / ( 2 int E^{-1} dx  *  int E phi''^2 dx ),

a HARMONIC mean of E over the span against a CURVATURE-weighted mean.  A uniform
angle scales both by the same factor and leaves J exactly invariant (so any change
is due to steering); a steered field lowers E where phi'' ~ 0.

Design: theta(x) mirror-symmetric, piecewise linear on K+1 knots over the half
span; tow-curvature (slope) bound |theta'| <= kappa_tow; frequency floor
omega_1 >= f_hold * omega_UD.  Optimised by SLSQP on the condensed evaluator
(:class:`nltopt.relaxed_beam.RelaxedBeam`, adapted mode, exact relaxation) and
CONFIRMED on the full FE + SSM (:class:`nltopt.laminated_beam.LaminatedBeam`).

    python examples/beam_symmetric_steering.py     # -> results/beam_symmetric.npz
"""
import time
import numpy as np
from scipy.optimize import minimize

from nltopt.relaxed_beam import RelaxedBeam
from nltopt.laminated_beam import LaminatedBeam
from nltopt.laminate import Qbar
from nltopt import ssm_from_system

L, TH, WD = 0.30, 0.002, 0.02
K = 12
RB = RelaxedBeam(60, L, TH, WD)
XK = np.linspace(0.0, L / 2, K + 1)
Q11 = lambda t: Qbar(t)[0, 0]


def field(c, x):
    xh = np.where(x <= L / 2, x, L - x)
    return np.interp(xh, XK, c)


def modulus(c, x):
    return np.array([Q11(t) for t in field(c, x)])


def cond(c):
    return RB.evaluate(modulus(c, RB.xe))


def ssm_confirm(c, ne=30):
    """Full FE + cubic SSM: frequency, small-amplitude detuning coefficient
    (dOmega/omega per a^2) and the paper's RMS drift over a <= 0.4 h."""
    beam = LaminatedBeam(ne, L, TH, WD)
    th = field(c, beam.xe)
    beam.set_ply_angles(np.vstack([th, th]))
    ref = beam.midspan_w_dof()
    ssm = ssm_from_system(beam.to_system(), mode=0, ref_dof=ref, order=3)
    w1 = ssm.omega0
    rho = np.linspace(1e-6, 1.3 * 0.4 * TH / abs(ssm.phi[ref]), 60)
    amp = ssm.physical_amplitude(rho, dof=ref); om = ssm._reduced(rho).imag
    sel = np.isfinite(amp) & (amp <= 0.4 * TH)
    drift = (om[sel] - w1) / w1
    return dict(w1=w1, coeff=ssm.backbone_coeff() / w1, rms=np.sqrt(np.mean(drift ** 2)),
                curve=(om[sel] / w1, amp[sel] / TH), maxB=np.abs(beam.B).max())


def optimise(kappa_deg, f_hold):
    u = cond(np.zeros(K + 1))
    kap = np.deg2rad(kappa_deg)
    dx = np.diff(XK)
    cons = [dict(type="ineq", fun=lambda c: kap - np.abs(np.diff(c) / dx)),
            dict(type="ineq", fun=lambda c: cond(c)["omega1"] / u["omega1"] - f_hold)]
    seeds = []
    for cpos in (0.224 * L, 0.20 * L, 0.25 * L, 0.10 * L):
        seeds.append(np.minimum(np.deg2rad(80), np.deg2rad(80) * np.exp(-((XK - cpos) / 0.035) ** 2)))
    seeds.append(np.full(K + 1, 0.1))
    best = None
    for s0 in seeds:
        r = minimize(lambda c: cond(c)["J"] / u["J"], s0, method="SLSQP",
                     bounds=[(0.0, np.pi / 2)] * (K + 1), constraints=cons,
                     options=dict(maxiter=200, ftol=1e-10, eps=1e-4))
        c = r.x
        ok = (np.abs(np.diff(c) / dx).max() <= kap * 1.001) and \
             (cond(c)["omega1"] / u["omega1"] >= f_hold - 1e-4)
        if ok and (best is None or r.fun < best[0]):
            best = (r.fun, c.copy())
    return best, u


def main():
    out = dict(xk=XK, L=L)
    base = ssm_confirm(np.zeros(K + 1))
    out.update(base_curve=np.array(base["curve"]), base_rms=base["rms"],
               base_coeff=base["coeff"], base_w1=base["w1"])
    print(f"UD (SSM): f1={base['w1']/2/np.pi:.1f} Hz  RMS drift={100*base['rms']:.2f}%")
    # uniform-angle invariance check (condensed + SSM)
    for t in (20, 45):
        cu = np.full(K + 1, np.deg2rad(t))
        s = ssm_confirm(cu)
        print(f"uniform {t:2d} deg: f/f_UD={s['w1']/base['w1']:.3f}  "
              f"coeff/coeff_UD={s['coeff']/base['coeff']:.4f}   (theory: exactly 1)")
        out[f"uniform{t}_ratio"] = s["coeff"] / base["coeff"]
    for kap in (900.0, 1800.0, 1e6):
        for fh in (0.98, 0.90):
            t0 = time.time()
            best, u = optimise(kap, fh)
            if best is None:
                print(f"kappa={kap:g} f>={fh}: no feasible design"); continue
            Jc, c = best
            s = ssm_confirm(c)
            tag = f"k{int(min(kap, 99999))}_f{int(round(100*fh))}"
            print(f"kappa_tow={kap:7g} deg/m  f>={fh:.2f}: condensed J/J_UD={Jc:.3f} | SSM: "
                  f"f/f_UD={s['w1']/base['w1']:.3f}  coeff ratio={s['coeff']/base['coeff']:.3f}  "
                  f"RMS {100*base['rms']:.2f}% -> {100*s['rms']:.2f}%  max|B|={s['maxB']:.0e} "
                  f"[{time.time()-t0:.0f}s]")
            out.update({tag + "_c": c, tag + "_Jcond": Jc, tag + "_coeff": s["coeff"] / base["coeff"],
                        tag + "_f": s["w1"] / base["w1"], tag + "_rms": s["rms"],
                        tag + "_curve": np.array(s["curve"])})
    np.savez("results/beam_symmetric.npz", **out)
    print("wrote results/beam_symmetric.npz")


if __name__ == "__main__":
    main()
