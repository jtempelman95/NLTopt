r"""Amplitude-robust composite beam by manufacturable fiber steering.

The novel design problem: design the tow layout of a clamp-clamp laminated
von Karman beam -- **each ply steered independently** as a per-station angle
field -- so its first-mode **backbone is as amplitude-invariant as possible**
(the resonance drifts as little as possible with amplitude), subject to a
**tow-curvature manufacturability constraint** on every ply.

Mechanism: the laminate bending-extension coupling ``B(theta)`` feeds a quadratic
transverse force that partially cancels the (strong) cubic membrane hardening of
a clamped beam.  Single-angle steering only scales ``EI, EA`` together and can't
touch the hardening; an *unsymmetric*, *steered* layup can.

We parametrize the tow path by a few smooth modes ``theta(x) = sum_k c_k
sin(k pi x/L)`` (smooth => printable; the tow curvature is bounded by the
coefficients) and minimize the backbone drift over an amplitude band with an SSM
backbone, against a unidirectional ``[0]`` baseline.

Honest scope: on a clamped beam the membrane hardening dominates and the ``B``
lever is bounded, so steering *reduces* the amplitude detuning substantially but
does not zero it (true iso-frequency would need a softening reserve, e.g. a
shallow initial curvature -- a documented extension).

    python examples/steered_beam_amplitude_robust.py
"""

import sys
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import numpy as np
import paperstyle as ps
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from nltopt.laminated_beam import LaminatedBeam
from nltopt import ssm_from_system

L, TH, WD, NE = 0.30, 0.002, 0.02, 10
NPLY = 2                        # plies (each steered independently)
KMODE = 2                       # sin modes per ply (+ a constant mean angle)
ORDER = 3                       # SSM order (cubic: robust; a steered beam hits a
                                # 3:1 internal resonance at 5th order, and the
                                # drift is hardening-dominated anyway)
A_MAX = 0.4 * TH                # amplitude band (~1.4 gyration radii)
KAPPA_TOW = np.deg2rad(900.0)   # max |dtheta/dx| (rad/m) -- manufacturability
NVAR = NPLY * (KMODE + 1)       # design variables: per ply, [mean, mode1, mode2]

beam = LaminatedBeam(NE, L, TH, WD, base_angle=0.0)
REF = beam.midspan_w_dof()
_xe = beam.xe


def _ply_field(coeffs, x):
    """One ply angle field: constant mean + sin modes."""
    th = np.full(np.shape(x), coeffs[0], dtype=float)
    for k, ck in enumerate(coeffs[1:], start=1):
        th = th + ck * np.sin(k * np.pi * x / L)
    return th


def ply_angles(c):
    """(n_ply, n_elem) angle field from the packed design vector c."""
    C = np.asarray(c).reshape(NPLY, KMODE + 1)
    return np.vstack([_ply_field(C[p], _xe) for p in range(NPLY)])


def max_tow_curvature(c):
    """max_p max_x |dtheta_p/dx| over all plies (manufacturability)."""
    C = np.asarray(c).reshape(NPLY, KMODE + 1)
    x = np.linspace(0, L, 200)
    worst = 0.0
    for p in range(NPLY):
        dth = np.zeros_like(x)
        for k, ck in enumerate(C[p, 1:], start=1):
            dth += ck * (k * np.pi / L) * np.cos(k * np.pi * x / L)
        worst = max(worst, np.max(np.abs(dth)))
    return worst


def linear_freq(c):
    """First-mode linear frequency (rad/s) -- cheap eig, no SSM."""
    from scipy.linalg import eigh
    beam.set_ply_angles(ply_angles(c))
    ix = np.ix_(beam.free, beam.free)
    w2 = eigh(beam.K_full[ix], beam.M_full[ix], eigvals_only=True)[0]
    return np.sqrt(max(w2, 0.0))


def backbone_drift(c):
    """RMS relative backbone drift |Omega(a)/omega1 - 1| over the amplitude band."""
    beam.set_ply_angles(ply_angles(c))
    ssm = ssm_from_system(beam.to_system(), mode=0, ref_dof=REF, order=ORDER)
    w1 = ssm.omega0
    rho = np.linspace(1e-6, 1.3 * A_MAX / abs(ssm.phi[REF]), 40)
    amp = ssm.physical_amplitude(rho, dof=REF)
    om = ssm._reduced(rho).imag
    sel = np.isfinite(amp) & np.isfinite(om) & (amp <= A_MAX)
    if sel.sum() < 3:
        return 1.0, w1, (np.array([1.0]), np.array([0.0]))   # penalty
    drift = (om[sel] - w1) / w1
    return np.sqrt(np.mean(drift ** 2)), w1, (om[sel] / w1, amp[sel] / TH)


def main():
    # baseline: unidirectional [0/0] (symmetric, no coupling)
    d0, w0, curve0 = backbone_drift(np.zeros(NVAR))
    print(f"baseline [0/0]:  f1={w0/2/np.pi:.1f} Hz   RMS drift={d0*100:.3f}%")

    # optimize tow path to minimize drift, s.t. tow-curvature constraint
    def obj(c):
        return backbone_drift(c)[0]

    # hold the linear frequency within 10% of baseline, so the amplitude-
    # robustness gain is genuine nonlinear tailoring, not just a softer beam
    w_floor = 0.90 * w0
    cons = [{"type": "ineq", "fun": lambda c: KAPPA_TOW - max_tow_curvature(c)},
            {"type": "ineq", "fun": lambda c: linear_freq(c) - w_floor}]
    bounds = [(np.deg2rad(-85), np.deg2rad(85))] * NVAR
    best = None
    rng = np.random.default_rng(1)
    seeds = []
    for _ in range(4):
        s = np.zeros(NVAR)
        s[0] = np.deg2rad(rng.uniform(-25, 25))          # top-ply mean (near axial)
        s[KMODE + 1] = np.deg2rad(rng.uniform(-25, 25))  # bottom-ply mean
        s += np.deg2rad(8 * rng.standard_normal(NVAR))
        seeds.append(s)
    for seed in seeds:
        r = minimize(obj, seed, method="SLSQP", bounds=bounds,
                     constraints=cons, options={"maxiter": 30, "ftol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    c = best.x
    d1, w1, curve1 = backbone_drift(c)
    kap = np.rad2deg(max_tow_curvature(c))
    pa = np.rad2deg(ply_angles(c))
    print(f"optimized:      f1={w1/2/np.pi:.1f} Hz   RMS drift={d1*100:.3f}%   "
          f"reduction {100*(1-d1/d0):.0f}%")
    print(f"  ply means (deg): {np.round(pa.mean(axis=1),1)}   "
          f"max|dtheta/dx|={kap:.0f} deg/m (limit {np.rad2deg(KAPPA_TOW):.0f})")

    np.savez("results/results_steered_beam.npz",
             x=_xe, ply_angles=pa, L=L,
             base_curve=np.array(curve0), opt_curve=np.array(curve1),
             drift0=d0, drift1=d1, f1_base=w0 / 2 / np.pi, f1_opt=w1 / 2 / np.pi,
             kappa=kap, kappa_lim=np.rad2deg(KAPPA_TOW))

    # ---- figure -----------------------------------------------------------
    fig, (axT, axB) = plt.subplots(1, 2, figsize=(ps.FULL, 2.9))
    for p, col in zip(range(NPLY), [ps.GREEN, ps.STEEL]):
        axT.plot(_xe / L, pa[p], "o-", color=col, lw=1.5, ms=4, label=f"ply {p+1}")
    axT.axhline(0, color=ps.GREY, ls="--", lw=1, label=r"baseline [0/0]")
    axT.set_xlabel(r"$x / L$"); axT.set_ylabel(r"ply angle $\theta$ (deg)")
    axT.set_title("(a) steered laminate")
    ps.grid(axT); ps.despine(axT)
    axT.legend(frameon=False, loc="best")

    om0, a0 = curve0
    om1, a1 = curve1
    axB.plot(om0, a0, color=ps.GREY, lw=1.6, label=rf"baseline (drift {d0*100:.2f}\%)")
    axB.plot(om1, a1, color=ps.GREEN, lw=1.6, label=rf"steered (drift {d1*100:.2f}\%)")
    axB.axvline(1.0, color=ps.GRID, ls=":", lw=1)
    axB.set_xlabel(r"$\Omega/\omega_1$"); axB.set_ylabel(r"mid-span amplitude $a/h$")
    axB.set_title("(b) steering flattens the detuning")
    ps.grid(axB); ps.despine(axB)
    axB.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    ps.save(fig, "/home/jrt/NLTopt/figs/steered_beam_amplitude_robust")


if __name__ == "__main__":
    main()
