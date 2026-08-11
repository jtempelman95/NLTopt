r"""Backbone-shape optimization of a von Karman beam -- replicating Pozzi,
Marconi, Jain, Li & Braghin (2025), Sec. 5.2 / Fig. 3.

The paper shapes an initially curved, clamp-clamp von Karman beam so that its
first-mode **backbone** follows a prescribed softening target, solving

    min_{A1,A2,h,L}   A2 * L
    s.t.  Omega(RMS v = c1 h0) = omega0
          Omega(RMS v = c2 h0) = k2 omega0            (softening: k2 < 1)
          bounds (Table 1)

with design variables the initial-shape amplitudes ``A1, A2`` (of
``y = A1 sin(pi x/L) + A2 sin(2 pi x/L)``), thickness ``h`` and length ``L``;
``h0`` / ``omega0`` are the initial thickness / first frequency; ``v`` is the
mid-span transverse motion.

What we reproduce here
----------------------
The **method and behaviour**: a straight beam only *hardens*; the optimizer
introduces an initial curvature whose quadratic nonlinearity makes the backbone
*soften*, and drives it through two prescribed (amplitude, frequency) points --
computed entirely from the NLTopt cubic-order SSM backbone, with SLSQP.

Two honest departures from the paper's exact numbers:
  * we use the condensed (axially-immovable shallow-arch) von Karman model,
    where the paper uses a full 3-DOF/node element;
  * the target amplitudes ``c1, c2`` are placed in the cubic SSM's verified-
    accurate window (the paper's 0.4 h0 sits at ~2 gyration radii, beyond
    cubic order -- reaching it needs the arbitrary-order SSM, a roadmap item).
We therefore match the *result's character* (curved, softening, threading the
two targets) and verify the optimized design against full-order EPMC, rather
than reproducing A1=10.7 mm etc. to the digit.

    python examples/beam_backbone_opt.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pyhb.nonlinear_modes import nonlinear_mode
from nltopt import ssm_from_system
from nltopt.beam import curved_vk_beam

# --- fixed data -------------------------------------------------------------
H0 = 0.010            # initial thickness  (10 mm)
L0 = 1.000            # initial length     (1000 mm)
E, RHO, NEL = 90e9, 7850.0, 10
C1, C2 = 0.10, 0.20   # target RMS amplitudes as fractions of h0 (cubic-valid)
K2 = 0.95             # softening target at C2*h0

# design-variable bounds (Table 1, in SI): A1, A2 in [0,20]mm; h [1,100]mm; L [0.5,1.5]m
BOUNDS = [(0.0, 0.020), (0.0, 0.020), (0.001, 0.100), (0.5, 1.5)]


def _omega0():
    d = curved_vk_beam(0.0, 0.0, H0, L0, E=E, rho=RHO, n_elem=NEL)
    return d["omega1"], d


OMEGA0, INIT = _omega0()


def backbone_points(mu):
    """Return (Omega at C1 h0, Omega at C2 h0) for design mu=[A1,A2,h,L]."""
    A1, A2, h, L = mu
    d = curved_vk_beam(A1, A2, h, L, E=E, rho=RHO, n_elem=NEL)
    ssm = ssm_from_system(d["system"], mode=0, ref_dof=d["ref"])
    o1, _ = ssm.omega_at_amplitude(C1 * H0, kind="rms")
    o2, _ = ssm.omega_at_amplitude(C2 * H0, kind="rms")
    return o1, o2


def objective(mu):
    return mu[1] * mu[3]                       # A2 * L


def con1(mu):
    return backbone_points(mu)[0] - OMEGA0     # = 0


def con2(mu):
    return backbone_points(mu)[1] - K2 * OMEGA0  # = 0


def main():
    print(f"omega0 (initial straight beam) = {OMEGA0:.3f} rad/s")

    mu0 = np.array([0.004, 0.001, 0.009, 1.0])   # slightly curved start (A1>0)
    res = minimize(objective, mu0, method="SLSQP",
                   bounds=BOUNDS,
                   constraints=[{"type": "eq", "fun": con1},
                                {"type": "eq", "fun": con2}],
                   options={"maxiter": 80, "ftol": 1e-10})

    A1, A2, h, L = res.x
    o1, o2 = backbone_points(res.x)
    resid = max(abs(o1 / OMEGA0 - 1.0), abs(o2 / OMEGA0 - K2))
    # SLSQP flags a boundary optimum (A2 -> 0) as "unsuccessful"; the meaningful
    # test is whether the backbone targets are met.
    print(f"\nSLSQP: {res.nit} iters, obj(A2*L)={res.fun:.4e}, "
          f"targets met={resid < 1e-3} (max residual {resid:.2e})")
    print(f"  A1 = {A1*1e3:7.3f} mm   A2 = {A2*1e3:7.3f} mm   "
          f"h = {h*1e3:7.3f} mm   L = {L*1e3:8.2f} mm")
    print(f"    (A2 -> 0: a single-harmonic arch already threads both targets;"
          f" the paper's wider 0.2/0.4 h0 split needs the 2nd harmonic)")
    print(f"  constraint 1:  Omega(0.1 h0)/omega0 = {o1/OMEGA0:.4f}  (target 1.000)")
    print(f"  constraint 2:  Omega(0.2 h0)/omega0 = {o2/OMEGA0:.4f}  (target {K2:.3f})")

    # ---- backbones: initial straight vs optimized (SSM) + EPMC check -------
    d_opt = curved_vk_beam(A1, A2, h, L, E=E, rho=RHO, n_elem=NEL)
    ssm_opt = ssm_from_system(d_opt["system"], mode=0, ref_dof=d_opt["ref"])
    ssm_ini = ssm_from_system(INIT["system"], mode=0, ref_dof=INIT["ref"])

    def ssm_curve(ssm, amp_max):
        rho = np.linspace(1e-7, 4 * amp_max / abs(ssm.phi[ssm.ref_dof]), 300)
        rms = ssm.rms_amplitude(rho)
        om = ssm.lambda1.imag + ssm.gamma.imag * rho ** 2 / 4.0
        keep = rms <= amp_max
        return om[keep] / OMEGA0, rms[keep] / H0

    amp_max = 0.26 * H0
    oi, ai = ssm_curve(ssm_ini, amp_max)
    oo, ao = ssm_curve(ssm_opt, amp_max)

    # Full-order verification of the optimized backbone at the two target
    # amplitudes.  (This deep, strongly-softening arch drives EPMC's self-excited
    # continuation near snap-through, so instead of tracing the whole -- partly
    # spurious -- branch we read the physical backbone point at each target: run
    # a short continuation and, in a narrow amplitude window around the target,
    # take the highest-frequency point = the primary branch.)
    ref = d_opt["ref"]

    def epmc_omega_at(target_rms):
        a_end = np.log10(1.4 * target_rms * np.sqrt(2) / abs(ssm_opt.phi[ref]))
        nm = nonlinear_mode(d_opt["system"], H=7, mode_index=0, ref_dof=ref,
                            log_a_start=a_end - 2.0, log_a_end=a_end,
                            ds=0.05, ds_max=0.10, max_steps=500)
        rms = nm["phys_amplitude"] / np.sqrt(2.0)
        win = np.abs(rms - target_rms) < 0.04 * target_rms
        return nm["omega"][win][np.argmax(nm["omega"][win])] if win.any() else np.nan

    epmc_pts = [(epmc_omega_at(C1 * H0) / OMEGA0, C1),
                (epmc_omega_at(C2 * H0) / OMEGA0, C2)]
    print(f"  EPMC verification:  Omega/omega0 = "
          f"{epmc_pts[0][0]:.4f} @0.1h0, {epmc_pts[1][0]:.4f} @0.2h0  "
          f"(SSM targets 1.000, {K2:.3f})")

    fig, (axS, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # (a) optimized shape
    xs = np.linspace(0, L, 200)
    y = A1 * np.sin(np.pi * xs / L) + A2 * np.sin(2 * np.pi * xs / L)
    axS.plot(xs / L, y * 1e3, color="#2e7d32", lw=2.2)
    axS.axhline(0, color="0.6", lw=1.0, ls="--", label="initial (straight)")
    axS.set_xlabel("x / L"); axS.set_ylabel("initial shape y  [mm]")
    axS.set_title("(a) optimized beam shape")
    axS.legend(frameon=False, fontsize=9); axS.grid(alpha=0.25)

    # (b) backbones
    axB.plot(oi, ai, color="#b0b0b0", lw=2.0, label="initial straight (hardening)")
    axB.plot(oo, ao, color="#2e7d32", lw=2.4, label="optimized (softening, SSM)")
    axB.plot([p[0] for p in epmc_pts], [p[1] for p in epmc_pts], "kD", ms=8,
             mfc="none", mew=1.6, label="EPMC (full order) check")
    axB.plot([1.0], [C1], "o", color="#d1495b", ms=9, label="targets")
    axB.plot([K2], [C2], "o", color="#d1495b", ms=9)
    axB.axvline(1.0, color="0.85", lw=0.8, ls=":")
    axB.set_xlabel(r"$\Omega/\omega_0$"); axB.set_ylabel(r"RMS mid-span $v / h_0$")
    axB.set_title("(b) backbone: straight -> optimized softening")
    axB.legend(frameon=False, fontsize=8.5, loc="upper left")
    axB.grid(alpha=0.25)

    fig.suptitle("Von Karman beam backbone optimization (cf. Pozzi et al. 2025, Fig. 3)",
                 fontweight="bold")
    fig.tight_layout()
    out = "beam_backbone_opt.png"
    fig.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
