"""Duffing backbone from the cubic-order SSM -- vs analytic and vs EPMC.

The single-DOF Duffing oscillator  ``x'' + omega0^2 x + knl x^3 = 0``  has the
well-known first-order backbone

    Omega(A) = omega0 (1 + 3 knl A^2 / (8 omega0^2)).

This is the smallest possible SSM check: no slaving, so the reduced cubic
coefficient ``gamma`` alone sets the curvature.  We overlay
  * the SSM backbone (this repo),
  * the analytic first-order backbone,
  * PyHB's EPMC nonlinear-mode continuation (full order),
and print the coefficients.

    python examples/duffing_ssm.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyhb import MechanicalSystem, CubicSpring
from pyhb.nonlinear_modes import nonlinear_mode

from nltopt import ssm_from_system

omega0_sq, knl = 1.0, 1.0
omega0 = np.sqrt(omega0_sq)


def build():
    return MechanicalSystem([[1.0]], [[0.0]], [[omega0_sq]], [CubicSpring([1.0], knl)])


# A cubic (3rd-order) 2-D SSM is an asymptotic small-amplitude expansion: its
# backbone slope is exact, and its error grows like A^4.  We therefore present
# it within (and just past) its valid window -- and shade where the frequency
# shift exceeds ~6%, beyond which a *cubic* model should not be trusted (higher
# expansion order, on the roadmap, extends this range).
A_VALID = 0.35          # ~5-6% frequency shift for these parameters


def main():
    ssm = ssm_from_system(build())

    # SSM backbone -- sample generously, then present the valid window
    rho = np.linspace(1e-4, 0.9, 400)
    bb = ssm.backbone(rho)
    A_ssm, Om_ssm = bb["amplitude"], bb["omega"]

    # analytic first-order backbone (a parabola), over the plotted amplitude range
    A_an = np.linspace(0, 0.55, 200)
    Om_an = omega0 * (1.0 + 3.0 * knl * A_an ** 2 / (8.0 * omega0_sq))

    # EPMC (full order) -- the reference
    nm = nonlinear_mode(build(), H=9, log_a_start=-2.5, log_a_end=-0.05,
                        ds=0.04, ds_max=0.08)

    print(f"SSM   gamma            = {ssm.gamma:.6f}")
    print(f"SSM   backbone kappa   = {ssm.backbone_coeff():.6f}")
    print(f"exact backbone kappa   = {3*knl/(8*omega0):.6f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    # split the SSM curve into its trusted window and its extrapolation
    valid = A_ssm <= A_VALID
    ax.plot(Om_an, A_an, color="0.65", lw=3.2, label="analytic (1st order)")
    ax.plot(nm["omega"], nm["phys_amplitude"], "k--", lw=1.7, label="EPMC (full order)")
    ax.plot(Om_ssm[valid], A_ssm[valid], color="#d1495b", lw=2.2,
            label="SSM cubic (trusted)")
    ax.plot(Om_ssm[~valid], A_ssm[~valid], color="#d1495b", lw=1.3, ls=":",
            label="SSM cubic (extrapolated)")
    ax.axhspan(A_VALID, 0.55, color="0.9", zorder=0)
    ax.annotate("cubic order\nun-trusted here", (1.11, 0.47), color="0.4",
                fontsize=8.5, ha="center", va="center")
    ax.set_xlabel(r"backbone frequency $\Omega$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_title("Duffing backbone: cubic SSM vs analytic vs EPMC\n"
                 "(exact slope; A$^4$ truncation error at large amplitude)")
    ax.set_xlim(0.995, 1.16)
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = "duffing_ssm_backbone.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # honest error report vs the *exact* conservative Duffing backbone
    # (elliptic-integral solution) -- independent of EPMC sampling
    from scipy.special import ellipk

    def om_exact(A):
        m = A ** 2 / (2.0 * (1.0 + A ** 2))          # scipy uses parameter m = k^2
        return (np.pi / 2.0) * np.sqrt(1.0 + A ** 2) / ellipk(m)

    print("\n  A     Om_SSM   Om_exact   err(% of shift)")
    for At in [0.1, 0.2, 0.3, 0.4]:
        i = np.argmin(np.abs(A_ssm - At))
        oe = om_exact(A_ssm[i])
        err = 100 * (Om_ssm[i] - oe) / (oe - 1.0)
        print(f"{A_ssm[i]:5.2f}  {Om_ssm[i]:8.5f} {oe:8.5f}   {err:6.1f}")


if __name__ == "__main__":
    main()
