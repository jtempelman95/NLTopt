r"""Backbone-sensitivity validation -- replicating Pozzi, Marconi, Jain, Li &
Braghin (2025), Sec. 5.1 / Fig. 2.

Two coupled nonlinear oscillators (Eq. 33; ``m=1, k=1, k2=0.5, k3=0.2``, Rayleigh
``betaR=0.1``).  We compute the first-mode **backbone sensitivity**
``dOmega/dmu`` at fixed RMS amplitude of ``x2`` for each system parameter
``mu in {m, k, k2, k3}``, and use it to *predict* the perturbed backbone,

    Omega(mu + dmu) ~= Omega(mu) + (dOmega/dmu) dmu           (paper Eq. 34)

for the perturbations ``dm=0.01m, dk=0.01k, dk2=0.03k2, dk3=0.03k3``.  The
prediction (crosses) is overlaid on the *actually recomputed* perturbed backbone
(colored line) and the nominal backbone (black), reproducing Fig. 2:
perturbations in ``m, k`` shift the frequency; ``k2, k3`` bend the backbone at
higher amplitude.

The backbone here **softens then hardens** (a "backward-C"): the quadratic
nonlinearity softens at low amplitude, the cubic hardening turns it back near
RMS x2 ~ 1.  Capturing that turn-back needs a **high-order** SSM -- a cubic
(order-3) SSM gives only a monotone parabola.  We use **order 7**, which tracks
EPMC through the turn-back (the turn-back sits near the SSM's convergence radius
for this system, so order 7 is about as far as a finite expansion reaches).

Notes.  The paper computes ``dOmega/dmu`` with the analytic **adjoint** (App. F)
and a 5th-order SSM.  Here we use a **finite-difference** sensitivity through our
order-7 SSM -- fine for four parameters; the analytic adjoint (roadmap item c)
computes the *same* quantity at a cost independent of the number of parameters.

    python examples/coupled_osc_sensitivity.py
"""

import sys
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import numpy as np
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nltopt import ssm_from_system
from nltopt.benchmarks import coupled_oscillators

NOMINAL = dict(m=1.0, k=1.0, k2=0.5, k3=0.2)
FRAC = dict(m=0.01, k=0.01, k2=0.03, k3=0.03)          # perturbation fractions
DOF = 1                                                 # track x2
ORDER = 7                                                # SSM order (turn-back)
LEVELS = np.linspace(0.12, 1.05, 9)                     # RMS x2 sample points
RMS_MAX = 1.25
COLORS = dict(m="#1f77b4", k="#ff7f0e", k2="#2ca02c", k3="#d62728")


def _ssm(params):
    sys = coupled_oscillators(**params)
    return ssm_from_system(sys, mode=0, ref_dof=DOF, order=ORDER)


def omega_at_levels(params, levels):
    """Backbone frequency at each prescribed RMS-``x2`` level."""
    ssm = _ssm(params)
    return np.array([ssm.omega_at_amplitude(a, dof=DOF, kind="rms")[0]
                     for a in levels])


def backbone_curve(params, rms_max):
    """Dense (Omega, RMS x2) backbone curve for plotting a line (full SSM order)."""
    ssm = _ssm(params)
    rho = np.linspace(1e-5, 5.0, 400)
    rms = ssm.rms_amplitude(rho, dof=DOF)
    om = ssm._reduced(rho).imag                     # full order (soften-to-harden)
    keep = rms <= rms_max
    return om[keep], rms[keep]


def sensitivity(param, levels, eps_rel=1e-4):
    """Central-difference dOmega/dparam at each RMS level (fixed amplitude)."""
    v = NOMINAL[param]
    eps = eps_rel * v
    p_plus = dict(NOMINAL, **{param: v + eps})
    p_minus = dict(NOMINAL, **{param: v - eps})
    return (omega_at_levels(p_plus, levels) - omega_at_levels(p_minus, levels)) / (2 * eps)


def main():
    ssm0 = _ssm(NOMINAL)
    print(f"nominal: omega0={ssm0.omega0:.4f}  lambda1={ssm0.lambda1:.4f}  "
          f"gamma={ssm0.gamma:.3e} ({'soften' if ssm0.gamma.imag<0 else 'harden'})")

    om_nom = omega_at_levels(NOMINAL, LEVELS)
    rms_max = RMS_MAX
    on_line, ar_line = backbone_curve(NOMINAL, rms_max)

    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, 5.0), sharex=False, sharey=True)
    print(f"\n{'param':>5} {'dOmega/dmu @0.3':>16} {'dmu':>8} {'max |pred-actual|':>18}")
    for ax, param in zip(axes.ravel(), ["m", "k", "k2", "k3"]):
        dmu = FRAC[param] * NOMINAL[param]
        sens = sensitivity(param, LEVELS)
        om_pred = om_nom + sens * dmu                      # Eq. (34)

        p_pert = dict(NOMINAL, **{param: NOMINAL[param] + dmu})
        om_act = omega_at_levels(p_pert, LEVELS)
        on_p, ar_p = backbone_curve(p_pert, rms_max)

        sub = param if len(param) == 1 else f"{param[0]}_{param[1]}"
        ax.plot(on_line, ar_line, color=ps.INK, lw=1.3)
        ax.plot(on_p, ar_p, color=ps.STEEL, lw=1.5)
        ax.plot(om_pred, LEVELS, "+", color=ps.RED, ms=6.5, mew=1.3)
        ax.set_title(rf"$\delta {sub} = {FRAC[param]*100:.0f}\%$")
        ps.grid(ax); ps.despine(ax)

        s_at = np.interp(0.30, LEVELS, sens)
        err = np.max(np.abs(om_pred - om_act))
        print(f"{param:>5} {s_at:16.4f} {dmu:8.4f} {err:18.2e}")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"RMS amplitude of $x_2$")
    for ax in axes[1, :]:
        ax.set_xlabel(r"backbone frequency $\Omega$")
    handles = [Line2D([0], [0], color=ps.INK, lw=1.3, label="nominal backbone"),
               Line2D([0], [0], color=ps.STEEL, lw=1.5, label="perturbed backbone (recomputed)"),
               Line2D([0], [0], color=ps.RED, marker="+", ls="none", ms=6.5, mew=1.3,
                      label="sensitivity prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    ps.save(fig, "/home/jrt/NLTopt/figs/coupled_osc_sensitivity")


if __name__ == "__main__":
    main()
