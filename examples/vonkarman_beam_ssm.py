"""Clamped-clamped von Karman beam backbone from the cubic-order SSM.

Literature-style replication (cf. Pozzi, Marconi, Jain, Li & Braghin 2025,
Sec. 5.2): a geometrically nonlinear, clamped-clamped beam whose mid-plane
stretching (axially-immovable ends) gives a distributed *cubic* nonlinearity
and a **hardening** first-mode backbone.

We plot the backbone in the standard dimensionless form used throughout the
nonlinear-beam literature:

    x-axis:  Omega / omega1        (frequency, normalized by the linear mode)
    y-axis:  A / r                 (mid-span amplitude in *gyration radii*,
                                     r = sqrt(EI / EA) ~ beam thickness)

in which the first-mode backbone is essentially thickness-independent,
``Omega/omega1 = 1 + c (A/r)^2`` with ``c ~ 0.022`` for clamped-clamped ends.
We use a slender beam (L/r = 100) so those amplitudes are physically small
deflections, compute the backbone with the NLTopt cubic-order SSM, and
cross-check against PyHB's EPMC.

This is the bridge to composites: replacing the uniform ``EI``/``EA`` with a
laminated, fiber-angle-dependent ``EI(theta)``/``EA(theta)`` turns the very same
backbone into a *design output* of the tow path -- the next milestone.

    python examples/vonkarman_beam_ssm.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyhb import MechanicalSystem, BeamModel
from pyhb.nonlinear_elements import VonKarmanBeamForce
from pyhb.nonlinear_modes import nonlinear_mode

from nltopt import ssm_from_system

N_ELEM = 12
LENGTH = 1.0
EA = 1.0
SLENDERNESS = 20.0                  # L / r  (A/r=2 <-> deflection 0.1 L: valid von Karman)
EI = (LENGTH / SLENDERNESS) ** 2 * EA
RHOA = 1.0
R_GYR = np.sqrt(EI / EA)            # gyration radius ~ thickness
AR_VALID = 1.6                      # cubic SSM trusted to ~5-6% frequency shift
AR_MAX = 2.0


def build():
    beam = BeamModel(n_elem=N_ELEM, length=LENGTH, EI=EI, rhoA=RHOA, EA=EA)
    beam.clamped_clamped()
    vk = VonKarmanBeamForce(beam.Kg, EA / (2.0 * LENGTH))
    sys = MechanicalSystem(beam.M, np.zeros_like(beam.M), beam.K, [vk])
    ref = beam.free_index_of(beam.node_dof(N_ELEM // 2, "w"))  # mid-span w
    return sys, beam, ref


def main():
    sys, beam, ref = build()
    ssm = ssm_from_system(sys, mode=0, ref_dof=ref)
    w0 = ssm.omega0
    c = ssm.backbone_coeff() * R_GYR ** 2 / w0

    print(f"linear omega1          = {w0:.4f}")
    print(f"gyration radius r      = {R_GYR:.4f}   (L/r = {SLENDERNESS:.0f})")
    print(f"SSM gamma (cubic)      = {ssm.gamma:.4e}  ->  "
          f"{'hardening' if ssm.hardening else 'softening'}")
    print(f"normalized backbone    = Omega/omega1 = 1 + {c:.4f} (A/r)^2")

    # SSM backbone, amplitude expressed in gyration radii
    rho = np.linspace(1e-6, 1.3 * AR_MAX * R_GYR / abs(ssm.phi[ref]), 400)
    bb = ssm.backbone(rho)
    ar_ssm = bb["amplitude"] / R_GYR
    on_ssm = bb["omega"] / w0
    keep = ar_ssm <= AR_MAX
    ar_ssm, on_ssm = ar_ssm[keep], on_ssm[keep]
    valid = ar_ssm <= AR_VALID

    # EPMC cross-check (full order); H=5 and a coarse step are ample for this
    # smooth low-amplitude backbone and keep the example fast (~10 s).  We stop
    # just past A/r = AR_MAX and bound max_steps so the continuation can never
    # wander (the default cap is 10000).
    a_end = np.log10(1.1 * AR_MAX * R_GYR / abs(ssm.phi[ref]))
    nm = nonlinear_mode(sys, H=5, mode_index=0, ref_dof=ref,
                        log_a_start=a_end - 1.6, log_a_end=a_end,
                        ds=0.10, ds_max=0.15, max_steps=300)
    ar_epmc = nm["phys_amplitude"] / R_GYR
    on_epmc = nm["omega"] / w0
    ek = ar_epmc <= AR_MAX

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(on_epmc[ek], ar_epmc[ek], "k--", lw=1.7, label="EPMC (full order)")
    ax.plot(on_ssm[valid], ar_ssm[valid], color="#2e7d32", lw=2.4,
            label="SSM cubic (trusted)")
    ax.plot(on_ssm[~valid], ar_ssm[~valid], color="#2e7d32", lw=1.4, ls=":",
            label="SSM cubic (extrapolated)")
    ax.axhspan(AR_VALID, AR_MAX, color="0.92", zorder=0)
    ax.annotate("cubic order\nun-trusted", (1.007, 1.82), color="0.45",
                fontsize=8.5, ha="center", va="center")
    ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"normalized backbone frequency $\Omega/\omega_1$")
    ax.set_ylabel(r"mid-span amplitude $A/r$   ($r=\sqrt{EI/EA}$)")
    ax.set_title("Clamped-clamped von Karman beam: first-mode backbone\n"
                 f"hardening, $\\Omega/\\omega_1 = 1 + {c:.3f}\\,(A/r)^2$")
    ax.set_xlim(0.999, on_ssm[valid].max() + 0.01)
    ax.set_ylim(0, AR_MAX)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = "vonkarman_beam_ssm_backbone.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
