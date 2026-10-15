r"""Visualize the SPECTRAL SUBMANIFOLD itself -- the 2-D invariant manifold the
reduction lives on -- for the von Karman plate.

The SSM is the smoothest invariant surface tangent to the master eigenplane; the
reduction slaves every other coordinate to the two master coordinates.  For the
plate the most physical slaved coordinate is the IN-PLANE (membrane) motion: it is
driven at twice the transverse frequency and grows like the square of the
transverse amplitude -- i.e. the manifold's curvature IS the membrane relaxation
that softens the hardening (Sec. plate).  We reconstruct the manifold
``z(rho,theta) = sum_m w_m p1^a p2^b`` (p1 = (rho/2) e^{i theta}) and draw it as a
surface over the master mode's phase plane, with the periodic backbone orbits on it.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/ssm_manifold.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps
import numpy as np
import matplotlib.pyplot as plt

import plate_framework_prototype as P
from plate_confirm import build_plate
from nltopt import ssm_from_system


def manifold_state(s, rho, ths):
    """Full physical state z=(x, xdot) on the manifold orbit of amplitude rho."""
    r = 0.5 * rho
    p1 = r * np.exp(1j * ths); p2 = np.conj(p1)
    z = np.zeros((ths.size, 2 * s.n), dtype=complex)
    for (a, b), wm in s._all_coeffs().items():
        z += np.outer(p1 ** a * p2 ** b, wm)
    return z.real                                            # (n_theta, 2n): [x | xdot]


def panel(ax, s, pl, ref, title, color):
    n = s.n
    w11 = s._all_coeffs().get((1, 1), np.zeros(2 * n)).real  # |p1|^2 (slaving) coeff
    inp = np.arange(pl.nw, pl.ndof)                          # in-plane DOF block
    slave = int(inp[np.argmax(np.abs(w11[:n][inp]))])        # most-slaved in-plane DOF

    amax = 0.34 * pl.h
    rho_max = 2.0 * amax / abs(s.phi[ref])
    RHO = np.linspace(1e-4, rho_max, 30)
    TH = np.linspace(0, 2 * np.pi, 120)
    Xr = np.empty((RHO.size, TH.size)); Vr = np.empty_like(Xr); Xs = np.empty_like(Xr)
    for i, rho in enumerate(RHO):
        z = manifold_state(s, rho, TH)
        Xr[i] = z[:, ref] / pl.h                             # transverse displ / h
        Vr[i] = z[:, n + ref] / (s.omega0 * pl.h)            # transverse velocity (normalized)
        Xs[i] = z[:, slave] * 1e3                            # slaved in-plane [mm-scale modal]

    ax.plot_surface(Xr, Vr, Xs, rcount=30, ccount=60, cmap="viridis",
                    alpha=0.82, linewidth=0, antialiased=True)
    # tangent (linear) eigenplane at the origin: flat z_slave = 0
    ax.plot_surface(Xr, Vr, np.zeros_like(Xs), color="0.7", alpha=0.12, linewidth=0)
    # periodic backbone orbits (constant-rho rings) drawn ON the manifold
    for frac in (0.45, 0.72, 1.0):
        z = manifold_state(s, frac * rho_max, TH)
        ax.plot(z[:, ref] / pl.h, z[:, n + ref] / (s.omega0 * pl.h), z[:, slave] * 1e3,
                color=color, lw=2.2)
    ax.set_xlabel(r"transverse $q/h$", fontsize=9, labelpad=4)
    ax.set_ylabel(r"velocity $\dot q/\omega_1 h$", fontsize=9, labelpad=4)
    ax.set_zlabel(r"slaved membrane DOF", fontsize=9, labelpad=8)
    ax.set_title(title, fontsize=11, pad=0)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="z", pad=6)
    ax.zaxis.get_offset_text().set_visible(False)
    ax.set_zticklabels([])


def main():
    pl0, ref0 = build_plate(np.zeros(P.M))
    s0 = ssm_from_system(pl0.to_system(), mode=0, ref_dof=ref0, order=3)
    print("Optimizing (relaxed SSM objective)...")
    from design_source import heldfreq_design
    best = (None, heldfreq_design())          # held-frequency optimum (single source)
    plo, refo = build_plate(best[1])
    so = ssm_from_system(plo.to_system(), mode=0, ref_dof=refo, order=3)

    fig = plt.figure(figsize=(ps.FULL, 3.2))
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    panel(axA, s0, pl0, ref0, rf"(a) unidirectional ($f_1={s0.omega0/2/np.pi:.0f}$ Hz)", ps.STEEL)
    panel(axB, so, plo, refo, rf"(b) optimized tow ($f_1={so.omega0/2/np.pi:.0f}$ Hz)", ps.GREEN)
    fig.tight_layout()
    ps.save(fig, (_ROOT + "/figs/ssm_manifold"))


if __name__ == "__main__":
    main()
