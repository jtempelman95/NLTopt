r"""Membrane relaxation in the von Karman plate: the in-plane DOF slave to w^2 and
soften the cubic hardening.

Three panels:
 (a) the backbone hardening coefficient vs. the number of in-plane modes -- from
     the restrained (immovable-edge, purely cubic) upper bound down to the
     converged, physically-correct value;
 (b) the first-mode backbone, restrained vs. relaxed (the relaxation bends the
     curve back toward iso-frequency);
 (c) the in-plane displacement field slaved to the fundamental transverse mode
     (the membrane relaxation pattern) that the SSM order-2 manifold captures.

    python examples/plate_membrane_relaxation.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nltopt.plate import VonKarmanPlate
from nltopt import ssm_from_system

A, B, H = 0.20, 0.15, 0.002


def main():
    ref = VonKarmanPlate(A, B, H, n_modes=3, n_in=0).midspan_mode()

    # (a) convergence of the hardening coefficient with in-plane modes ----------
    ns = [0, 1, 2, 3, 4, 5, 6]
    bc = []
    for n_in in ns:
        pl = VonKarmanPlate(A, B, H, n_modes=3, n_in=n_in, nq=12)
        ssm = ssm_from_system(pl.to_system(), mode=0, ref_dof=pl.midspan_mode(), order=3)
        bc.append(ssm.backbone_coeff())
    bc = np.array(bc)

    # (b) backbone curves, restrained (n_in=0) vs relaxed (n_in=5) --------------
    def backbone(n_in):
        pl = VonKarmanPlate(A, B, H, n_modes=3, n_in=n_in, nq=12)
        r = pl.midspan_mode()
        ssm = ssm_from_system(pl.to_system(), mode=0, ref_dof=r, order=3)
        rho = np.linspace(1e-9, 0.9 * H / abs(ssm.phi[r]), 60)
        bbk = ssm.backbone(rho)
        keep = bbk["amplitude"] <= 0.6 * H
        return bbk["omega"][keep] / ssm.omega0, bbk["amplitude"][keep] / H
    o_r, a_r = backbone(0)
    o_x, a_x = backbone(5)

    # (c) in-plane field slaved to the fundamental transverse mode -------------
    pl = VonKarmanPlate(A, B, H, n_modes=1, n_in=6, nq=14)
    el = pl._force
    amp = 0.5 * H
    q = np.zeros(pl.ndof); q[0] = amp / abs(np.sin(np.pi/2)**2)   # fundamental w
    f, *_ = el._internal(q)                                        # in-plane drive
    Kmem = pl._Kmem
    su, sv = pl._su, pl._sv
    Kin = Kmem[pl.nw:, pl.nw:]
    qin = -np.linalg.solve(Kin, f[pl.nw:])                        # slaved in-plane
    qu, qv = qin[:pl.nu], qin[pl.nu:]
    gx = np.linspace(0, A, 26); gy = np.linspace(0, B, 20)
    GX, GY = np.meshgrid(gx, gy)
    U = np.zeros_like(GX); V = np.zeros_like(GX)
    for c, (mi, ni) in enumerate(pl.in_modes):
        s = np.sin(mi*np.pi*GX/A) * np.sin(ni*np.pi*GY/B)
        U += qu[c] * s; V += qv[c] * s

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ns, bc, "o-", color="#2e7d32", lw=2)
    ax[0].axhline(bc[0], color="0.6", ls="--", lw=1, label="restrained (immovable)")
    ax[0].axhline(bc[-1], color="#d1495b", ls=":", lw=1.5, label="relaxed (converged)")
    ax[0].set_xlabel("in-plane modes per direction  $n_{in}$")
    ax[0].set_ylabel(r"backbone hardening $d\Omega/dA^2$")
    ax[0].set_title(f"(a) membrane relaxation softens hardening "
                    f"({100*(1-bc[-1]/bc[0]):.0f}%)")
    ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(o_r, a_r, color="0.55", lw=2.4, label="restrained (immovable edges)")
    ax[1].plot(o_x, a_x, color="#2e7d32", lw=2.4, label="relaxed (in-plane DOF)")
    ax[1].axvline(1.0, color="0.85", ls=":", lw=1)
    ax[1].set_xlabel(r"$\Omega/\omega_1$"); ax[1].set_ylabel(r"mid-plate amplitude $A/h$")
    ax[1].set_title("(b) backbone: relaxation reduces detuning")
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=0.3)

    sp = 1
    ax[2].quiver(GX[::sp, ::sp], GY[::sp, ::sp], U[::sp, ::sp], V[::sp, ::sp],
                 np.hypot(U, V)[::sp, ::sp], cmap="viridis", pivot="mid", scale=None)
    ax[2].set_aspect("equal"); ax[2].set_xlim(0, A); ax[2].set_ylim(0, B)
    ax[2].set_title(r"(c) in-plane field slaved to $w^2$ (membrane relaxation)")

    fig.suptitle("Von Karman plate: in-plane relaxation slaves to the transverse mode "
                 "(SSM order-2 manifold)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("figs/plate_membrane_relaxation.png", dpi=130)
    print(f"restrained bc={bc[0]:.3e}  relaxed bc={bc[-1]:.3e}  "
          f"({100*(1-bc[-1]/bc[0]):.0f}% reduction)")
    print("wrote figs/plate_membrane_relaxation.png")


if __name__ == "__main__":
    main()
