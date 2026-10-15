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
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import numpy as np
import paperstyle as ps
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
    fig, ax = plt.subplots(1, 3, figsize=(ps.FULL, 2.55))
    ax[0].plot(ns, bc, "o-", color=ps.GREEN, lw=1.5, ms=4)
    ax[0].axhline(bc[0], color=ps.GREY, ls="--", lw=1, label="restrained (immovable)")
    ax[0].axhline(bc[-1], color=ps.RED, ls=":", lw=1.3, label="relaxed (converged)")
    ax[0].set_xlabel(r"in-plane modes / direction  $n_{\mathrm{in}}$")
    ax[0].set_ylabel(r"backbone hardening $d\Omega/dA^2$")
    ax[0].set_title("(a) relaxation softens hardening")
    ps.grid(ax[0]); ps.despine(ax[0])
    ax[0].legend(frameon=False, loc="best")

    ax[1].plot(o_r, a_r, color=ps.GREY, lw=1.6, label="restrained")
    ax[1].plot(o_x, a_x, color=ps.GREEN, lw=1.6, label="relaxed")
    ax[1].axvline(1.0, color=ps.GRID, ls=":", lw=1)
    ax[1].set_xlabel(r"$\Omega/\omega_1$"); ax[1].set_ylabel(r"mid-plate amplitude $A/h$")
    ax[1].set_title("(b) relaxation reduces detuning")
    ps.grid(ax[1]); ps.despine(ax[1])
    ax[1].legend(frameon=False, loc="best")

    sp = 1
    ax[2].quiver(GX[::sp, ::sp], GY[::sp, ::sp], U[::sp, ::sp], V[::sp, ::sp],
                 np.hypot(U, V)[::sp, ::sp], cmap="cividis", pivot="mid", scale=None)
    ax[2].set_aspect("equal"); ax[2].set_xlim(0, A); ax[2].set_ylim(0, B)
    ax[2].set_xlabel(r"$x$ (m)"); ax[2].set_ylabel(r"$y$ (m)")
    ax[2].set_title(r"(c) in-plane field slaved to $w^2$")
    ps.despine(ax[2])

    fig.tight_layout()
    ps.save(fig, (_ROOT + "/figs/plate_membrane_relaxation"))
    print(f"restrained bc={bc[0]:.3e}  relaxed bc={bc[-1]:.3e}  "
          f"({100*(1-bc[-1]/bc[0]):.0f}% reduction)")


if __name__ == "__main__":
    main()
