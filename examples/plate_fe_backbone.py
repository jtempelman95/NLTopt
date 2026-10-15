r"""Nonlinear FE ground truth for the plate BACKBONE coefficient.

Extend the validated ACM bending FE (fe_plate) with a Q4 membrane, and compute the
RELAXED effective cubic gamma_eff = knl - 2 g^T K_in^-1 g by STATIC in-plane
condensation on the mesh -- the exact same backbone coefficient the paper's plate
results use (Eq. gammaeff), now discretized by FE. Compare to the assumed-modes
(Galerkin sine) gamma_eff for UD (validation: must match) and a steered field (the
discretization error in the NONLINEAR hardening). No EPMC / full-order continuation:
the condensation is exact at cubic order and costs one sparse solve."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys; sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
from scipy.linalg import eigh
from nltopt.laminate import Qbar, RHO
from nltopt.plate import VonKarmanPlate
import plate_fe_check as FE

A, B, H, CB, Q0 = FE.A, FE.B, FE.H, FE.CB, FE.Q0
CM = H                                              # A_membrane = CM * Qbar


def acm_Ci(lx, ly):
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    C = np.zeros((12, 12))
    for k, (s, t) in enumerate(corners):
        C[3*k+0] = FE._mono(s, t); C[3*k+1] = FE._mono_t(s, t)/ly; C[3*k+2] = -FE._mono_s(s, t)/lx
    return np.linalg.inv(C)


def q4_B(s, t, lx, ly):
    Ns = np.array([-(1-t), (1-t), t, -t]) / lx
    Nt = np.array([-(1-s), -s, s, (1-s)]) / ly
    Bm = np.zeros((3, 8))
    for i in range(4):
        Bm[0, 2*i] = Ns[i]; Bm[1, 2*i+1] = Nt[i]; Bm[2, 2*i] = Nt[i]; Bm[2, 2*i+1] = Ns[i]
    return Bm


def fe_backbone(theta_fun, nx=28, ny=21):
    lx, ly = A/nx, B/ny; nnode = (nx+1)*(ny+1)
    nid = lambda ix, iy: iy*(nx+1)+ix
    Kb = np.zeros((3*nnode, 3*nnode)); Mb = np.zeros((3*nnode, 3*nnode))
    Kin = np.zeros((2*nnode, 2*nnode)); Ci = acm_Ci(lx, ly)
    cells = []
    for iy in range(ny):
        for ix in range(nx):
            th = theta_fun((ix+0.5)*lx, (iy+0.5)*ly)
            Am = CM*Qbar(th, Q0)
            ke, me = FE._elem(lx, ly, CB*Qbar(th, Q0))
            kin_e = np.zeros((8, 8))
            for i, s in enumerate(FE._GS):
                for j, t in enumerate(FE._GS):
                    wq = FE._GWS[i]*FE._GWS[j]*lx*ly
                    Bm = q4_B(s, t, lx, ly); kin_e += wq*(Bm.T @ Am @ Bm)
            nodes = [nid(ix, iy), nid(ix+1, iy), nid(ix+1, iy+1), nid(ix, iy+1)]
            bd = np.array([[3*n, 3*n+1, 3*n+2] for n in nodes]).ravel()
            md = np.array([[2*n, 2*n+1] for n in nodes]).ravel()
            Kb[np.ix_(bd, bd)] += ke; Mb[np.ix_(bd, bd)] += me; Kin[np.ix_(md, md)] += kin_e
            cells.append((bd, md, Am))
    # SS bending mode (w=0 on boundary)
    bfix = [3*nid(ix, iy) for ix in range(nx+1) for iy in range(ny+1) if ix in (0, nx) or iy in (0, ny)]
    bfree = np.setdiff1d(np.arange(3*nnode), bfix)
    w2, V = eigh(Kb[np.ix_(bfree, bfree)], Mb[np.ix_(bfree, bfree)])
    phi = np.zeros(3*nnode); phi[bfree] = V[:, 0]
    phi /= np.max(np.abs(phi[0::3]))               # normalize peak transverse deflection = 1
    Kb_m = phi @ Kb @ phi; Mmod = phi @ Mb @ phi; om1 = np.sqrt(Kb_m/Mmod)
    # nonlinear membrane coupling g and raw cubic knl (von Karman)
    g = np.zeros(2*nnode); knl = 0.0
    for (bd, md, Am) in cells:
        ae = Ci @ phi[bd]
        for i, s in enumerate(FE._GS):
            for j, t in enumerate(FE._GS):
                wq = FE._GWS[i]*FE._GWS[j]*lx*ly
                wx = (FE._mono_s(s, t) @ ae)/lx; wy = (FE._mono_t(s, t) @ ae)/ly
                eNL = np.array([0.5*wx*wx, 0.5*wy*wy, wx*wy])
                Bm = q4_B(s, t, lx, ly)
                g[md] += wq*(Bm.T @ Am @ eNL)
                knl += wq*2.0*(eNL @ Am @ eNL)
    mfix = [d for ix in range(nx+1) for iy in range(ny+1) if ix in (0, nx) or iy in (0, ny)
            for d in (2*nid(ix, iy), 2*nid(ix, iy)+1)]          # u=v=0 on boundary
    mfree = np.setdiff1d(np.arange(2*nnode), mfix)
    gf = g[mfree]; ustar = np.linalg.solve(Kin[np.ix_(mfree, mfree)], gf)
    gamma = knl - 2.0*(gf @ ustar)
    beta = (3.0/8.0)*(gamma/Kb_m)*H**2                          # Omega/omega1 = 1 + beta (a/h)^2
    return om1/(2*np.pi), gamma, Kb_m, beta


def modal_backbone(theta_fun, n_modes=8, n_in=3):
    pl = VonKarmanPlate(A, B, H, n_modes=n_modes, n_in=n_in, nq=16)
    pl.set_theta(np.array([theta_fun(x, y) for x, y in zip(pl.qx, pl.qy)]))
    nw = pl.nw
    w2, V = eigh(pl.K[:nw, :nw], pl.M[:nw, :nw]); c = V[:, 0]
    cen = np.array([np.sin(m*np.pi*0.5)*np.sin(n*np.pi*0.5) for (m, n) in pl.w_modes])
    c = c / (c @ cen)                                           # peak deflection = 1
    gradw = np.stack([c @ pl.PHIx, c @ pl.PHIy])               # (2,Nq)
    eNL = np.stack([0.5*gradw[0]**2, 0.5*gradw[1]**2, gradw[0]*gradw[1]])
    # modal in-plane strain-displacement (sin-sin u,v modes)
    cols = []
    for comp in ("u", "v"):
        for k in range(pl.nu):
            z = np.zeros(pl.Nq)
            cols.append(np.stack([pl.PSIx[k], z, pl.PSIy[k]]) if comp == "u"
                        else np.stack([z, pl.PSIy[k], pl.PSIx[k]]))
    Bmem = np.stack(cols, axis=1)                              # (3, ndof_in, Nq)
    Aq = pl.Aq                                                 # (Nq,3,3) membrane stiffness
    Kin = np.einsum('q,aiq,qab,bjq->ij', pl.qw, Bmem, Aq, Bmem)
    g = np.einsum('q,aiq,qab,bq->i', pl.qw, Bmem, Aq, eNL)
    knl = 2.0*np.einsum('q,aq,qab,bq->', pl.qw, eNL, Aq, eNL)
    gamma = knl - 2.0*(g @ np.linalg.solve(Kin, g))
    Kb = c @ pl.K[:nw, :nw] @ c
    om1 = np.sqrt(Kb/(c @ pl.M[:nw, :nw] @ c))
    return om1/(2*np.pi), gamma, Kb, (3.0/8.0)*(gamma/Kb)*H**2


if __name__ == "__main__":
    fields = {"UD [0deg] (homogeneous)": lambda x, y: 0.0,
              "steered (20+45 sin.sin)": lambda x, y: np.deg2rad(20+45*np.sin(np.pi*x/A)*np.sin(np.pi*y/B))}
    print(f"backbone: Omega/omega1 = 1 + beta*(a/h)^2 ; detuning at a/h=0.4 = 0.16*beta\n")
    print(f"  {'field':>26} {'model':>8} {'f1[Hz]':>7} {'gamma_eff':>12} {'beta':>9} {'det@0.4h':>9}")
    for name, fn in fields.items():
        ff, gf, kbf, bf = fe_backbone(fn)
        fm, gm, kbm, bm = modal_backbone(fn, n_modes=8)
        fm2, gm2, kbm2, bm2 = modal_backbone(fn, n_modes=24)
        print(f"  {name:>26} {'FE':>8} {ff:7.1f} {gf:12.3e} {bf:9.4f} {0.16*bf*100:8.2f}%")
        print(f"  {'':>26} {'modal-8':>8} {fm:7.1f} {gm:12.3e} {bm:9.4f} {0.16*bm*100:8.2f}%")
        print(f"  {'':>26} {'modal-24':>8} {fm2:7.1f} {gm2:12.3e} {bm2:9.4f} {0.16*bm2*100:8.2f}%")
        print(f"  {'':>26} {'FE vs modal-24 detuning error':>8} -> {100*abs(bf-bm2)/abs(bf):.0f}%\n")
