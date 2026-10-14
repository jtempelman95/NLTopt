r"""Conforming Kirchhoff plate FE: 16-DOF Bogner-Fox-Schmit (BFS) bicubic-Hermite
rectangular element (C1-continuous, nodal DOF w, w_x, w_y, w_xy). Carries full
anisotropic D(theta) (D16,D26) and the piston-theory aero. Self-validated against the
closed-form frequency, then used for the definitive coupled-case flutter boundary that
the non-conforming ACM element got wrong."""
import sys; sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
from scipy.linalg import eigh, eig
from nltopt.laminate import Qbar, RHO
import plate_fe_check as CH        # A,B,H,CB,Q0 and orthotropic closed form
import plate_flutter as PF

A, B, H, CB, Q0 = CH.A, CH.B, CH.H, CH.CB, CH.Q0
BETA = 2.0/np.sqrt(PF.MACH**2 - 1.0)
_G, _Gw = np.polynomial.legendre.leggauss(4)
_S = 0.5*(_G+1); _W = 0.5*_Gw            # gauss nodes/weights on [0,1]


def _herm(s, L):
    """1-D cubic Hermite [H0val,H0slope,H1val,H1slope] and 1st,2nd x-derivatives."""
    v = np.array([1-3*s**2+2*s**3, L*(s-2*s**2+s**3), 3*s**2-2*s**3, L*(-s**2+s**3)])
    d1 = np.array([(-6*s+6*s**2)/L, 1-4*s+3*s**2, (6*s-6*s**2)/L, -2*s+3*s**2])
    d2 = np.array([(-6+12*s)/L**2, (-4+6*s)/L, (6-12*s)/L**2, (-2+6*s)/L])
    return v, d1, d2


# 16 DOF: node order (0,0),(1,0),(1,1),(0,1); per node (w,w_x,w_y,w_xy)
_PAIRS = []
for (ix, iy) in [(0, 0), (1, 0), (1, 1), (0, 1)]:
    _PAIRS += [(2*ix, 2*iy), (2*ix+1, 2*iy), (2*ix, 2*iy+1), (2*ix+1, 2*iy+1)]


def _elem(lx, ly, D):
    Ke = np.zeros((16, 16)); Me = np.zeros((16, 16)); Ge = np.zeros((16, 16))
    for i, s in enumerate(_S):
        vx, d1x, d2x = _herm(s, lx)
        for j, t in enumerate(_S):
            vy, d1y, d2y = _herm(t, ly)
            wq = _W[i]*_W[j]*lx*ly
            Nw = np.array([vx[a]*vy[b] for (a, b) in _PAIRS])
            Nwx = np.array([d1x[a]*vy[b] for (a, b) in _PAIRS])
            Bk = np.array([[d2x[a]*vy[b], vx[a]*d2y[b], 2*d1x[a]*d1y[b]] for (a, b) in _PAIRS]).T
            Ke += wq*(Bk.T @ D @ Bk)
            Me += wq*RHO*H*np.outer(Nw, Nw)
            Ge += wq*np.outer(Nw, Nwx)
    return Ke, Me, Ge


def assemble(theta_e, nx, ny):
    lx, ly = A/nx, B/ny; nnode = (nx+1)*(ny+1)
    nid = lambda ix, iy: iy*(nx+1)+ix
    K = np.zeros((4*nnode, 4*nnode)); M = np.zeros_like(K); G = np.zeros_like(K)
    e = 0
    for iy in range(ny):
        for ix in range(nx):
            ke, me, ge = _elem(lx, ly, CB*Qbar(theta_e[e], Q0))
            nodes = [nid(ix, iy), nid(ix+1, iy), nid(ix+1, iy+1), nid(ix, iy+1)]
            dofs = np.array([[4*n, 4*n+1, 4*n+2, 4*n+3] for n in nodes]).ravel()
            K[np.ix_(dofs, dofs)] += ke; M[np.ix_(dofs, dofs)] += me; G[np.ix_(dofs, dofs)] += ge
            e += 1
    # simply-supported (Navier): w=0 all boundary; tangential slope & cross-deriv on edges
    fix = []
    for ix in range(nx+1):
        for iy in range(ny+1):
            n = nid(ix, iy)
            if ix in (0, nx) or iy in (0, ny):
                fix.append(4*n)                                  # w=0 (Navier SS)
                if ix in (0, nx): fix.append(4*n+2)              # tangential slope w_y on vertical edges
                if iy in (0, ny): fix.append(4*n+1)              # tangential slope w_x on horizontal edges
                # w_xy left free everywhere (nonzero for the sine modes; a natural DOF)
    free = np.setdiff1d(np.arange(4*nnode), np.unique(fix))
    return K[np.ix_(free, free)], M[np.ix_(free, free)], G[np.ix_(free, free)]


def bfs_freqs(theta_e, nx, ny, k=3):
    K, M, _ = assemble(theta_e, nx, ny)
    w2 = eigh(K, M, eigvals_only=True)
    return np.sqrt(np.abs(w2[:k]))/(2*np.pi)


def bfs_qcr(theta_e, nx, ny, nr=30, C0=PF.C0):
    K, M, G = assemble(theta_e, nx, ny)
    w2, Phi = eigh(K, M); Phi = Phi[:, :nr]
    Kr = Phi.T @ K @ Phi; Gr = Phi.T @ G @ Phi; Cr = C0*np.eye(nr)
    Z, I = np.zeros((nr, nr)), np.eye(nr)
    mre = lambda q: eig(np.block([[Z, I], [-(Kr+BETA*q*Gr), -Cr]]),
                        np.block([[I, Z], [Z, I]]), right=False).real.max()
    hi = 1e3
    while mre(hi) < 0:
        hi *= 2
    lo = 0.0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        if mre(mid) > 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi), np.sqrt(w2[0])/(2*np.pi)


if __name__ == "__main__":
    print("BFS conforming element -- self-validation (UD [0], vs closed-form orthotropic):")
    for nx, ny in [(8, 6), (12, 9), (16, 12)]:
        f = bfs_freqs(np.zeros(nx*ny), nx, ny, 3)
        cf = np.sort([CH.ortho_closed_form(1, 1), CH.ortho_closed_form(2, 1), CH.ortho_closed_form(1, 2)])
        print(f"  mesh {nx}x{ny}: f1,f2,f3 = {np.round(f,1)} Hz   closed-form {np.round(cf,1)}")
