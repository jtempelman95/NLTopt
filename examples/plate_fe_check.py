r"""FE ground-truth check for the assumed-modes von Karman plate.

Hand-rolled anisotropic Kirchhoff plate FE (12-DOF ACM rectangular element; carries
the full D(theta) including the D16,D26 bend-twist couplings a STEERED laminate
creates). Self-validated against the closed-form specially-orthotropic simply-
supported frequency, then used to check the assumed-modes (Galerkin sine) model on a
HETEROGENEOUS steered plate -- the validation we never had (all prior 'ground truth'
was EPMC / time-domain of the SAME modal model)."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys; sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
from scipy.linalg import eigh
from nltopt.laminate import Qbar, lamina_Q, RHO
from nltopt.plate import VonKarmanPlate

Q0 = lamina_Q()
A, B, H = 0.20, 0.15, 0.002
CB = H**3 / 12.0

# ACM 12-term monomials in local (s,t) in [0,1]; and 2nd derivatives
def _mono(s, t):
    return np.array([1, s, t, s*s, s*t, t*t, s**3, s*s*t, s*t*t, t**3, s**3*t, s*t**3])
def _mono_s(s, t):
    return np.array([0, 1, 0, 2*s, t, 0, 3*s*s, 2*s*t, t*t, 0, 3*s*s*t, t**3])
def _mono_t(s, t):
    return np.array([0, 0, 1, 0, s, 2*t, 0, s*s, 2*s*t, 3*t*t, s**3, 3*s*t*t])
def _mono_ss(s, t):
    return np.array([0, 0, 0, 2, 0, 0, 6*s, 2*t, 0, 0, 6*s*t, 0])
def _mono_tt(s, t):
    return np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*s, 6*t, 0, 6*s*t])
def _mono_st(s, t):
    return np.array([0, 0, 0, 0, 1, 0, 0, 2*s, 2*t, 0, 3*s*s, 3*t*t])

_G, _Gw = np.polynomial.legendre.leggauss(4)
_GS = 0.5*(_G+1)                                    # gauss pts on [0,1]
_GWS = 0.5*_Gw


def _elem(lx, ly, Db):
    """ACM element bending stiffness + consistent mass for bending stiffness Db (3x3)."""
    # C: nodal DOF [w, w_y, -w_x] at 4 corners -> monomial coeffs
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    C = np.zeros((12, 12))
    for k, (s, t) in enumerate(corners):
        C[3*k+0] = _mono(s, t)
        C[3*k+1] = _mono_t(s, t) / ly              # w_y
        C[3*k+2] = -_mono_s(s, t) / lx             # -w_x
    Ci = np.linalg.inv(C)
    Ke = np.zeros((12, 12)); Me = np.zeros((12, 12))
    for i, s in enumerate(_GS):
        for j, t in enumerate(_GS):
            wq = _GWS[i]*_GWS[j]*lx*ly
            Bk = np.vstack([_mono_ss(s, t)/lx**2, _mono_tt(s, t)/ly**2,
                            2*_mono_st(s, t)/(lx*ly)])          # curvature (3x12) in monomials
            Ke += wq * (Bk.T @ Db @ Bk)
            m = _mono(s, t)[None, :]
            Me += wq * RHO*H * (m.T @ m)
    return Ci.T @ Ke @ Ci, Ci.T @ Me @ Ci


def fe_frequencies(theta_fun, nx=24, ny=18, nmodes=6):
    xs = np.linspace(0, A, nx+1); ys = np.linspace(0, B, ny+1)
    lx, ly = A/nx, B/ny
    nnode = (nx+1)*(ny+1); ndof = 3*nnode
    nid = lambda ix, iy: iy*(nx+1)+ix
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    for iy in range(ny):
        for ix in range(nx):
            xc, yc = (ix+0.5)*lx, (iy+0.5)*ly
            Db = CB * Qbar(theta_fun(xc, yc), Q0)
            ke, me = _elem(lx, ly, Db)
            nodes = [nid(ix, iy), nid(ix+1, iy), nid(ix+1, iy+1), nid(ix, iy+1)]
            dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in nodes]).ravel()
            K[np.ix_(dofs, dofs)] += ke; M[np.ix_(dofs, dofs)] += me
    # simply-supported (Navier): w=0 on all boundary nodes
    fixed = []
    for ix in range(nx+1):
        for iy in range(ny+1):
            if ix in (0, nx) or iy in (0, ny):
                fixed.append(3*nid(ix, iy))          # w = 0
    free = np.setdiff1d(np.arange(ndof), fixed)
    w2, V = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    w2 = w2[w2 > 1e-6][:nmodes]
    return np.sqrt(w2)/(2*np.pi), free, V, (nx, ny)


def ortho_closed_form(m, n):
    D11 = CB*Qbar(0.0, Q0)[0, 0]; D22 = CB*Qbar(0.0, Q0)[1, 1]
    D12 = CB*Qbar(0.0, Q0)[0, 1]; D66 = CB*Qbar(0.0, Q0)[2, 2]
    val = (D11*(m/A)**4 + 2*(D12+2*D66)*(m/A)**2*(n/B)**2 + D22*(n/B)**4)
    return np.pi**2*np.sqrt(val/(RHO*H))/(2*np.pi)


def modal_frequencies(theta_fun, n_modes=8):
    pl = VonKarmanPlate(A, B, H, n_modes=n_modes, n_in=1, nq=16)
    th_q = np.array([theta_fun(x, y) for x, y in zip(pl.qx, pl.qy)])
    pl.set_theta(th_q)
    return pl.natural_frequencies()[:6] / (2 * np.pi)          # rad/s -> Hz


if __name__ == "__main__":
    print("=== (1) FE self-validation: UD [0] plate vs closed-form orthotropic ===")
    for nx, ny in [(16, 12), (24, 18), (32, 24)]:
        f, *_ = fe_frequencies(lambda x, y: 0.0, nx, ny, 3)
        cf = np.array([ortho_closed_form(1, 1), ortho_closed_form(2, 1), ortho_closed_form(1, 2)])
        print(f"  mesh {nx}x{ny}: FE f1,f2,f3 = {np.round(f[:3],1)} Hz   closed-form {np.round(np.sort(cf),1)} Hz")
    fm = modal_frequencies(lambda x, y: 0.0)
    print(f"  assumed-modes UD: {np.round(fm[:3],1)} Hz")

    print("\n=== (2) HETEROGENEOUS STEERED plate: FE (converged) vs assumed-modes ===")
    steer = lambda x, y: np.deg2rad(20.0 + 45.0*np.sin(np.pi*x/A)*np.sin(np.pi*y/B))
    ffe_c, *_ = fe_frequencies(steer, 32, 24, 6)
    ffe_f, *_ = fe_frequencies(steer, 44, 33, 6)
    print(f"  FE mesh convergence  f1: 32x24 = {ffe_c[0]:.1f} Hz,  44x33 = {ffe_f[0]:.1f} Hz")
    print(f"  {'n_modes':>8} {'modal f1':>9} {'vs FE':>7}   {'modal f2':>9} {'modal f3':>9}")
    for nm in (8, 15, 24, 35, 48):
        fmm = modal_frequencies(steer, nm)
        print(f"  {nm:8d} {fmm[0]:8.1f}H {100*(fmm[0]-ffe_f[0])/ffe_f[0]:+5.1f}%   "
              f"{fmm[1]:8.1f}H {fmm[2]:8.1f}H")
    print(f"  FE (44x33):        {ffe_f[0]:8.1f}H          {ffe_f[1]:8.1f}H {ffe_f[2]:8.1f}H")
