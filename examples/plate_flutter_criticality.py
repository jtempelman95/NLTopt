r"""Stuart-Landau LCO amplitude from the SSM/normal-form coefficient of the flutter
mode -- the cheap analytical design surrogate -- checked vs the time-domain LCO."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys; sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
from scipy.linalg import eig
import plate_flutter as F
import plate_flutter_lco as L

pl = L.build_nl(np.full(F.P.M, np.deg2rad(30)))
G = F.aero_G(pl); qcr = F.qcr(pl, G); T3 = L.relaxed_cubic_tensor(pl)
sw = pl._sw; nw = pl.nw
Mw = np.array(pl.M[sw, sw]); Kw = np.array(pl.K[sw, sw]); Cw = F.C0 * Mw
cen = np.array([np.sin(m*np.pi*0.5)*np.sin(n*np.pi*0.5) for (m, n) in pl.w_modes])
Z = np.zeros((nw, nw))


def pencil(q):                                          # SSM first-order form: A v = lam B v
    Kae = Kw + F.BETA * q * G
    A = np.block([[-Kae, Z], [Z, Mw]]); B = np.block([[Cw, Mw], [Mw, Z]])
    return A, B


# flutter eigenpair at q_cr (critical mode: max Re, upper half-plane)
A, B = pencil(qcr)
lr, Vr = eig(A, B); ll, Vl = eig(A.T, B.T)
up = lr.imag > 1.0; idx = np.where(up)[0]; i1 = idx[np.argmax(lr[idx].real)]
lam1, v1 = lr[i1], Vr[:, i1]
j1 = int(np.argmin(np.abs(ll - np.conjugate(lam1)))); u1 = Vl[:, j1]   # left eigvec for conj(lam1), per nltopt
v1 = v1 / (cen @ v1[:nw])                               # unit centre deflection per unit p
u1 = u1 / np.conjugate(np.vdot(u1, B @ v1))             # u1* B v1 = 1
print(f"flutter mode: f_f = {lam1.imag/(2*np.pi):.0f} Hz, Re(lam1)@qcr = {lam1.real:.2e}")

# Stuart-Landau cubic: gamma = u1* G_(2,1),  G_(2,1)_disp = -3 T3(v1,v1,conj v1)
v1n = v1[:nw]
g21 = -3.0 * np.einsum('ijkl,j,k,l->i', T3, v1n, v1n, np.conjugate(v1n))
gamma = np.vdot(u1, np.concatenate([g21, np.zeros(nw)]))
print(f"Stuart-Landau gamma = {gamma:.3e}   Re(gamma) {'<0 SUPERcritical' if gamma.real<0 else '>0 SUBcritical'}")


def lam_at(q):
    lr = eig(*pencil(q), right=False)
    return lr[np.argmin(np.abs(lr - lam1))]


print(f"\n  {'q/qcr':>7} {'SSM a/h':>9} {'time-domain a/h':>16}")
for fac, atd in [(1.10, 0.214), (1.30, 0.504)]:
    al = lam_at(fac*qcr).real
    p2 = -al / gamma.real                               # |p|^2 = -Re(lam)/Re(gamma)
    a_h = 2*np.sqrt(max(p2, 0.0)) / F.H                 # a = 2|p|*(cen.v1n=1); a/h
    print(f"  {fac:7.2f} {a_h:9.3f} {atd:16.3f}")
