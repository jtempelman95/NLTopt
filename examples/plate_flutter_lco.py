r"""Post-flutter LIMIT CYCLE of the steered panel, by direct time integration of the
import os
self-excited nonlinear aeroelastic plate (von Karman + piston-theory aero, in-plane
statically condensed).  Above q_cr the flat panel is unstable; the geometric
nonlinearity saturates the growth into a bounded limit-cycle oscillation (LCO)."""
import sys; sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
from scipy.integrate import solve_ivp
from nltopt.plate import VonKarmanPlate
import plate_flutter as F

A, B, H = F.A, F.B, F.H
PHIC = None


def build_nl(theta_hat, nmodes=8, n_in=3):
    pl = VonKarmanPlate(A, B, H, n_modes=nmodes, n_in=n_in, nq=12)
    pl.set_theta(F.plate_Phi(pl) @ theta_hat)
    return pl


def relaxed_cubic_tensor(pl):
    """Extract the pure-cubic relaxed transverse force f(q)=T3(q,q,q) as a tensor,
    by symmetric polarization (the flat B=0 plate has no transverse quadratic force)."""
    sw = pl._sw; si = slice(pl.nw, pl.ndof); n = pl.nw
    Kin = np.array(pl.K[si, si]); force = pl._force; qfull = np.zeros(pl.ndof)

    def frel(qw):
        qfull[:] = 0.0; qfull[sw] = qw
        drive = force._internal(qfull)[0][si]
        qfull[si] = -np.linalg.solve(Kin, drive)
        return force._internal(qfull)[0][sw]

    E = np.eye(n); fe = [frel(E[j]) for j in range(n)]
    T = np.zeros((n, n, n, n))
    from itertools import permutations
    for j in range(n):
        for k in range(j, n):
            for l in range(k, n):
                v = (frel(E[j]+E[k]+E[l]) - frel(E[j]+E[k]) - frel(E[j]+E[l])
                     - frel(E[k]+E[l]) + fe[j]+fe[k]+fe[l]) / 6.0
                for (a, b, c) in set(permutations((j, k, l))):
                    T[:, a, b, c] = v
    return T


def lco(pl, G, q, T3, C0=F.C0, periods=500, ic=0.03, y0=None):
    sw = pl._sw
    Mw = np.array(pl.M[sw, sw]); Kw = np.array(pl.K[sw, sw]); Minv = np.linalg.inv(Mw)
    Cw = C0 * Mw; Kae = Kw + F.BETA * q * G
    cen = np.array([np.sin(m*np.pi*0.5)*np.sin(n*np.pi*0.5) for (m, n) in pl.w_modes])
    w1 = np.sqrt(np.linalg.eigvalsh(np.linalg.solve(Mw, Kw)).min())

    def rhs(t, yy):
        qw, vw = yy[:pl.nw], yy[pl.nw:]
        fnl = np.einsum('ijkl,j,k,l->i', T3, qw, qw, qw)
        return np.concatenate([vw, -Minv @ (Cw @ vw + Kae @ qw + fnl)])

    Tp = 2*np.pi/w1
    if y0 is None:
        y0 = np.zeros(2*pl.nw); y0[:pl.nw] = ic * H * np.ones(pl.nw)/np.linalg.norm(np.ones(pl.nw))
    te = np.linspace((periods-100)*Tp, periods*Tp, 100*36)
    sol = solve_ivp(rhs, (0, periods*Tp), y0, t_eval=te, method="RK45", rtol=1e-7, atol=1e-10)
    wc = sol.y[:pl.nw].T @ cen
    tail = wc[len(wc)//2:]
    return np.max(np.abs(tail))/H, w1/(2*np.pi), sol.y[:, -1]


if __name__ == "__main__":
    import time
    plu = build_nl(np.full(F.P.M, np.deg2rad(30))); Gu = F.aero_G(plu); qcu = F.qcr(plu, Gu)
    t0 = time.time(); T3 = relaxed_cubic_tensor(plu)
    # verify the extracted cubic tensor reproduces the condensed force
    rng = np.random.default_rng(0); x = 0.3*H*rng.standard_normal(plu.nw)
    sw = plu._sw; si = slice(plu.nw, plu.ndof); Kin = plu.K[si, si]; qf = np.zeros(plu.ndof)
    qf[sw] = x; drv = plu._force._internal(qf)[0][si]; qf[si] = -np.linalg.solve(Kin, drv)
    fexact = plu._force._internal(qf)[0][sw]
    ffast = np.einsum('ijkl,j,k,l->i', T3, x, x, x)
    print(f"tensor extract in {time.time()-t0:.1f}s; f_fast vs f_exact rel err = "
          f"{np.linalg.norm(ffast-fexact)/np.linalg.norm(fexact):.1e}")
    print(f"uniform 30deg: q_cr = {qcu/1e3:.0f} kPa\n")
    print("  time-domain LCO of the uniform panel vs dynamic pressure:")
    for fac in (0.9, 1.1, 1.3):
        t0 = time.time(); a, f1, _ = lco(plu, Gu, fac*qcu, T3)
        print(f"    q = {fac:.2f} q_cr = {fac*qcu/1e3:6.0f} kPa:  LCO a/h = {a:.4f}  "
              f"({'STABLE' if a < 0.02 else 'FLUTTER (LCO)'})  [{time.time()-t0:.0f}s]")
