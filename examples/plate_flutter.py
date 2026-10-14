r"""SUPERSONIC PANEL FLUTTER by fiber steering -- push the aeroelastic Hopf out.

A thin laminated von Karman panel in supersonic flow (first-order piston theory)
loses stability by *coalescence flutter*: as the dynamic pressure q rises two
structural modes merge and a complex-conjugate eigenvalue pair crosses into the
right half-plane (a Hopf).  Below the critical q_cr the panel is quiet; above it,
it flutters.  The whole design goal is to keep q_cr *above* the flight envelope.

Frequency and flutter read the tow-angle field through DIFFERENT operators -- the
fundamental through the bending curvature, the flutter through the streamwise aero
coupling G = \int phi_i d(phi_j)/dx -- so at a REQUIRED operating frequency f1 a
steered field beats the best uniform layup.  We drive the design by q_cr's analytic
adjoint (implicit eigenvalue sensitivity on the nonsymmetric flutter pencil) under
the tow-curvature (curl) manufacturability constraint.

Verifications built in:
  * 1-D calibration -> classical Dowell critical dynamic pressure lambda_cr ~ 343.36
  * analytic dq_cr/dtheta vs central finite differences  (< 1e-4)

Result on a 200x150x2 mm CFRP panel (Mach 2): a curl-feasible steered field gives a
1.91x flutter dynamic pressure at held f1 = 330 Hz and held mass vs the best uniform
angle (4057 vs 2121 kPa), exceeding even the flow-aligned [0] panel.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_flutter.py
"""
import os
import sys
sys.path.insert(0, "/home/jrt/wavetopo")

import numpy as np
from scipy.linalg import eig, eigh
from concurrent.futures import ProcessPoolExecutor

from wavetopo.cfrp import CSRBFMapping
from wavetopo.cfrp_optimizer import MMA
from nltopt.plate import VonKarmanPlate
from nltopt.laminate import lamina_Q
import plate_framework_prototype as P            # mesh, RBF (Phi/Phi_x/Phi_y), curl_jac

MACH = 2.0
BETA = 2.0 / np.sqrt(MACH**2 - 1.0)              # 2/sqrt(M^2-1); q multiplies this
C0 = 30.0                                        # light fixed mass-prop. damping (regularizes the Hopf)
A, B, H = P.A, P.B, P.H
Q0 = lamina_Q()


# --------------------------------------------------------------------------- #
#  1-D calibration: classical simply-supported panel, piston theory
# --------------------------------------------------------------------------- #
def calibration_lambda(N=16):
    """a=1, D=1, rho*h=1, no aero damping (Dowell coalescence flutter).
    Aero coupling G[i,j] = 2ij/(i^2-j^2) for i+j odd.  Lambda_cr -> ~343.36."""
    m = np.arange(1, N + 1)
    Kb = np.diag((m * np.pi) ** 4 / 2.0)
    Mm = np.diag(np.full(N, 0.5))
    G = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if (i + j) % 2 == 1:
                G[i - 1, j - 1] = 2.0 * i * j / (i * i - j * j)
    flut = lambda L: np.max(np.abs(eig(Kb + L * G, Mm, right=False).imag)) > 1e-6
    lo, hi = 0.0, 800.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if flut(mid) else (hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  laminate-angle derivative of the reduced stiffness (analytic)
# --------------------------------------------------------------------------- #
def dQbar_dtheta(theta, Q=Q0):
    m, n = np.cos(theta), np.sin(theta)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    dm2n2 = 2 * m * n * (m2 - n2)
    dm4, dn4 = -4 * m2 * m * n, 4 * n2 * n * m
    dm4n4 = dm4 + dn4
    d = np.zeros((3, 3))
    d[0, 0] = Q11 * dm4 + 2 * (Q12 + 2 * Q66) * dm2n2 + Q22 * dn4
    d[1, 1] = Q11 * dn4 + 2 * (Q12 + 2 * Q66) * dm2n2 + Q22 * dm4
    d[0, 1] = d[1, 0] = (Q11 + Q22 - 4 * Q66) * dm2n2 + Q12 * dm4n4
    d[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * dm2n2 + Q66 * dm4n4
    dm3n, dmn3 = m4 - 3 * m2 * n2, 3 * m2 * n2 - n4
    d[0, 2] = d[2, 0] = (Q11 - Q12 - 2 * Q66) * dm3n + (Q12 - Q22 + 2 * Q66) * dmn3
    d[1, 2] = d[2, 1] = (Q11 - Q12 - 2 * Q66) * dmn3 + (Q12 - Q22 + 2 * Q66) * dm3n
    return d


# --------------------------------------------------------------------------- #
#  plate + aero + flutter eigenproblem
# --------------------------------------------------------------------------- #
def plate_Phi(pl):
    class DM: pass
    dm = DM(); dm.cx = pl.qx; dm.cy = pl.qy; dm.N = pl.qx.size
    return CSRBFMapping(dm, (P.rbf.sx, P.rbf.sy), r_s=P.rbf.r_s).Phi


def build(theta_hat, nmodes=8):
    pl = VonKarmanPlate(A, B, H, n_modes=nmodes, n_in=1, nq=12)
    pl._Phi = plate_Phi(pl)
    pl.set_theta(pl._Phi @ theta_hat)
    return pl


def aero_G(pl):
    r"""Piston-theory coupling on the transverse block: nonsymmetric \int phi_i phi_j,x."""
    return np.einsum("q,iq,jq->ij", pl.qw, pl.PHI, pl.PHIx)


def _pencil(pl, G, q):
    sw = pl._sw
    Kb = np.array(pl.K[sw, sw], float); Mw = np.array(pl.M[sw, sw], float)
    nw = Kb.shape[0]; Z, I = np.zeros((nw, nw)), np.eye(nw)
    Amat = np.block([[Z, I], [-(Kb + BETA * q * G), -C0 * Mw]])
    Bmat = np.block([[I, Z], [Z, Mw]])
    return Amat, Bmat, nw


def max_re(pl, G, q):
    Amat, Bmat, _ = _pencil(pl, G, q)
    return eig(Amat, Bmat, right=False).real.max()


def qcr(pl, G):
    """Critical dynamic pressure: smallest q with max Re(s) > 0 (bracket then bisect)."""
    hi = 1e3
    while max_re(pl, G, hi) < 0:
        hi *= 2.0
        if hi > 1e15:
            return np.nan
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if max_re(pl, G, mid) > 0 else (mid, hi)
    return 0.5 * (lo + hi)


def _dQ_scaled(pl):
    return np.stack([pl.qw[q] * pl.dens_E[q] * pl.CB * dQbar_dtheta(pl.theta_q[q])
                     for q in range(pl.Nq)])


def qcr_grad(pl, G, qc):
    """Analytic dq_cr/dtheta_hat by implicit eigenvalue sensitivity of the Hopf:
    Re s(theta, q_cr) = 0  =>  dq_cr/dtheta = -Re(ds/dtheta)/Re(ds/dq)."""
    Amat, Bmat, nw = _pencil(pl, G, qc)
    w, vl, vr = eig(Amat, Bmat, left=True, right=True)
    c = int(np.argmax(w.real))
    u, v = vl[:, c], vr[:, c]
    ub, vt = u[nw:], v[:nw]
    cB = np.vdot(u, Bmat @ v)
    kU = np.einsum("i,iaq->aq", np.conjugate(ub), pl.KAP)
    kV = np.einsum("j,jbq->bq", vt, pl.KAP)
    gfield = np.einsum("aq,qab,bq->q", kU, _dQ_scaled(pl), kV)
    grad_hat = pl._Phi.T @ gfield
    dsdtheta = -grad_hat / cB
    dsdq = -BETA * np.vdot(ub, G @ vt) / cB
    return -(dsdtheta.real) / (dsdq.real)


def w1sq_and_grad(pl):
    """Fundamental omega_1^2 and its analytic gradient (Fox-Kapoor / curvature field)."""
    sw = pl._sw
    Kb = np.array(pl.K[sw, sw], float); Mw = np.array(pl.M[sw, sw], float)
    vals, vecs = eigh(Kb, Mw)
    phi1 = vecs[:, 0]
    kU = np.einsum("i,iaq->aq", phi1, pl.KAP)
    gfield = np.einsum("aq,qab,bq->q", kU, _dQ_scaled(pl), kU)
    return vals[0], pl._Phi.T @ gfield


def f1_of(pl):
    return np.sqrt(w1sq_and_grad(pl)[0]) / (2 * np.pi)


def uniform_pareto(degs):
    out = []
    for d in degs:
        pl = build(np.full(P.M, np.deg2rad(d)))
        out.append((d, f1_of(pl), qcr(pl, aero_G(pl))))
    return out


# --------------------------------------------------------------------------- #
#  design: maximize q_cr at held f1, curl-feasible  (MMA + augmented Lagrangian)
# --------------------------------------------------------------------------- #
def optimize(w1sq_tgt, zeta_all, seed_deg, iters=90):
    th = np.full(P.M, np.deg2rad(seed_deg))
    q0 = qcr(build(th), aero_G(build(th)))
    lam = np.full(P.N, 15.0); lam_f = 0.0; mu = 5.0
    mma = MMA(np.full(P.M, -np.pi / 2), np.full(P.M, np.pi / 2))
    best = None
    for _ in range(iters):
        for _ in range(5):
            pl = build(th); G = aero_G(pl); qc = qcr(pl, G)
            dq = qcr_grad(pl, G, qc)
            w1sq, dw1 = w1sq_and_grad(pl)
            gth = -(dq / q0)                                   # minimize -q_cr/q0
            zeta = np.cos(P.Phi @ th) * (P.Phi_x @ th) + np.sin(P.Phi @ th) * (P.Phi_y @ th)
            gc = zeta ** 2 - zeta_all ** 2
            fac = (lam + mu * np.maximum(gc, -lam / mu)) * (gc > -lam / mu)
            gth += (1.0 / P.N) * (P.curl_jac(th, P.Phi @ th).T @ (2 * zeta * fac))
            h = w1sq / w1sq_tgt - 1.0                          # frequency equality AL
            gth += (lam_f + mu * h) * (dw1 / w1sq_tgt)
            th = np.clip(mma.update(th, gth), -np.pi / 2, np.pi / 2)
        pl = build(th); qc = qcr(pl, aero_G(pl)); w1sq = w1sq_and_grad(pl)[0]
        zeta = np.cos(P.Phi @ th) * (P.Phi_x @ th) + np.sin(P.Phi @ th) * (P.Phi_y @ th)
        zmax = float(np.max(np.abs(zeta))); h = w1sq / w1sq_tgt - 1.0
        if (zmax / zeta_all < 1.01) and (abs(h) < 3e-3) and (best is None or qc > best[0]):
            best = (qc, th.copy(), zmax, np.sqrt(w1sq) / (2 * np.pi))
        gc = zeta ** 2 - zeta_all ** 2
        lam = np.maximum(lam + mu * np.maximum(gc, -lam / mu), 0.0)
        lam_f = lam_f + mu * h
        if best is None or (zmax / zeta_all >= 1.01) or abs(h) >= 3e-3:
            mu = min(1.25 * mu, 1e7)
    return best


def _opt_task(args):
    return optimize(*args)


def _fd_check():
    rng = np.random.default_rng(0)
    th = np.deg2rad(20.0) * np.ones(P.M) + np.deg2rad(8.0) * rng.standard_normal(P.M)
    pl = build(th); G = aero_G(pl); qc = qcr(pl, G); g = qcr_grad(pl, G, qc)
    h = 1e-4; err = 0.0
    for k in rng.choice(P.M, size=6, replace=False):
        e = np.zeros(P.M); e[k] = h
        qp = qcr(build(th + e), aero_G(build(th + e)))
        qm = qcr(build(th - e), aero_G(build(th - e)))
        err = max(err, abs(g[k] - (qp - qm) / (2 * h)) / (abs((qp - qm) / (2 * h)) + 1e-30))
    return err


def main():
    lam = calibration_lambda()
    print(f"[calibration] 1-D piston-theory lambda_cr = {lam:.2f}  "
          f"(Dowell ~343.36, err {abs(lam-343.36)/343.36:.1e})")
    print(f"[adjoint FD-check] max rel err = {_fd_check():.2e}  (must be < 1e-4)\n")

    print(f"Panel {A*1e3:.0f}x{B*1e3:.0f}x{H*1e3:.0f} mm, flow // x, Mach {MACH}, "
          f"{build(np.zeros(P.M)).nw} modes, {P.M} design vars")
    print("uniform-angle front (deg, f1 Hz, q_cr kPa):")
    fine = uniform_pareto(range(0, 91, 2))
    for d, f, q in fine[::5]:
        print(f"  {d:3d}   {f:6.1f}   {q/1e3:8.1f}")
    ff = np.array([p[1] for p in fine]); fq = np.array([p[2] for p in fine])

    targets_deg = [30, 40, 50]
    zeta_all = 12.0
    tasks = []
    for td in targets_deg:
        w1sq_t = w1sq_and_grad(build(np.full(P.M, np.deg2rad(td))))[0]
        for seed in (0, 15, 30, 45, 60):
            tasks.append((w1sq_t, zeta_all, seed))
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as ex:
        res = list(ex.map(_opt_task, tasks))

    print("\n  target f1   best uniform   STEERED q_cr    gain   max|curl|")
    per = {}
    for (w1sq_t, _, _), r in zip(tasks, res):
        if r is None:
            continue
        key = round(np.sqrt(w1sq_t) / (2 * np.pi), 1)
        if key not in per or r[0] > per[key][0]:
            per[key] = r
    for td in targets_deg:
        f1t = f1_of(build(np.full(P.M, np.deg2rad(td))))
        bu = fq[np.abs(ff - f1t) < 6.0].max()
        if round(f1t, 1) in per:
            qc, th, zmax, _ = per[round(f1t, 1)]
            print(f"  {f1t:7.1f}Hz {bu/1e3:11.0f}k {qc/1e3:11.0f}k  {qc/bu:5.2f}x  {zmax:8.2f}")
            np.save(os.path.join(os.path.dirname(__file__), "..", "figs",
                                 f"flutter_steered_{td}.npy"), th)


if __name__ == "__main__":
    main()
