r"""Mode-ADAPTIVE reduced model: the master transverse mode is the fundamental of
the multi-mode Ritz eigenproblem, so it reshapes with the design -- and the linear
frequency it reports is the REAL multi-mode frequency, not a fixed-mode proxy.

This closes the last caveat of the two-field study: the earlier single-fixed-mode
frequency mis-tracked strongly-steered designs (a design that reads "held" on the
(1,1) mode was really 1.7x stiffer).  Here the frequency is constrained on the
adapted mode, so the shipped device actually sits at its mandated frequency.

Objective  J = gamma_eff / omega1^2  (relative amplitude detuning),
both evaluated on the adapted mode  w = sum_k v_k phi_k,  (v, omega1) the
fundamental Ritz pair of  K_b(theta) v = omega1^2 M v.

The gradient is analytic and FD-verified:
  * the frequency (eigenvalue) sensitivity is Fox-Kapoor, no eigenvector derivative;
  * the eigenvector part of gamma_eff collapses to a single ADJOINT CURVATURE field
    kappa_beta = sum_{j!=0} beta_j kappa_{v_j}.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_ritz.py
"""
import sys
sys.path.insert(0, "/home/jrt/wavetopo")
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import paperstyle as ps
import numpy as np
from scipy.linalg import eigh
import plate_framework_prototype as P
from plate_topopt import BL                                   # in-plane strain-displacement (3,ndof,N)
from nltopt.laminate import Qbar, lamina_Q, RHO
from wavetopo.cfrp_optimizer import MMA

A, B, H = P.A, P.B, P.H
cx, cy, area, N, M = P.cx, P.cy, P.area, P.N, P.M
CB, CM, Phi, Q0 = P.CB, P.CM, P.Phi, P.Q0

# ---- multi-mode transverse basis (quarter-symmetric: couples to the (1,1)) --
TMODES = [(1, 1), (1, 3), (3, 1), (3, 3)]
nT = len(TMODES)
PHItx = np.empty((nT, N)); PHIty = np.empty((nT, N)); KAPt = np.empty((nT, 3, N)); PHIt = np.empty((nT, N))
for k, (m, n) in enumerate(TMODES):
    kx, ky = m*np.pi/A, n*np.pi/B
    sx, sy, cxx, cyy = np.sin(kx*cx), np.sin(ky*cy), np.cos(kx*cx), np.cos(ky*cy)
    PHIt[k] = sx*sy; PHItx[k] = kx*cxx*sy; PHIty[k] = ky*sx*cyy
    KAPt[k, 0] = kx**2*sx*sy; KAPt[k, 1] = ky**2*sx*sy; KAPt[k, 2] = -2*kx*ky*cxx*cyy
Mmass = RHO*H*np.einsum("e,ie,je->ij", area, PHIt, PHIt)      # modal mass (theta-independent)
PHIC = np.array([np.sin(m*np.pi/2)*np.sin(n*np.pi/2) for (m, n) in TMODES])  # mode midspan values


def _perelem(theta):
    Qb = np.stack([Qbar(theta[e], Q0) for e in range(N)])     # (N,3,3)
    return Qb, CM*Qb                                          # Qb, membrane A(theta)


def forward(theta_hat):
    theta = Phi @ theta_hat
    Qb, Am = _perelem(theta)
    Kbmat = np.einsum("e,iae,eab,jbe->ij", area, KAPt, CB*Qb, KAPt)   # (nT,nT)
    w2, V = eigh(Kbmat, Mmass)                                # ascending, M-normalized
    v, om2 = V[:, 0], w2[0]
    wx = v @ PHItx; wy = v @ PHIty                            # adapted mode slopes
    MMBa = np.vstack([wx**2, wy**2, 2*wx*wy])                 # (grad w)^2 (3,N)
    KAPv = np.einsum("k,kae->ae", v, KAPt)                    # adapted curvature (3,N)
    Kin = np.einsum("e,aie,eab,bje->ij", area, BL, Am, BL)
    g   = 0.5*np.einsum("e,aie,eab,be->i", area, BL, Am, MMBa)
    knl = 0.5*np.einsum("e,ae,eab,be->", area, MMBa, Am, MMBa)
    ustar = np.linalg.solve(Kin, g)
    eps_u = np.einsum("aie,i->ae", BL, ustar)
    gamma = knl - 2.0*(g @ ustar)
    mid = v @ PHIC                                            # adapted-mode midspan value
    Jphys = gamma / (om2 * mid**2)                            # PHYSICAL detuning (per midspan a)
    return dict(J=gamma/om2, Jphys=Jphys, gamma=gamma, om2=om2, w1=np.sqrt(om2), mid=mid,
                theta=theta, Qb=Qb, Am=Am, w2=w2, V=V, v=v, wx=wx, wy=wy, MMBa=MMBa,
                KAPv=KAPv, eps_u=eps_u, ustar=ustar)


def grad(st, d=1e-4):
    """Analytic dJphys/dtheta_hat and dw1/dtheta_hat on the adapted mode.  Two
    eigenvector-sensitivity terms each collapse to an adjoint curvature field:
    kappa_beta for gamma_eff, kappa_mid for the mode's midspan value."""
    Am, om2, gamma, mid = st["Am"], st["om2"], st["gamma"], st["mid"]
    MMBa, KAPv, eps_u, theta = st["MMBa"], st["KAPv"], st["eps_u"], st["theta"]
    w2, V, wx, wy = st["w2"], st["V"], st["wx"], st["wy"]

    F = area[:, None] * np.einsum("eab,eb->ea", Am, (MMBa - 2.0*eps_u).T)   # dgamma/dMMBa
    dgamma_dv = np.empty(nT)
    for k in range(nT):
        dM = np.vstack([2*wx*PHItx[k], 2*wy*PHIty[k], 2*(wx*PHIty[k] + wy*PHItx[k])])
        dgamma_dv[k] = np.sum(F.T * dM)
    inv = 1.0 / (om2 - w2[1:])                                # 1/(w0^2 - wj^2), j>=1
    KAPb   = np.einsum("k,kae->ae", V[:, 1:] @ ((dgamma_dv @ V[:, 1:]) * inv), KAPt)   # gamma adjoint
    KAPmid = np.einsum("k,kae->ae", V[:, 1:] @ ((PHIC @ V[:, 1:]) * inv), KAPt)        # midspan adjoint

    dgam = np.empty(N); dom2 = np.empty(N); dmid = np.empty(N)
    for e in range(N):
        dQ = (Qbar(theta[e] + d, Q0) - Qbar(theta[e] - d, Q0)) / (2*d)
        dA, dD = CM*dQ, CB*dQ
        m, u, kv = MMBa[:, e], eps_u[:, e], KAPv[:, e]
        dgam[e] = area[e] * (0.5*(m @ dA @ m) - 2.0*(u @ dA @ m) + 2.0*(u @ dA @ u)
                             + KAPb[:, e] @ dD @ kv)
        dom2[e] = area[e] * (kv @ dD @ kv)
        dmid[e] = area[e] * (KAPmid[:, e] @ dD @ kv)
    # Jphys = gamma / (om2 * mid^2)
    dJp = dgam/(om2*mid**2) - gamma*dom2/(om2**2*mid**2) - 2*gamma*dmid/(om2*mid**3)
    dJphys = Phi.T @ dJp
    dw1 = Phi.T @ (dom2 / (2*np.sqrt(om2)))
    return dJphys, dw1


def _fd_checks():
    rng = np.random.default_rng(3)
    th = np.deg2rad(15.0)*np.ones(M) + np.deg2rad(10)*rng.standard_normal(M)
    st = forward(th); gJ, gw = grad(st)
    eJ = ew = 0.0
    for i in range(8):
        tp = th.copy(); tp[i] += 1e-6; tm = th.copy(); tm[i] -= 1e-6
        fJ = (forward(tp)["Jphys"] - forward(tm)["Jphys"]) / 2e-6
        fw = (forward(tp)["w1"] - forward(tm)["w1"]) / 2e-6
        eJ = max(eJ, abs(gJ[i]-fJ)/(abs(fJ)+1e-30))
        ew = max(ew, abs(gw[i]-fw)/(abs(fw)+1e-30))
    print(f"[FD] dJ/dtheta   max rel err = {eJ:.2e}   (must be < 1e-4)")
    print(f"[FD] dw1/dtheta  max rel err = {ew:.2e}   (must be < 1e-4)")
    print(f"     adapted-mode f1 = {st['w1']/2/np.pi:.1f} Hz  (fundamental of the Ritz basis)")


def backbone_hz(st, r):
    """Absolute backbone frequency [Hz] at amplitude a/h = r (cubic Duffing form,
    A = r*H the modal amplitude)."""
    f1 = st["w1"] / (2*np.pi)
    return f1 * (1.0 + 0.375 * st["Jphys"] * (r*H)**2)


def optimize_points(points, R_mm=25.0, iters=200, verbose=True):
    """Steer the tow field so the REAL (adapted-mode) backbone threads prescribed
    design points ``points = [(f_Hz, a/h), ...]`` in the frequency-amplitude
    plane, subject to the curl (manufacturability) limit."""
    zeta_all = 1.0 / (R_mm / 1000.0)
    th = np.deg2rad(25.0) * np.ones(M)
    lam = np.full(P.N, 10.0); mu = 5.0
    mma = MMA(np.full(M, -np.pi/2), np.full(M, np.pi/2))
    best = None
    for k in range(iters):
        for _ in range(5):
            st = forward(th); dJp, dw1 = grad(st)
            f1 = st["w1"] / (2*np.pi); df1 = dw1 / (2*np.pi)
            gth = np.zeros(M)
            for ftar, r in points:                            # least-squares to the points
                c = 0.375 * (r*H)**2
                fm = f1 * (1.0 + c*st["Jphys"])
                dfm = df1 * (1.0 + c*st["Jphys"]) + f1 * c * dJp
                gth += 2.0 * (fm - ftar) * dfm
            zeta, theta = P.curl(th); g = zeta**2 - zeta_all**2
            fac = (lam + mu*np.maximum(g, -lam/mu)) * (g > -lam/mu)
            gth += (1.0/P.N) * (P.curl_jac(th, theta).T @ (2*zeta*fac))
            th = np.clip(mma.update(th, gth), -np.pi/2, np.pi/2)
        st = forward(th); zeta, _ = P.curl(th); zmax = float(np.abs(zeta).max())
        err = np.sqrt(sum((backbone_hz(st, r) - ftar)**2 for ftar, r in points))
        feas = zmax/zeta_all < 1.01
        if feas and (best is None or err < best[0]):
            best = (err, th.copy(), zmax)
        lam = np.maximum(lam + mu*np.maximum(g, -lam/mu), 0.0)
        if not feas:
            mu = min(1.25*mu, 1e7)
        if verbose and k % 25 == 0:
            print(f"  it{k:3d}: rms point error = {err:5.1f} Hz  max|zeta|={zmax:5.1f}/{zeta_all:.0f}")
    return best


def _demo():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # design spec: a harvester that sits near a mandated line and threads 3 points
    # spec: an amplitude-stable resonator mandated near ~340 Hz that stays within
    # ~+4% detuning to a/h=0.48 (vs a naive panel's +16%).  The points trace this
    # flat-but-hardening target backbone -- a real positive slope inside the panel's
    # achievable envelope, not a vertical/flat stack it cannot deliver.
    pts = [(338.0, 0.16), (347.0, 0.34), (358.0, 0.48)]       # (f_Hz, a/h) on ONE cubic backbone
    print("threading the backbone through design points (mode-adaptive, real f1)...")
    err, th, zmax = optimize_points(pts, R_mm=25.0, iters=220, verbose=True)
    st = forward(th); st0 = forward(np.zeros(M))
    print(f"\nachieved: f1={st['w1']/2/np.pi:.1f} Hz (real adapted-mode), rms point error {err:.1f} Hz, "
          f"curl-feasible R>={1000/zmax:.0f}mm")
    for ftar, r in pts:
        print(f"   point ({ftar:.0f} Hz, a/h={r:.2f}) -> design backbone {backbone_hz(st, r):.1f} Hz")

    rr = np.linspace(0, 0.5, 120)
    fig, ax = plt.subplots(1, 2, figsize=(ps.FULL, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax[0].plot([backbone_hz(st0, r) for r in rr], rr, color=ps.GREY, lw=1.7,
               label=rf"unidirectional ($f_1={st0['w1']/2/np.pi:.0f}$ Hz)")
    ax[0].plot([backbone_hz(st, r) for r in rr], rr, color=ps.GREEN, lw=1.8,
               label=rf"designed toolpath ($f_1={st['w1']/2/np.pi:.0f}$ Hz)")
    for ftar, r in pts:
        ax[0].plot(ftar, r, "o", ms=7, mfc="none", mec=ps.RED, mew=1.5, zorder=5)
    ax[0].plot([], [], "o", mfc="none", mec=ps.RED, mew=1.5, label="prescribed design points")
    ax[0].set_xlabel(r"resonance frequency  (Hz)"); ax[0].set_ylabel(r"amplitude  $a/h$")
    ax[0].set_title("(a) backbone threaded through points")
    ps.grid(ax[0]); ps.despine(ax[0])
    ax[0].legend(frameon=False, loc="lower right")

    nelx, nely = P.mesh.nelx, P.mesh.nely
    xs = (np.arange(nelx)+0.5)*P.mesh.dx; ys = (np.arange(nely)+0.5)*P.mesh.dy
    tg = (P.Phi@th).reshape(nely, nelx)
    Rmm = np.clip(1000.0/np.maximum(np.abs(P.curl(th)[0]), 1e-6), 0, 200).reshape(nely, nelx)
    sp = ax[1].streamplot(xs, ys, np.cos(tg), np.sin(tg), density=2.0, color=Rmm,
                          cmap="RdYlGn", norm=plt.Normalize(25, 120), linewidth=1.0, arrowsize=1e-6)
    ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title(rf"(b) the toolpath ($R\geq{1000/zmax:.0f}$ mm)")
    plt.colorbar(sp.lines, ax=ax[1], fraction=0.046, pad=0.02, label=r"min tow radius $R$ (mm)")
    fig.tight_layout()
    ps.save(fig, "/home/jrt/NLTopt/figs/plate_designpoints")


if __name__ == "__main__":
    _fd_checks()
    if "--demo" in sys.argv or True:
        print()
        _demo()
