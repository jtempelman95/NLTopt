r"""Two-field topology + fiber-steering optimization of the plate's NONLINEAR
backbone.

Adds the SECOND wavetopo design field -- the structural DENSITY z -- to the tow
orientation theta, so the design is where material goes AND which way the fibers
run.  The element stiffnesses are SIMP-scaled by the (filtered) density
``E(y)=eps+(1-eps)y^p``, ``y=Pz``, exactly as in wavetopo's linear composite TO
(cfrp_problem), but here they feed the RELAXED SSM backbone coefficient
``gamma_eff = knl - 2 g^T K_in^{-1} g`` (Eq. gamma_eff).  The design minimizes the
frequency-normalized amplitude detuning ``J = gamma_eff / K_b`` subject to:
  * the tow-curvature (curl) manufacturability limit,
  * a material-volume budget (mean density <= v_all),
  * a linear-frequency floor (so the gain is nonlinear tailoring, not a lighter
    or softer plate).

Both sensitivities are analytic and FD-verified (dJ/dz and dJ/dtheta), and the
orientation gradient passes the isotropy-annihilation check (isotropic Q -> the
tow angle is inert -> dJ/dtheta = 0), per the wavetopo gradient rules.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_topopt.py --check
    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_topopt.py
"""
import sys
sys.path.insert(0, "/home/jrt/wavetopo")
sys.path.insert(0, "/home/jrt/NLTopt/examples")
import paperstyle as ps

import numpy as np
import plate_framework_prototype as P
from wavetopo.cfrp_problem import density_filter
from wavetopo.cfrp import heaviside, dheaviside
from nltopt.laminate import Qbar, lamina_Q, RHO

ETA = 0.5                                                     # Heaviside threshold
BETA_MAX = 4.0                                                # projection sharpness (cap)

# ---- reuse the prototype's mesh, kinematics, CS-RBF, curl -------------------
mesh, Phi = P.mesh, P.Phi
cx, cy, area, N, M = P.cx, P.cy, P.area, P.N, P.M
KAP, MMB, BL = P.KAP, P.MMB, P.BL
CB, CM, H, A, B = P.CB, P.CM, P.H, P.A, P.B
Q0 = P.Q0
phi = P.phi                                                   # transverse mode at centres

# ---- density regularization (wavetopo linear filter) + SIMP ----------------
R_FILT = 0.018
Pd = density_filter(mesh, R_FILT)
PENAL, EPSD = 3.0, 1e-4

# quarter-symmetry (the (1,1) mode is symmetric about x=A/2 and y=B/2): project
# the density onto the symmetric subspace so the topology converges to the
# physical optimum rather than a lopsided local minimum.
from scipy.spatial import cKDTree as _cKDTree
_tree = _cKDTree(np.column_stack([cx, cy]))
def _rmap(fx, fy):
    tgt = np.column_stack([(A - cx) if fx else cx, (B - cy) if fy else cy])
    return _tree.query(tgt)[1]
_MX, _MY, _MXY = _rmap(1, 0), _rmap(0, 1), _rmap(1, 1)
def symmetrize(v):
    return 0.25 * (v + v[_MX] + v[_MY] + v[_MXY])


def _perelem(theta):
    """Per-element (density-free) quadratic forms for orientation ``theta``."""
    Qb = np.stack([Qbar(theta[e], Q0) for e in range(N)])     # (N,3,3)
    Am, Dm = CM * Qb, CB * Qb
    kb_e  = area * np.einsum("ae,eab,be->e", KAP, Dm, KAP)              # (N,)
    knl_e = 0.5 * area * np.einsum("ae,eab,be->e", MMB, Am, MMB)       # (N,)
    Kin_e = area[:, None, None] * np.einsum("aie,eab,bje->eij", BL, Am, BL)  # (N,nd,nd)
    g_e   = 0.5 * area[:, None] * np.einsum("aie,eab,be->ei", BL, Am, MMB)   # (N,nd)
    return Qb, Am, kb_e, knl_e, Kin_e, g_e


def forward(z, theta_hat, beta=8.0):
    """Assemble the density-weighted relaxed backbone objective J=gamma_eff/K_b.

    Three-field density: raw z -> filter (P) -> Heaviside project (beta) ->
    SIMP.  ``beta`` continuation crisps the topology to 0/1."""
    theta = Phi @ theta_hat
    yf = Pd @ z                                               # filtered density
    y = heaviside(yf, ETA, beta)                              # projected physical density
    E = EPSD + (1.0 - EPSD) * y ** PENAL                      # SIMP
    Qb, Am, kb_e, knl_e, Kin_e, g_e = _perelem(theta)
    Kb  = E @ kb_e
    knl = E @ knl_e
    Kin = np.einsum("e,eij->ij", E, Kin_e)
    g   = np.einsum("e,ei->i", E, g_e)
    ustar = np.linalg.solve(Kin, g)                           # adjoint / condensation state
    gamma = knl - 2.0 * (g @ ustar)
    Mmod = RHO * H * np.sum(y * area * phi ** 2)              # mass ~ physical density
    w1 = np.sqrt(Kb / Mmod)
    return dict(J=gamma / Kb, Kb=Kb, gamma=gamma, y=y, yf=yf, beta=beta, E=E,
                ustar=ustar, w1=w1, theta=theta, kb_e=kb_e, knl_e=knl_e,
                Kin_e=Kin_e, g_e=g_e, Mmod=Mmod)


def grad(st, d=1e-4):
    """Analytic sensitivities from the forward state ``st``.  Returns dJ/dz,
    dJ/dtheta, and the raw dK_b sensitivities (dz, dtheta) the frequency
    constraint reuses -- so the (expensive) per-element theta loop runs once."""
    E, Kb, gamma, ustar = st["E"], st["Kb"], st["gamma"], st["ustar"]
    kb_e, knl_e, Kin_e, g_e = st["kb_e"], st["knl_e"], st["Kin_e"], st["g_e"]
    theta, y, yf, beta = st["theta"], st["y"], st["yf"], st["beta"]
    eu = np.einsum("aie,i->ae", BL, ustar)                    # adjoint strain field (3,N)
    dE_dy = (1.0 - EPSD) * PENAL * y ** (PENAL - 1.0)
    dproj = dheaviside(yf, ETA, beta)                         # d(projected)/d(filtered)

    # --- dJ/dz and dK_b/dz : through SIMP E(y), Heaviside, and the filter P ---
    dgam_dE = knl_e - 4.0 * (g_e @ ustar) + 2.0 * np.einsum("i,eij,j->e", ustar, Kin_e, ustar)
    dJ_dE = dgam_dE / Kb - gamma / Kb ** 2 * kb_e
    dmass = RHO * H * area * phi ** 2                         # dMmod/dy
    dJ_dz    = Pd.T @ (dJ_dE * dE_dy * dproj)
    dKb_dz   = Pd.T @ (kb_e * dE_dy * dproj)
    dMmod_dz = Pd.T @ (dmass * dproj)
    dvol_dz  = Pd.T @ (dproj / N)

    # --- dJ/dtheta and dK_b/dtheta : per-element dQbar, weighted by E ---
    dgam_dth = np.empty(N); dKb_dth_e = np.empty(N)
    for e in range(N):
        dQ = (Qbar(theta[e] + d, Q0) - Qbar(theta[e] - d, Q0)) / (2 * d)
        dA, dD = CM * dQ, CB * dQ
        m, u, ka = MMB[:, e], eu[:, e], KAP[:, e]
        dgam_dth[e]  = E[e] * area[e] * (0.5*(m @ dA @ m) - 2.0*(u @ dA @ m) + 2.0*(u @ dA @ u))
        dKb_dth_e[e] = E[e] * area[e] * (ka @ dD @ ka)
    dJ_dth  = Phi.T @ (dgam_dth / Kb - gamma / Kb ** 2 * dKb_dth_e)
    dKb_dth = Phi.T @ dKb_dth_e
    return dict(dJ_dz=dJ_dz, dJ_dth=dJ_dth, dKb_dz=dKb_dz, dKb_dth=dKb_dth,
                dMmod_dz=dMmod_dz, dvol_dz=dvol_dz)


def _fd_checks():
    rng = np.random.default_rng(1)
    z = 0.6 + 0.2 * rng.standard_normal(N); z = np.clip(z, 0.05, 1.0)
    th = np.deg2rad(15.0) * np.ones(M) + np.deg2rad(8) * rng.standard_normal(M)
    G = grad(forward(z, th)); gz, gt = G["dJ_dz"], G["dJ_dth"]

    ez = 0.0
    for i in rng.choice(N, 6, replace=False):
        zp = z.copy(); zp[i] += 1e-6; zm = z.copy(); zm[i] -= 1e-6
        fd = (forward(zp, th)["J"] - forward(zm, th)["J"]) / 2e-6
        ez = max(ez, abs(gz[i] - fd) / (abs(fd) + 1e-30))
    et = 0.0
    for i in range(6):
        tp = th.copy(); tp[i] += 1e-6; tm = th.copy(); tm[i] -= 1e-6
        fd = (forward(z, tp)["J"] - forward(z, tm)["J"]) / 2e-6
        et = max(et, abs(gt[i] - fd) / (abs(fd) + 1e-30))
    print(f"[FD] dJ/dz  max rel err = {ez:.2e}   (must be < 1e-4)")
    print(f"[FD] dJ/dth max rel err = {et:.2e}   (must be < 1e-4)")

    # isotropy-annihilation: isotropic Q => tow angle inert => dJ/dtheta = 0
    Qc = lamina_Q().copy()
    Qiso = np.array([[Qc[0, 0], Qc[0, 1], 0.0],
                     [Qc[0, 1], Qc[0, 0], 0.0],
                     [0.0, 0.0, 0.5 * (Qc[0, 0] - Qc[0, 1])]])
    global Q0
    Q0_save = Q0; Q0 = Qiso
    gt_iso = grad(forward(z, th))["dJ_dth"]
    Q0 = Q0_save
    print(f"[isotropy] max|dJ/dtheta| = {np.max(np.abs(gt_iso)):.2e}   (must be ~0)")


def optimize(v_all=0.5, R_mm=20.0, freq_band=(0.97, 1.05), iters=320, verbose=True):
    """Joint (density z, orientation theta) design: min J=gamma_eff/K_b s.t. the
    curl (manufacturability) limit, a volume budget mean(y)<=v_all, and a
    linear-frequency BAND freq_band*w1_ref (a floor AND a ceiling -- the ceiling
    forbids the spurious low-density localized mode where the SIMP eps-stiffness
    with vanishing mass sends w1 up and gamma_eff trivially to zero).  MMA on
    [z; theta] with an augmented-Lagrangian for the constraints."""
    from wavetopo.cfrp_optimizer import MMA
    zeta_all = 1.0 / (R_mm / 1000.0)
    f_lo, f_hi = freq_band
    ref = forward(np.ones(N), np.zeros(M))                    # solid unidirectional
    w1_ref, J_ref = ref["w1"], ref["J"]

    x = np.concatenate([v_all * np.ones(N), np.deg2rad(20.0) * np.ones(M)])
    lo = np.concatenate([0.05 * np.ones(N), -np.pi/2 * np.ones(M)])
    hi = np.concatenate([np.ones(N),         np.pi/2 * np.ones(M)])
    mma = MMA(lo, hi)
    lam_c = np.full(N, 10.0); lam_v = lam_lo = lam_hi = 0.0; mu = 5.0
    best = None
    for k in range(iters):
        beta = min(1.0 * 1.5 ** (k // 30), BETA_MAX)         # Heaviside continuation
        z, th = x[:N], x[N:]
        st = forward(z, th, beta); G = grad(st)
        c = st["w1"] / (2*st["Kb"])                           # dw1/dKb
        dw1_dz  = c*G["dKb_dz"] - st["w1"]/(2*st["Mmod"]) * G["dMmod_dz"]
        dw1_dth = c*G["dKb_dth"]
        # curl (per element, theta only)
        zeta, theta = P.curl(th); gc = zeta**2 - zeta_all**2
        fac_c = (lam_c + mu*np.maximum(gc, -lam_c/mu)) * (gc > -lam_c/mu)
        gth_c = (1.0/N) * (P.curl_jac(th, theta).T @ (2*zeta*fac_c))
        # volume EQUALITY (z only):  mean(y) = v_all  (a fixed material budget, so
        # the objective can't trivially drive material to zero)
        gv = st["y"].mean() - v_all
        fac_v = lam_v + mu*gv                                 # always active (equality)
        gz_v = fac_v * G["dvol_dz"]
        # frequency band:  g_lo = f_lo*w1_ref - w1 <= 0 ;  g_hi = w1 - f_hi*w1_ref <= 0
        g_lo = f_lo*w1_ref - st["w1"]; g_hi = st["w1"] - f_hi*w1_ref
        fac_lo = (lam_lo + mu*max(g_lo, -lam_lo/mu)) * (g_lo > -lam_lo/mu)
        fac_hi = (lam_hi + mu*max(g_hi, -lam_hi/mu)) * (g_hi > -lam_hi/mu)
        gf_z  = (-fac_lo + fac_hi) * dw1_dz
        gf_th = (-fac_lo + fac_hi) * dw1_dth
        # augmented objective gradient on [z; theta]
        g = np.empty_like(x)
        g[:N] = symmetrize(G["dJ_dz"]/J_ref + gz_v + gf_z)   # keep topology quarter-symmetric
        g[N:] = G["dJ_dth"]/J_ref + gth_c + gf_th
        x = np.clip(mma.update(x, g), lo, hi)
        # feasibility + best-feasible retention (at the running beta)
        z, th = x[:N], x[N:]; st = forward(z, th, beta)
        zmax = float(np.max(np.abs(P.curl(th)[0])))
        vol = st["y"].mean(); wr = st["w1"]/w1_ref
        feas = (zmax/zeta_all < 1.02) and (vol <= v_all + 5e-2) \
            and (f_lo - 1e-2 <= wr <= f_hi + 1e-2)
        cand = dict(J=st["J"], z=z.copy(), th=th.copy(), w1r=wr, vol=vol, zmax=zmax, beta=beta)
        if feas and (best is None or st["J"] < best["J"]):
            best = cand
        lam_c = np.maximum(lam_c + mu*np.maximum(gc, -lam_c/mu), 0.0)
        lam_v = lam_v + mu*gv                                 # equality: sign-free
        lam_lo = max(lam_lo + mu*max(g_lo, -lam_lo/mu), 0.0)
        lam_hi = max(lam_hi + mu*max(g_hi, -lam_hi/mu), 0.0)
        if not feas:
            mu = min(1.3*mu, 1e7)
        if verbose and k % 20 == 0:
            print(f"  it{k:3d}: J={100*(1-st['J']/J_ref):+5.1f}%  vol={vol:.2f} "
                  f"w1/ref={wr:.3f} max|z|={zmax:5.1f}/{zeta_all:.0f}  mu={mu:.1e}"
                  + ("" if best is None else f"  best={100*(1-best['J']/J_ref):+.1f}%"))
    return J_ref, w1_ref, best or cand                        # fall back to final iterate


def _backbone(z, theta_hat, amax=0.6):
    """Omega/omega1 vs a/h for a (z, theta) design, via the 1-DOF relaxed SSM."""
    st = forward(z, theta_hat, BETA_MAX)
    bc, w1 = P.backbone_coeff_ssm(st["Kb"], st["gamma"])
    aa = np.linspace(0.0, amax * H, 120)
    return 1.0 + bc * aa ** 2 / w1, aa / H


def figure(z, th, Jref, path="figs/plate_topopt.png"):
    """True manufacturable-toolpath + topology figure, rendered with wavetopo's
    own tow/curl renderers (cfrp_viz)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wavetopo.cfrp_viz import orientation_streamlines, curl_map

    theta = Phi @ th
    st = forward(z, th, BETA_MAX)
    dens = st["y"]                                             # the crisp physical topology
    zeta, _ = P.curl(th)

    fig, axg = plt.subplots(2, 2, figsize=(ps.FULL, 4.6)); ax = axg.ravel()
    # (a) topology
    dg = dens.reshape(mesh.nely, mesh.nelx)
    im0 = ax[0].imshow(dg, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                       extent=[0, A, 0, B], aspect="equal")
    ax[0].set_title(r"(a) topology $z$")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)
    # (b) the manufacturable toolpath (fiber streamlines inside the structure)
    orientation_streamlines(ax[1], mesh, theta, dens, color=ps.RED, linewidth=0.7)
    ax[1].set_title("(b) fiber toolpaths")
    # (c) curl / steering feasibility
    im2 = curl_map(ax[2], mesh, zeta, dens)
    ax[2].set_title(rf"(c) $|\zeta|$ (max {np.max(np.abs(zeta)):.0f}/m)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03)
    # (d) the payoff: flattened backbone at held frequency
    o_b, a_b = _backbone(np.ones(N), np.zeros(M))
    o_o, a_o = _backbone(z, th)
    d_b = 100*(np.interp(0.5, a_b, o_b) - 1.0); d_o = 100*(np.interp(0.5, a_o, o_o) - 1.0)
    ax[3].plot(o_b, a_b, color=ps.GREY, lw=1.6, label=rf"solid $[0]$ ({d_b:.1f}\%)")
    ax[3].plot(o_o, a_o, color=ps.GREEN, lw=1.6, label=rf"topology+tow ({d_o:.1f}\%)")
    ax[3].axvline(1.0, color=ps.GRID, lw=1, ls=":")
    ax[3].set_xlabel(r"$\Omega/\omega_1$")
    ax[3].set_ylabel(r"amplitude $a/h$")
    ax[3].set_title(rf"(d) detuning $-{100*(1-d_o/d_b):.0f}\%$")
    ps.grid(ax[3]); ps.despine(ax[3])
    ax[3].legend(frameon=False, loc="lower right")
    for a in ax[:3]:
        a.set_xlabel(r"$x$ (m)"); a.set_ylabel(r"$y$ (m)")
    fig.tight_layout()
    ps.save(fig, "/home/jrt/NLTopt/figs/plate_topopt")
    print("wrote plate_topopt")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _fd_checks()
    else:
        _fd_checks()
        print("\nJoint (density + orientation) optimization:")
        Jref, w1ref, best = optimize(v_all=0.5, R_mm=20.0, freq_band=(0.93, 1.12),
                                     iters=350, verbose=True)
        print(f"\ntwo-field optimum: detuning reduction {100*(1-best['J']/Jref):.0f}% "
              f"vs solid unidirectional, vol={best['vol']:.2f}, f1 held "
              f"({best['w1r']:.2f} x ref), curl-feasible R>={1000/best['zmax']:.0f}mm")
        figure(best["z"], best["th"], Jref)
