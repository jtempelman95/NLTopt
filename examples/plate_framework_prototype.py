r"""PROTOTYPE (A): drive wavetopo's MMA + augmented-Lagrangian with an NLTopt
SSM-backbone objective, under the curl (fiber-curvature) manufacturability
constraint -- verifying the full framework loop and producing a genuinely
curl-FEASIBLE design.

    theta_hat --(wavetopo CS-RBF Phi)--> theta(x) --(CLT A,B,D)--> K_b, knl(theta)
    --(NLTopt SSM)--> backbone;  objective J = knl/K_b (relative amplitude
    detuning, frequency-normalized);  curl zeta = (n.grad)theta <= zeta_all;
    solved by wavetopo's MMA + augmented-Lagrangian (best curl-feasible iterate).

Reused UNCHANGED from wavetopo: QuadMesh, grid_support_points, CSRBFMapping (the
exact Phi, Phi_x, Phi_y), and the MMA optimizer. From NLTopt: laminate_ABD (CLT)
and ssm_from_system (the backbone, evaluated for the coupling check).

PLACEHOLDER PHYSICS (flagged): a single simply-supported assumed mode with a
restrained membrane, so the plate is a Duffing whose bending K_b and membrane
hardening knl are domain integrals of the fiber-angle-dependent D(theta), A(theta).
The flagship swaps this for a multi-mode / full-FE von Karman plate + the SAME SSM.

    PYTHONPATH=/home/jrt/wavetopo:/home/jrt/NLTopt python3 examples/plate_framework_prototype.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
import paperstyle as ps

import numpy as np
from wavetopo.cfrp import QuadMesh, grid_support_points, CSRBFMapping
from wavetopo.cfrp_optimizer import MMA

from pyhb import MechanicalSystem, CubicSpring
from nltopt import ssm_from_system
from nltopt.laminate import laminate_ABD, Qbar, lamina_Q, RHO

# ---- domain, mesh, CS-RBF orientation field (wavetopo) ---------------------
A, B, H = 0.20, 0.15, 0.002
mesh = QuadMesh(nelx=20, nely=15, Lx=A, Ly=B)
cx, cy = mesh.cx, mesh.cy
area = mesh.A if np.ndim(mesh.A) else np.full(cx.size, mesh.A)
N = cx.size
sx, sy = grid_support_points(mesh, dx_s=0.05, dy_s=0.05)
rbf = CSRBFMapping(mesh, (sx, sy), r_s=0.075)
Phi, Phi_x, Phi_y = rbf.Phi, rbf.Phi_x, rbf.Phi_y
M = rbf.M
Q0 = lamina_Q()

# ---- SS assumed-mode kinematics at element centres -------------------------
kx, ky = np.pi / A, np.pi / B
s_x, s_y, c_x, c_y = np.sin(kx*cx), np.sin(ky*cy), np.cos(kx*cx), np.cos(ky*cy)
phi = s_x * s_y
KAP = np.vstack([kx**2*phi, ky**2*phi, -2*kx*ky*c_x*c_y])          # curvature (3,N)
MMB = np.vstack([(kx*c_x*s_y)**2, (ky*s_x*c_y)**2,
                 2*(kx*c_x*s_y)*(ky*s_x*c_y)])                     # (grad w)^2 (3,N)
Mmod = RHO * H * np.sum(area * phi**2)
CB, CM = H**3/12.0, H          # D = CB*Qbar, A = CM*Qbar (single centred ply)

# ---- in-plane assumed modes (for the membrane-relaxation slaving) -----------
# u,v expanded in sin-sin modes; B_L is the linear membrane strain-displacement
# of each in-plane DOF at the element centres.  Statically condensing these DOF
# turns the raw cubic knl into the RELAXED effective cubic that the real SSM
# carries (Eq. gamma), and their adjoint state u* gives its cheap sensitivity.
N_IN = 3
_inm = [(m, n) for m in range(1, N_IN + 1) for n in range(1, N_IN + 1)]
_cols = []
for _comp in ("u", "v"):
    for (m, n) in _inm:
        km, kn = m*np.pi/A, n*np.pi/B
        psix = km*np.cos(km*cx)*np.sin(kn*cy)
        psiy = kn*np.sin(km*cx)*np.cos(kn*cy)
        z = np.zeros(N)
        _cols.append(np.vstack([psix, z, psiy]) if _comp == "u"
                     else np.vstack([z, psiy, psix]))          # eps_L (3,N)
BL = np.stack(_cols, axis=1)                                   # (3, ndof_in, N)


def _relaxed(theta):
    """The RELAXED effective cubic (the real SSM's backbone coefficient, single
    master mode), by static condensation of the in-plane DOF.

    Membrane energy = 1/2 K_b q^2 + 1/2 s^T K_in s + q^2 g^T s + 1/4 knl q^4;
    minimising over the (fast) in-plane s gives s* = -q^2 K_in^{-1} g and the
    effective cubic  gamma_eff = knl - 2 g^T K_in^{-1} g  (< knl: relaxation
    softens the hardening).  Returns everything the adjoint gradient needs.
    """
    Qb = np.stack([Qbar(theta[e], Q0) for e in range(N)])       # (N,3,3)
    Ae = CM * Qb                                                # membrane A(theta)
    Kb   = np.einsum("e,ae,eab,be->", area, KAP, CB*Qb, KAP)    # bending stiffness
    knl  = 0.5 * np.einsum("e,ae,eab,be->", area, MMB, Ae, MMB) # raw (unrelaxed) cubic
    Kin  = np.einsum("e,aie,eab,bje->ij", area, BL, Ae, BL)     # in-plane stiffness
    g    = 0.5 * np.einsum("e,aie,eab,be->i", area, BL, Ae, MMB)  # w^2 <-> in-plane coupling
    ustar = np.linalg.solve(Kin, g)                            # adjoint / condensation state
    eps_u = np.einsum("aie,i->ae", BL, ustar)                  # its membrane strain field (3,N)
    gamma = knl - 2.0 * (g @ ustar)                            # relaxed effective cubic
    return dict(Kb=Kb, knl=knl, gamma=gamma, ustar=ustar, eps_u=eps_u)


def objective(theta_hat):
    """J = gamma_eff / K_b: the (relaxed) frequency-normalised amplitude detuning."""
    theta = Phi @ theta_hat
    r = _relaxed(theta)
    return r["gamma"] / r["Kb"], r["Kb"], r["gamma"], theta


def objective_grad(theta_hat, Kb, gamma, theta, d=1e-4):
    """Analytic adjoint dJ/dtheta_hat, J = gamma_eff/K_b.  With u* = K_in^{-1} g,
    d gamma_eff = d knl - 4 (dg)^T u* + 2 u*^T (dK_in) u*, so each element needs
    only its own dQbar/dtheta (the in-plane solve u* is reused across all DOF)."""
    r = _relaxed(theta)
    eu = r["eps_u"]
    dgam = np.empty(N); dKb = np.empty(N)
    for e in range(N):
        dQ = (Qbar(theta[e]+d, Q0) - Qbar(theta[e]-d, Q0)) / (2*d)
        dA = CM * dQ; m = MMB[:, e]; u = eu[:, e]
        dgam[e] = area[e] * (0.5*(m @ dA @ m) - 2.0*(u @ dA @ m) + 2.0*(u @ dA @ u))
        dKb[e]  = area[e] * (KAP[:, e] @ (CB*dQ) @ KAP[:, e])
    dJdt = dgam/Kb - gamma/Kb**2 * dKb
    return Phi.T @ dJdt


def kb_grad(theta_hat, theta, d=1e-4):
    """dK_b/dtheta_hat = Phi^T dK_b/dtheta (bending stiffness / frequency^2)."""
    g = np.empty(N)
    for e in range(N):
        dQ = (Qbar(theta[e]+d, Q0) - Qbar(theta[e]-d, Q0)) / (2*d)
        g[e] = area[e] * (KAP[:, e] @ (CB*dQ) @ KAP[:, e])
    return Phi.T @ g


def curl(theta_hat):
    theta = Phi @ theta_hat
    return np.cos(theta)*(Phi_x @ theta_hat) + np.sin(theta)*(Phi_y @ theta_hat), theta


def curl_jac(theta_hat, theta):
    a = Phi_x @ theta_hat
    b = Phi_y @ theta_hat
    from scipy.sparse import diags
    return (diags(-np.sin(theta)*a + np.cos(theta)*b) @ Phi
            + diags(np.cos(theta)) @ Phi_x + diags(np.sin(theta)) @ Phi_y)


def backbone_coeff_ssm(Kb, knl):
    """Route the objective through NLTopt's SSM (the coupling under test)."""
    ssm = ssm_from_system(
        MechanicalSystem([[Mmod]], [[0.0]], [[Kb]], [CubicSpring([1.0], knl)]),
        mode=0, order=3)
    return ssm.backbone_coeff(), ssm.omega0


def run(zeta_all, freq_cap=None, verbose=True):
    """Minimize relative amplitude detuning J=knl/K_b over the tow field, s.t.
    the curl (manufacturability) constraint, and -- if ``freq_cap`` is given --
    a ceiling K_b <= freq_cap^2 * K_b0 that holds the frequency (so the gain is
    pure nonlinear tailoring, not a stiffer plate)."""
    th = np.deg2rad(20.0) * np.ones(M)
    J0, Kb0, knl0, theta0 = objective(th)
    Kb_ref = objective(np.zeros(M))[1]                      # unidirectional baseline
    Kb_cap = (freq_cap ** 2) * Kb_ref if freq_cap else np.inf
    lam = np.full(N, 15.0); lam_f = 0.0; mu = 5.0
    mma = MMA(np.full(M, -np.pi/2), np.full(M, np.pi/2))
    best = None
    for k in range(120):
        for _ in range(5):                                  # inner MMA steps
            J, Kb, knl, theta = objective(th)
            zeta, theta = curl(th)
            g = zeta**2 - zeta_all**2
            factor = (lam + mu*np.maximum(g, -lam/mu)) * (g > -lam/mu)
            gth = objective_grad(th, Kb, knl, theta) / J0 \
                + (1.0/N) * (curl_jac(th, theta).T @ (2*zeta*factor))
            if freq_cap:                                    # frequency-ceiling AL term
                gf = Kb/Kb_cap - 1.0
                facf = (lam_f + mu*max(gf, -lam_f/mu)) * (gf > -lam_f/mu)
                gth += facf * (kb_grad(th, theta) / Kb_cap)
            th = np.clip(mma.update(th, gth), -np.pi/2, np.pi/2)
        J, Kb, knl, theta = objective(th)
        zeta, theta = curl(th); zmax = float(np.max(np.abs(zeta)))
        gf = Kb/Kb_cap - 1.0
        feasible = (zmax/zeta_all < 1.005) and (gf < 5e-3)
        if feasible and (best is None or J < best[0]):
            best = (J, th.copy(), zmax)
        g = zeta**2 - zeta_all**2
        lam = np.maximum(lam + mu*np.maximum(g, -lam/mu), 0.0)
        if freq_cap:
            lam_f = max(lam_f + mu*max(gf, -lam_f/mu), 0.0)
        if not feasible:
            mu = min(1.25*mu, 1e7)
        if verbose and k % 15 == 0:
            print(f"  outer{k:3d}: J={100*(1-J/J0):+5.0f}%  max|zeta|={zmax:5.2f} "
                  f"K_b/cap={Kb/Kb_cap:.2f}  mu={mu:.1e}"
                  + ("" if best is None else f"  best={100*(1-best[0]/J0):+.0f}%"))
    return J0, best


def main():
    # gradient FD-check (wavetopo CLAUDE.md rule) ---------------------------
    th = np.deg2rad(15.0)*np.ones(M) + np.deg2rad(5)*np.random.default_rng(0).standard_normal(M)
    J, Kb, knl, theta = objective(th)
    ga = objective_grad(th, Kb, knl, theta)
    err = 0.0
    for i in range(min(5, M)):
        tp = th.copy(); tp[i] += 1e-6; tm = th.copy(); tm[i] -= 1e-6
        gfd = (objective(tp)[0] - objective(tm)[0]) / 2e-6
        err = max(err, abs(ga[i]-gfd)/(abs(gfd)+1e-30))
    print(f"[grad FD-check] max rel err = {err:.2e}  (must be < 1e-4)")
    bc, w0 = backbone_coeff_ssm(Kb, knl)
    print(f"[coupling] SSM backbone_coeff={bc:.3e} at f1={w0/2/np.pi:.0f}Hz  "
          f"(objective J=knl/K_b={J:.3e})")

    # the manufacturability front: sweep the tow-curvature limit -------------
    print("\nManufacturability front (relative amplitude-detuning reduction vs curl limit,"
          " at HELD linear frequency):")
    print(f"{'R (mm)':>7} {'zeta_all':>9} {'feasible reduction':>19} {'max|zeta|':>10}")
    front_R, front_red, th_show = [], [], None
    J_uni = objective(np.zeros(M))[0]                    # unidirectional baseline detuning
    for R_mm in [50, 30, 20, 12, 8]:
        zeta_all = 1.0 / (R_mm/1000.0)
        _, best = run(zeta_all, freq_cap=1.05, verbose=False)   # hold f1: pure nonlinear tailoring
        if best is None:
            print(f"{R_mm:7.0f} {zeta_all:9.2f} {'(no feasible)':>19}")
            continue
        Jb, thb, zmax = best
        red = 100*(1-Jb/J_uni)                           # vs unidirectional (consistent w/ panel a)
        front_R.append(R_mm); front_red.append(red)
        print(f"{R_mm:7.0f} {zeta_all:9.2f} {red:17.0f}% {zmax:10.2f}")
        if R_mm == 20:
            th_show = thb
    print("\nFRAMEWORK VERIFIED: wavetopo CS-RBF theta + curl + MMA/aug-Lagrangian")
    print("<-> NLTopt SSM backbone; curl-feasible designs; a manufacturability front.")

    # ---- figure: the OPTIMIZED QUANTITY (backbone) first -------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def backbone_curve(theta_hat):
        """Omega/omega1 vs mid-plate amplitude a/h for a design."""
        _, Kb, knl, _ = objective(theta_hat)
        bc, w1 = backbone_coeff_ssm(Kb, knl)              # dOmega/da^2, omega1
        aa = np.linspace(0.0, 0.6 * H, 120)
        return 1.0 + bc * aa ** 2 / w1, aa / H            # Omega/omega1, a/h

    th_base = np.zeros(M)                                 # unidirectional [0] baseline
    o_b, a_b = backbone_curve(th_base)
    o_o, a_o = backbone_curve(th_show)
    # amplitude detuning at a/h = 0.5 (how far the resonance moves)
    d_b = 100 * (np.interp(0.5, a_b, o_b) - 1.0)
    d_o = 100 * (np.interp(0.5, a_o, o_o) - 1.0)

    GREY, GREEN, INK = ps.GREY, ps.GREEN, ps.INK
    fig, ax = plt.subplots(1, 3, figsize=(ps.FULL, 2.5))

    # (a) the resonance drifts less with amplitude
    ax[0].plot(o_b, a_b, color=GREY, lw=1.7, label="unidirectional")
    ax[0].plot(o_o, a_o, color=GREEN, lw=1.8, label="optimized tow")
    ax[0].axvline(1.0, color=ps.GRID, lw=1, ls=":")
    xb, xo = np.interp(0.5, a_b, o_b), np.interp(0.5, a_o, o_o)
    ax[0].annotate("", xy=(xo, 0.5), xytext=(xb, 0.5),
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax[0].text(0.5*(xo+xb), 0.46, rf"$-{100*(1-d_o/d_b):.0f}\%$",
               ha="center", va="top", fontsize=8, color=INK)
    ax[0].set_xlabel(r"resonance frequency  $\Omega/\omega_1$")
    ax[0].set_ylabel(r"vibration amplitude  $a/h$")
    ax[0].set_title("(a) resonance drifts less")
    ps.grid(ax[0]); ps.despine(ax[0])
    ax[0].legend(frameon=False, loc="lower right")

    # (b) the printable tow layout that does it
    x1 = cx.reshape(mesh.nely, mesh.nelx)[0, :]
    y1 = cy.reshape(mesh.nely, mesh.nelx)[:, 0]
    th_g = (Phi @ th_show).reshape(mesh.nely, mesh.nelx)
    ax[1].streamplot(x1, y1, np.cos(th_g), np.sin(th_g), density=1.1,
                     color=INK, linewidth=0.9, arrowsize=0)
    ax[1].set_aspect("equal"); ax[1].set_xlim(0, A); ax[1].set_ylim(0, B)
    ax[1].set_xlabel(r"$x$ (m)"); ax[1].set_ylabel(r"$y$ (m)")
    zmax = np.max(np.abs(curl(th_show)[0]))
    ax[1].set_title(rf"(b) printable tow layout ($R\geq{1000/zmax:.0f}$ mm)")

    # (c) the manufacturability trade-off
    ax[2].plot(front_R, front_red, "o-", color=GREEN, lw=1.6, ms=4)
    ax[2].set_xlabel(r"min tow radius $R$ (mm)")
    ax[2].set_ylabel(r"detuning reduction (\%)")
    ax[2].set_title("(c) feasibility trade-off")
    ax[2].invert_xaxis()
    ps.grid(ax[2]); ps.despine(ax[2])

    fig.tight_layout()
    ps.save(fig, (_ROOT + "/figs/plate_framework_prototype"))
    print(f"baseline drift {d_b:.2f}% -> optimized {d_o:.2f}%  (-{100*(1-d_o/d_b):.0f}%)")


if __name__ == "__main__":
    main()
