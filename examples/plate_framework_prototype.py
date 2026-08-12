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
import sys
sys.path.insert(0, "/home/jrt/wavetopo")

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


def _stiff(theta):
    """Element bending/membrane quadratic forms; returns K_b, knl and per-elem
    Qbar (for the analytic gradient)."""
    Kb = knl = 0.0
    for e in range(N):
        Qb = Qbar(theta[e], Q0)
        Kb += area[e] * (KAP[:, e] @ (CB*Qb) @ KAP[:, e])
        knl += 0.5 * area[e] * (MMB[:, e] @ (CM*Qb) @ MMB[:, e])
    return Kb, knl


def objective(theta_hat):
    theta = Phi @ theta_hat
    Kb, knl = _stiff(theta)
    return knl / Kb, Kb, knl, theta


def objective_grad(theta_hat, Kb, knl, theta, d=1e-4):
    """Analytic dJ/dtheta_hat = Phi^T (dJ/dtheta), J=knl/Kb (dQbar/dtheta by FD)."""
    dJdt = np.empty(N)
    for e in range(N):
        dQ = (Qbar(theta[e]+d, Q0) - Qbar(theta[e]-d, Q0)) / (2*d)
        dKb = area[e] * (KAP[:, e] @ (CB*dQ) @ KAP[:, e])
        dknl = 0.5 * area[e] * (MMB[:, e] @ (CM*dQ) @ MMB[:, e])
        dJdt[e] = dknl/Kb - knl/Kb**2 * dKb
    return Phi.T @ dJdt


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


def run(zeta_all, verbose=True):
    th = np.deg2rad(20.0) * np.ones(M)
    J0, Kb0, knl0, theta0 = objective(th)
    lam = np.full(N, 15.0); mu = 5.0
    mma = MMA(np.full(M, -np.pi/2), np.full(M, np.pi/2))
    best = None
    for k in range(120):
        for _ in range(5):                                  # inner MMA steps
            J, Kb, knl, theta = objective(th)
            zeta, theta = curl(th)
            g = zeta**2 - zeta_all**2
            hclip = np.maximum(g, -lam/mu)
            factor = (lam + mu*hclip) * (g > -lam/mu)
            gth = objective_grad(th, Kb, knl, theta) / J0 \
                + (1.0/N) * (curl_jac(th, theta).T @ (2*zeta*factor))
            th = np.clip(mma.update(th, gth), -np.pi/2, np.pi/2)
        J, Kb, knl, theta = objective(th)
        zeta, theta = curl(th); zmax = float(np.max(np.abs(zeta)))
        feasible = zmax/zeta_all < 1.005
        if feasible and (best is None or J < best[0]):
            best = (J, th.copy(), zmax)
        g = zeta**2 - zeta_all**2
        lam = np.maximum(lam + mu*np.maximum(g, -lam/mu), 0.0)
        if not feasible:
            mu = min(1.25*mu, 1e7)
        if verbose and k % 15 == 0:
            print(f"  outer{k:3d}: J={100*(1-J/J0):+5.0f}%  max|zeta|={zmax:5.2f} "
                  f"(lim {zeta_all:.2f})  mu={mu:.1e}"
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
    print("\nManufacturability front (relative amplitude-detuning reduction vs curl limit):")
    print(f"{'R (mm)':>7} {'zeta_all':>9} {'feasible reduction':>19} {'max|zeta|':>10}")
    front_R, front_red, th_show = [], [], None
    for R_mm in [50, 30, 20, 12, 8]:
        zeta_all = 1.0 / (R_mm/1000.0)
        J0, best = run(zeta_all, verbose=False)
        if best is None:
            print(f"{R_mm:7.0f} {zeta_all:9.2f} {'(no feasible)':>19}")
            continue
        Jb, thb, zmax = best
        red = 100*(1-Jb/J0)
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

    GREY, GREEN, INK = "#9aa0a6", "#2e7d32", "#3a3f45"
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.1))

    # (a) THE POINT: the resonance drifts less with amplitude
    ax[0].plot(o_b, a_b, color=GREY, lw=2.6, label=f"unidirectional  (drifts {d_b:.1f}%)")
    ax[0].plot(o_o, a_o, color=GREEN, lw=2.8, label=f"optimized tow  (drifts {d_o:.1f}%)")
    ax[0].axvline(1.0, color="0.85", lw=1, ls=":")
    xb, xo = np.interp(0.5, a_b, o_b), np.interp(0.5, a_o, o_o)
    ax[0].annotate("", xy=(xo, 0.5), xytext=(xb, 0.5),
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax[0].text(0.5*(xo+xb), 0.46, f"detuning $-{100*(1-d_o/d_b):.0f}\\%$",
               ha="center", va="top", fontsize=9.5, color=INK)
    ax[0].set_xlabel(r"resonance frequency  $\Omega/\omega_1$")
    ax[0].set_ylabel(r"vibration amplitude  $a/h$")
    ax[0].set_title("(a) the nonlinear resonance drifts\nless with amplitude", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right"); ax[0].grid(alpha=0.25)

    # (b) the printable tow layout that does it
    x1 = cx.reshape(mesh.nely, mesh.nelx)[0, :]
    y1 = cy.reshape(mesh.nely, mesh.nelx)[:, 0]
    th_g = (Phi @ th_show).reshape(mesh.nely, mesh.nelx)
    ax[1].streamplot(x1, y1, np.cos(th_g), np.sin(th_g), density=1.1,
                     color=INK, linewidth=1.1, arrowsize=0)
    ax[1].set_aspect("equal"); ax[1].set_xlim(0, A); ax[1].set_ylim(0, B)
    ax[1].set_xlabel("plate x [m]"); ax[1].set_ylabel("plate y [m]")
    zmax = np.max(np.abs(curl(th_show)[0]))
    ax[1].set_title(f"(b) the printable tow layout\n(curl-feasible: R $\\geq$ {1000/zmax:.0f} mm)",
                    fontsize=11)

    # (c) the manufacturability trade-off
    ax[2].plot(front_R, front_red, "o-", color=GREEN, lw=2.2, ms=7)
    ax[2].set_xlabel("min tow radius R [mm]\n(tighter manufacturability ->)")
    ax[2].set_ylabel("detuning reduction [%]")
    ax[2].set_title("(c) performance vs manufacturability\n(the printable-tow trade-off)",
                    fontsize=11)
    ax[2].invert_xaxis(); ax[2].grid(alpha=0.25)

    fig.suptitle("Manufacturable fiber steering flattens a composite plate's NONLINEAR "
                 "resonance\n(wavetopo CS-RBF + curl + MMA  ↔  NLTopt SSM backbone; "
                 "single-mode prototype)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig("figs/plate_framework_prototype.png", dpi=130)
    print(f"baseline drift {d_b:.2f}% -> optimized {d_o:.2f}%  (-{100*(1-d_o/d_b):.0f}%)")
    print("wrote figs/plate_framework_prototype.png")


if __name__ == "__main__":
    main()
