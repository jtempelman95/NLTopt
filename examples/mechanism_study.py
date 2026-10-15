r"""Mechanism, anisotropy dose-response and manufacturability front of the steered
plate (held frequency, fixed mass), all on the adapted-mode relaxed evaluator.

(1) Local Fourier structure.  At every point the bending and relaxed-stretch
    energy densities of a single-ply laminate rotated to theta are
        e_K(x,theta) = k0 + k2 cos 2(theta-phi_K) + k4 cos 4(theta-phi_K),
        e_G(x,theta) = g0 + g2 cos 2(theta-phi_G) + g4 cos 4(theta-phi_G),
    with phi_K, phi_G the principal directions of the curvature and of the
    relaxed membrane strain and (k2, k4), (g2, g4) proportional to U2, U3.  We
    extract these numerically (5-term fit, exact) for the UD and optimised fields.
(2) Pointwise-optimality prediction.  Freezing the UD fields, the Lagrangian of
    min gamma/omega^2 s.t. omega = omega* separates pointwise:
        theta*(x) = argmin_theta [ e_G(x,theta)/gamma_0 - lam e_K(x,theta)/omega_0^2 ],
    lam fixed by int e_K(theta*) = omega*^2.  We evaluate the true J of theta*.
(3) Anisotropy dose-response: Q(s) family, s in [0, 1.1] (E1/E2 = 1 ... 32).
(4) Manufacturability front: R_min sweep at held frequency.

    python examples/mechanism_study.py      # -> results/mechanism.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np

sys.path.insert(0, (_ROOT + "/examples"))
import levers_study as LS
from nltopt.relaxed_plate import SteeredPlate, rotate, Q_family
from nltopt.laminate import lamina_Q

P = LS.PLATE
H = P.h
Q = lamina_Q()
TH5 = np.linspace(0, np.pi, 5, endpoint=False)


def fourier_fields(strain, C):
    """Fit e(theta) = c0 + c2c cos2t + c2s sin2t + c4c cos4t + c4s sin4t at every
    point, e = strain^T C(theta) strain (strain (3,Nq), C a 3x3 reference)."""
    E = np.stack([np.einsum("aq,ab,bq->q", strain, rotate(C, t), strain) for t in TH5])
    B = np.stack([np.ones(5), np.cos(2 * TH5), np.sin(2 * TH5), np.cos(4 * TH5), np.sin(4 * TH5)], 1)
    c = np.linalg.solve(B, E)                                    # (5, Nq), exact
    a2 = np.hypot(c[1], c[2]); phi2 = 0.5 * np.arctan2(c[2], c[1])
    a4 = np.hypot(c[3], c[4])
    return dict(c=c, a2=a2, phi2=phi2, a4=a4)


def eval_fourier(c, t):
    return c[0] + c[1] * np.cos(2 * t) + c[2] * np.sin(2 * t) + c[3] * np.cos(4 * t) + c[4] * np.sin(4 * t)


def pointwise_design(st0, omega_target, n_grid=361):
    """theta*(x) from frozen fields (no smoothness, no curl) with the frequency
    multiplier lam found by bisection."""
    fk = fourier_fields(st0["kv"], H ** 3 / 12 * Q)
    fg = fourier_fields(st0["eps_r"], 0.5 * H * Q)
    tg = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    eK = np.stack([eval_fourier(fk["c"], t) for t in tg])        # (n_grid, Nq)
    eG = np.stack([eval_fourier(fg["c"], t) for t in tg])
    g0, w0 = st0["gamma"], st0["om2"]

    def design(lam):
        idx = np.argmin(eG / g0 - lam * eK / w0, axis=0)
        th = tg[idx]
        om2 = np.sum(P.w * eK[idx, np.arange(P.Nq)])
        return th, om2

    lo, hi = -5.0, 5.0
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        _, om2 = design(lam)
        if om2 > omega_target ** 2:
            hi = lam
        else:
            lo = lam
    th, _ = design(0.5 * (lo + hi))
    return th, fk, fg


def main():
    sp = SteeredPlate(P)
    UD = sp.forward(0.0)
    w_ud, J_ud = UD["omega1"], UD["J"]
    out = dict(w_ud=w_ud, J_ud=J_ud)

    # ---- held-frequency optimum (R = 20 mm) and its fields --------------------
    t0 = time.time()
    opt = LS.optimise_steered(w_ud)
    st = sp.forward(LS.PHI @ opt["theta_hat"])
    print(f"[opt] J/J_UD={st['J']/J_ud:.3f} f/f_UD={st['omega1']/w_ud:.4f} "
          f"({time.time()-t0:.0f}s)")
    out.update(theta_hat=opt["theta_hat"], theta_opt=st["theta"])
    for tag, s in (("ud", UD), ("opt", st)):
        fk = fourier_fields(s["kv"], H ** 3 / 12 * Q)
        fg = fourier_fields(s["eps_r"], 0.5 * H * Q)
        out.update({f"{tag}_eK": s["eK"], f"{tag}_eG": s["eG"],
                    f"{tag}_phiK": fk["phi2"], f"{tag}_aK2": fk["a2"], f"{tag}_aK4": fk["a4"],
                    f"{tag}_phiG": fg["phi2"], f"{tag}_aG2": fg["a2"], f"{tag}_aG4": fg["a4"],
                    f"{tag}_knl": s["knl"], f"{tag}_relax": s["relax"], f"{tag}_gamma": s["gamma"],
                    f"{tag}_om2": s["om2"], f"{tag}_mid": s["mid"], f"{tag}_J": s["J"]})

    # ---- pointwise-optimality prediction from the UD fields -------------------
    th_star, fk, fg = pointwise_design(UD, w_ud)
    sstar = sp.forward(th_star)
    print(f"[pointwise theta* from UD fields] J/J_UD={sstar['J']/J_ud:.3f} "
          f"f/f_UD={sstar['omega1']/w_ud:.3f}")
    out.update(theta_star=th_star, star_J=sstar["J"], star_f=sstar["omega1"])
    # angular agreement between theta* and the smooth optimum, energy-weighted
    dphi = np.angle(np.exp(2j * (th_star - st["theta"]))) / 2
    wgt = UD["eK"] / UD["eK"].sum() + UD["eG"] / UD["eG"].sum()
    out["star_vs_opt_deg"] = np.rad2deg(np.sqrt(np.sum(wgt * dphi ** 2) / wgt.sum()))
    print(f"   energy-weighted rms |theta* - theta_opt| = {out['star_vs_opt_deg']:.1f} deg")

    # ---- manufacturability front ----------------------------------------------
    Rs = np.array([8, 12, 20, 30, 50, 80, 150, 400.0]) / 1000.0
    front = []
    for R in Rs:
        r = LS.optimise_steered(w_ud, R_min=R)
        front.append(np.nan if r is None else r["J"] / J_ud)
        print(f"[front] R_min={1000*R:5.0f} mm  J/J_UD={front[-1]:.3f}")
    out.update(front_R=Rs, front_J=np.array(front))

    # ---- anisotropy dose-response (held frequency of each material's own UD) ---
    S_list = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.1])
    dose = []
    for s in S_list:
        Qs = Q_family(s)
        sps = SteeredPlate(P, D0=H ** 3 / 12 * Qs, A0=H * Qs)
        u = sps.forward(0.0, full=False)
        # best uniform angle at the held frequency: scan theta for f = f_UD(s)
        tt = np.linspace(0, np.pi / 2, 91)
        fu = np.array([sps.forward(t, full=False)["omega1"] for t in tt])
        r = LS.optimise_steered(u["omega1"], D0=H ** 3 / 12 * Qs, A0=H * Qs)
        J_opt = np.nan if r is None else r["J"] / u["J"]
        dose.append((s, Qs[0, 0] / Qs[1, 1], J_opt, fu.min() / u["omega1"], fu.max() / u["omega1"]))
        print(f"[dose] s={s:.2f} E1/E2~{Qs[0,0]/Qs[1,1]:5.1f}  steered J/J_UD={J_opt:.3f}  "
              f"uniform f-range [{fu.min()/u['omega1']:.3f},{fu.max()/u['omega1']:.3f}] f_UD")
    out["dose"] = np.array(dose)
    np.savez("results/mechanism.npz", **out)
    print("wrote results/mechanism.npz")


if __name__ == "__main__":
    main()
