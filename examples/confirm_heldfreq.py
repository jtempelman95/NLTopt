r"""Independent confirmation of the held-frequency steered design, on the full
multi-mode von Karman plate (nltopt.plate.VonKarmanPlate: 36 transverse + 72
in-plane modes, Gauss quadrature, in-plane INERTIA retained), three ways:

  (1) cubic SSM of the full system (all in-plane dynamics slaved by the order-2
      homological equations -- no static-condensation assumption);
  (2) full-order EPMC nonlinear-mode continuation (harmonic balance, H = 5 here; the EPMC backbones
      stored in results/confirm.npz and quoted in the paper are re-computed and merged by
      examples/epmc_check.py, default H = 3 -- run it after this script);
  (3) direct time integration of harmonic up/down frequency sweeps (2% Rayleigh
      damping) on the in-plane-condensed system projected on the lowest r linear
      modes, whose exactly-cubic force is assembled by polarisation -- no
      invariant-manifold or harmonic assumption (r = 6, checked against r = 10).

Every amplitude is the TRUE mid-plate deflection: the transverse coordinates are
changed so that one DOF equals w(a/2, b/2) = c^T q (c_k = phi_k(centre)).

Also: the jump-free amplitude a_c (numerical saddle-node onset of the damped SSM
forced response) and the up/down hysteresis gap at a drive between the two
designs' thresholds.

    python examples/confirm_heldfreq.py        # -> results/confirm.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, (_WAVETOPO + ""))
sys.path.insert(0, (_ROOT + "/examples"))
from wavetopo.cfrp import CSRBFMapping
from design_source import heldfreq_design
from nltopt.plate import VonKarmanPlate
from nltopt import ssm_from_system
from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import GeneralNonlinearElement
from pyhb.nonlinear_modes import nonlinear_mode

A, B, H = 0.20, 0.15, 0.002
NM, NIN, NQ = 36, 6, 32
ZETA = 0.02


class _Duck:
    pass


def theta_on_plate(pl, theta_hat):
    xs = np.linspace(0, A, 5); ys = np.linspace(0, B, 4)
    SX, SY = np.meshgrid(xs, ys)
    m = _Duck(); m.cx, m.cy, m.N = pl.qx, pl.qy, pl.qx.size
    return CSRBFMapping(m, (SX.ravel(), SY.ravel()), r_s=0.075).Phi @ theta_hat


class _Transformed(GeneralNonlinearElement):
    """Nonlinear element in coordinates q' = S q (S invertible)."""

    def __init__(self, inner, Sinv):
        self.inner, self.Sinv = inner, Sinv
        self.n = Sinv.shape[0]

    def forces(self, q_time, u_time):
        q = q_time @ self.Sinv.T
        f, Kt, Ct = self.inner.forces(q, None)
        return (f @ self.Sinv, np.einsum("ji,sjk,kl->sil", self.Sinv, Kt, self.Sinv),
                np.zeros_like(Kt))


def build(theta_hat):
    """Full plate in coordinates whose DOF `ref` is the mid-plate deflection."""
    pl = VonKarmanPlate(A, B, H, n_modes=NM, n_in=NIN, nq=NQ)
    pl.set_theta(theta_on_plate(pl, theta_hat))
    ref = pl.midspan_mode()
    c = np.zeros(pl.ndof)
    c[:pl.nw] = [np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2) for (m, n) in pl.w_modes]
    S = np.eye(pl.ndof); S[ref] = c
    Sinv = np.linalg.inv(S)
    M = Sinv.T @ pl.M @ Sinv
    K = Sinv.T @ pl.K @ Sinv
    elem = _Transformed(pl._force, Sinv)
    return pl, ref, M, K, elem, Sinv


def ssm_backbone(theta_hat, damped=False):
    pl, ref, M, K, elem, _ = build(theta_hat)
    w1 = np.sqrt(eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])[0])
    C = 2 * ZETA * w1 * M if damped else np.zeros_like(M)
    ssm = ssm_from_system(MechanicalSystem(M, C, K, [elem]), mode=0, ref_dof=ref, order=3)
    return ssm, ref, M, K, elem


def small_amp_beta(amp, om, w1, amax=0.1):
    sel = (amp / H < amax) & (amp > 0)
    return np.polyfit((amp[sel] / H) ** 2, om[sel] / w1 - 1, 1)[0]


def task_ssm(theta_hat):
    ssm, ref, *_ = ssm_backbone(theta_hat)
    rho = np.linspace(1e-6, 1, 400) * 0.5 * H / abs(ssm.phi[ref])
    amp = ssm.physical_amplitude(rho, dof=ref); om = ssm._reduced(rho).imag
    return dict(w1=ssm.omega0, amp=amp, om=om, beta=small_amp_beta(amp, om, ssm.omega0))


def task_epmc(theta_hat):
    t0 = time.time()
    pl, ref, M, K, elem, _ = build(theta_hat)
    sysm = MechanicalSystem(M, np.zeros_like(M), K, [elem])
    ssm = ssm_from_system(sysm, mode=0, ref_dof=ref, order=3)
    a_end = np.log10(1.1 * 0.45 * H / abs(ssm.phi[ref]))
    nm = nonlinear_mode(sysm, H=5, mode_index=0, ref_dof=ref, log_a_start=a_end - 2.0,
                        log_a_end=a_end, ds=0.1, ds_max=0.15, max_steps=400)
    Ae, Oe = nm["phys_amplitude"], nm["omega"]
    o = np.argsort(Ae)
    return dict(amp=Ae[o], om=Oe[o], w1=ssm.omega0, beta=small_amp_beta(Ae[o], Oe[o], ssm.omega0),
                seconds=time.time() - t0)


# ---------------------------------------------------------------- time domain
def modal_cubic_model(theta_hat, r):
    """In-plane-condensed transverse system projected on the lowest r linear modes;
    the condensed force is exactly cubic, assembled by polarisation."""
    pl, ref, M, K, elem, Sinv = build(theta_hat)
    nw = pl.nw
    sw, si = slice(0, nw), slice(nw, pl.ndof)
    Kin = pl._Kmem[si, si]
    force = pl._force
    qfull = np.zeros(pl.ndof)

    def f_relaxed(qw):
        qfull[:] = 0.0; qfull[sw] = qw
        drive = force._internal(qfull)[0][si]
        qfull[si] = -np.linalg.solve(Kin, drive)
        return force._internal(qfull)[0][sw]

    lam, V = eigh(pl.K[sw, sw], pl.M[sw, sw])
    Phi = V[:, :r]                                       # mass-normalised
    cvec = np.array([np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2) for (m, n) in pl.w_modes])
    C3 = lambda eta: Phi.T @ f_relaxed(Phi @ eta)
    E = np.eye(r)
    T = np.zeros((r, r, r, r))
    for i in range(r):
        for j in range(i, r):
            for k in range(j, r):
                a, b, c = E[i], E[j], E[k]
                t = (C3(a + b + c) - C3(a + b) - C3(a + c) - C3(b + c)
                     + C3(a) + C3(b) + C3(c)) / 6.0
                for p in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
                    T[:, p[0], p[1], p[2]] = t
    return dict(w2=lam[:r], T=T, mid=Phi.T @ cvec, cmid=cvec @ Phi, Phi=Phi, ref=ref,
                w1=np.sqrt(lam[0]))


def sweep(model, Fmid, om_frac, up=True, n_om=36, periods=300):
    """Up or down harmonic sweep, force Fmid * c (a load distributed like the
    mid-plate functional, i.e. a point force at the centre), continuation."""
    w2, T, w1 = model["w2"], model["T"], model["w1"]
    r = w2.size
    f_mod = Fmid * model["mid"]
    oms = w1 * np.linspace(om_frac[0], om_frac[1], n_om)
    if not up:
        oms = oms[::-1]
    y = np.zeros(2 * r); amp = np.empty(n_om)
    for i, Om in enumerate(oms):
        def rhs(t, yy):
            q, v = yy[:r], yy[r:]
            fnl = np.einsum("ijkl,j,k,l->i", T, q, q, q)
            return np.concatenate([v, f_mod * np.cos(Om * t) - 2 * ZETA * w1 * v - w2 * q - fnl])
        Tp = 2 * np.pi / Om; ppw = 40
        te = np.linspace((periods - 32) * Tp, periods * Tp, 32 * ppw)
        sol = solve_ivp(rhs, (0, periods * Tp), y, t_eval=te, method="RK45", rtol=1e-8, atol=1e-13)
        wm = model["cmid"] @ sol.y[:r]
        def fund(sl):
            return np.hypot(2 * np.mean(wm[sl] * np.cos(Om * sol.t[sl])),
                            2 * np.mean(wm[sl] * np.sin(Om * sol.t[sl])))
        a_now, a_ref = fund(slice(24 * ppw, None)), fund(slice(0, 8 * ppw))
        amp[i] = a_now if abs(a_now - a_ref) <= 0.004 * max(a_now, 1e-30) else np.nan
        y = sol.y[:, -1]
    o = np.argsort(oms)                                  # always return ascending in Omega
    return oms[o] / w1, amp[o]


def task_sweep(args):
    theta_hat, r, Fmid, om_frac, up = args
    return sweep(modal_cubic_model(theta_hat, r), Fmid, om_frac, up=up)


class _CubicT(GeneralNonlinearElement):
    def __init__(self, T):
        self.T = T; self.n = T.shape[0]

    def forces(self, q_time, u_time):
        f = np.einsum("ijkl,sj,sk,sl->si", self.T, q_time, q_time, q_time)
        Kt = 3.0 * np.einsum("ijkl,sk,sl->sij", self.T, q_time, q_time)
        return f, Kt, np.zeros_like(Kt)


def free_backbone_modal(theta_hat, r):
    """Small-amplitude mid-plate detuning coefficient of the r-mode condensed cubic
    model (its own cubic SSM) -- the truncation check of the time-domain model."""
    m = modal_cubic_model(theta_hat, r)
    sysm = MechanicalSystem(np.eye(r), np.zeros((r, r)), np.diag(m["w2"]), [_CubicT(m["T"])])
    ssm = ssm_from_system(sysm, mode=0, ref_dof=0, order=3)
    return ssm.gamma.imag / (4.0 * ssm.omega0 * (m["cmid"] @ ssm.phi) ** 2) * H ** 2


# ---------------------------------------------------------------- jump onset
def frc_multivalued(ssm, ref, Fmid, cvec):
    force = Fmid * cvec
    fr = ssm.forced_response(force, dof=ref, n=600)
    ok = np.isfinite(fr["omega_lo"])
    a = fr["amplitude"][ok]; ol, oh = fr["omega_lo"][ok], fr["omega_hi"][ok]
    curve_o = np.concatenate([ol, oh[::-1]])
    d = np.diff(curve_o)
    nturn = int(np.sum(np.diff(np.sign(d[np.abs(d) > 0])) != 0))
    return nturn >= 2, (a.max() if a.size else 0.0)


def jump_free_amplitude(theta_hat):
    ssm, ref, M, K, elem = ssm_backbone(theta_hat, damped=True)
    pl, _, _, _, _, Sinv = build(theta_hat)
    # a point load at the centre, written in the transformed coordinates
    cfull = np.zeros(pl.ndof)
    cfull[:pl.nw] = [np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2) for (m, n) in pl.w_modes]
    cvec = Sinv.T @ cfull
    kF = K[ref, ref] * H
    lo, hi = 1e-5 * kF, 0.05 * kF
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if frc_multivalued(ssm, ref, mid, cvec)[0]:
            hi = mid
        else:
            lo = mid
    return dict(F_c=hi, a_c=frc_multivalued(ssm, ref, hi, cvec)[1] / H, cvec=cvec)


def main():
    th_opt = heldfreq_design()
    designs = {"ud": np.zeros(20), "opt": th_opt}
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        f_ssm = {k: ex.submit(task_ssm, v) for k, v in designs.items()}
        no_epmc = "--no-epmc" in sys.argv
        f_epmc = {} if no_epmc else {k: ex.submit(task_epmc, v) for k, v in designs.items()}
        f_jump = {k: ex.submit(jump_free_amplitude, v) for k, v in designs.items()}
        res_ssm = {k: f.result() for k, f in f_ssm.items()}
        res_jump = {k: f.result() for k, f in f_jump.items()}
        print(f"[ssm+jump done {time.time()-t0:.0f}s]")
        for k in designs:
            s, j = res_ssm[k], res_jump[k]
            print(f"  {k}: f1={s['w1']/2/np.pi:.1f} Hz  beta_SSM={s['beta']:.4f}  a_c/h={j['a_c']:.3f}  F_c={j['F_c']:.3e}")
            out.update({f"{k}_ssm_w1": s["w1"], f"{k}_ssm_amp": s["amp"], f"{k}_ssm_om": s["om"],
                        f"{k}_ssm_beta": s["beta"], f"{k}_ac": j["a_c"], f"{k}_Fc": j["F_c"]})
        # a drive between the two thresholds: UD folds, steered does not
        Fdrive = np.sqrt(res_jump["ud"]["F_c"] * res_jump["opt"]["F_c"])
        out["Fdrive"] = Fdrive
        omr = (0.97, 1.12)
        f_sw = {(k, up, r): ex.submit(task_sweep, (v, r, Fdrive, omr, up))
                for k, v in designs.items() for up in (True, False) for r in (6,)}
        res_sw = {kk: f.result() for kk, f in f_sw.items()}
        for (k, up, r), (o, a) in res_sw.items():
            out[f"{k}_sweep_{'up' if up else 'down'}_o"] = o
            out[f"{k}_sweep_{'up' if up else 'down'}_a"] = a / H
        for k in designs:
            gap = np.nanmax(np.abs(out[f"{k}_sweep_up_a"] - out[f"{k}_sweep_down_a"]))
            out[f"{k}_hyst_gap"] = gap
            print(f"  {k}: up/down hysteresis gap = {gap:.3f} h  (drive between thresholds)")
        res_epmc = {k: f.result() for k, f in f_epmc.items()}
    for k in res_epmc:
        e = res_epmc[k]
        print(f"  {k}: EPMC beta={e['beta']:.4f}  ({e['seconds']:.0f}s)")
        out.update({f"{k}_epmc_amp": e["amp"], f"{k}_epmc_om": e["om"], f"{k}_epmc_beta": e["beta"],
                    f"{k}_epmc_w1": e["w1"], f"{k}_epmc_seconds": e["seconds"]})
    # truncation check of the time-domain model
    for k, v in designs.items():
        for r in (6, 10):
            out[f"{k}_modal_beta_r{r}"] = free_backbone_modal(v, r)
        print(f"  {k}: modal-model beta r=6 {out[f'{k}_modal_beta_r6']:.4f}  r=10 {out[f'{k}_modal_beta_r10']:.4f}")
    np.savez("results/confirm.npz", **out)
    print(f"wrote results/confirm.npz  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
