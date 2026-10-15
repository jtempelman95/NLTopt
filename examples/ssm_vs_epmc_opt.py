r"""FRAMEWORK COMPARISON on the Pozzi et al. (2025, Fig. 3) benchmark: shape a
clamped--clamped von Karman beam's first-mode backbone from HARDENING to a
prescribed SOFTENING by adding initial curvature -- optimized once with the SSM
adjoint (cheap, closed-form) and once with full-order EPMC (harmonic-balance
nonlinear-mode continuation), and compared.

Mechanism (Pozzi's): a straight beam only hardens; an initial curvature
``y = A1 sin(pi x/L)`` injects a quadratic nonlinearity whose slaved response
SOFTENS the first-mode backbone (the SSM cubic coefficient goes negative).  The
curvature also stiffens the beam linearly, so the meaningful, curvature-invariant
measure is the backbone RELATIVE to each design's OWN linear frequency
``omega1``.  We design ``A1`` so the relative detuning at RMS = 0.2 h0 hits a
softening target, and check that both frameworks land on the same A1.

    python examples/ssm_vs_epmc_opt.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
sys.path.insert(0, (_ROOT + "/examples"))
import numpy as np, time
import paperstyle as ps
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pyhb.nonlinear_modes import nonlinear_mode
from nltopt import ssm_from_system
from nltopt.beam import curved_vk_beam

H0, L0, E, RHO, NEL = 0.010, 1.0, 90e9, 7850.0, 10
CAMP = 0.20                                                   # target amplitude (RMS / h0)
TARGET = -0.015                                              # softening: Omega(0.2h0)/omega1 - 1
BOUNDS = (0.002, 0.020)                                      # A1 in [2, 20] mm
_nev = {"ssm": 0, "epmc": 0}


def _beam(A1):
    d = curved_vk_beam(float(A1), 0.0, H0, L0, E=E, rho=RHO, n_elem=NEL)
    return d, ssm_from_system(d["system"], mode=0, ref_dof=d["ref"])


def ssm_delta(A1):
    _nev["ssm"] += 1
    d, s = _beam(A1)
    o, _ = s.omega_at_amplitude(CAMP * H0, kind="rms")
    return o / d["omega1"] - 1.0


def epmc_delta(A1, H=5):
    _nev["epmc"] += 1
    d, s = _beam(A1); w1 = d["omega1"]; ref = d["ref"]; phi = abs(s.phi[ref])
    tr = CAMP * H0
    a_end = np.log10(1.35 * tr * np.sqrt(2) / phi)
    nm = nonlinear_mode(d["system"], H=H, mode_index=0, ref_dof=ref,
                        log_a_start=a_end - 1.5, log_a_end=a_end,
                        ds=0.05, ds_max=0.11, max_steps=300)
    rms = np.asarray(nm["phys_amplitude"]) / np.sqrt(2.0)
    om = np.asarray(nm["omega"]) / w1
    o = np.argsort(rms)
    return float(np.interp(tr, rms[o], om[o])) - 1.0


def optimize(delta_fn, key):
    _nev[key] = 0; t0 = time.time()
    r = minimize_scalar(lambda A1: (delta_fn(A1) - TARGET) ** 2,
                        bounds=BOUNDS, method="bounded", options={"xatol": 5e-6})
    return float(r.x), time.time() - t0, _nev[key]


def main():
    dstraight = ssm_delta(BOUNDS[0]); _nev["ssm"] = 0
    d0, s0 = _beam(1e-6)
    print(f"straight beam: f1={d0['omega1']/2/np.pi:.1f} Hz, "
          f"relative detuning at 0.2h0 = {ssm_delta(1e-6)*100:+.2f}%  (hardening)")
    print(f"target: soften to {TARGET*100:+.1f}% at 0.2h0\n"); _nev["ssm"] = 0

    aS, tS, nS = optimize(ssm_delta, "ssm")
    dS = ssm_delta(aS)
    print(f"[SSM ] A1={aS*1e3:6.3f} mm  ->  detuning {dS*100:+.3f}%  "
          f"({nS} backbone evals, {tS:.2f}s)")
    aE, tE, nE = optimize(epmc_delta, "epmc")
    dE = epmc_delta(aE)
    print(f"[EPMC] A1={aE*1e3:6.3f} mm  ->  detuning {dE*100:+.3f}%  "
          f"({nE} backbone evals, {tE:.1f}s)")

    print(f"\nAGREEMENT: A1_SSM={aS*1e3:.3f} vs A1_EPMC={aE*1e3:.3f} mm  "
          f"(|diff| {abs(aS-aE)*1e3:.3f} mm)")
    print(f"COST: SSM {tS:.2f}s/{nS} evals vs EPMC {tE:.1f}s/{nE} evals  "
          f"-> ~{(tE/nE)/(tS/max(nS,1)):.0f}x per backbone")

    # ---- figure: hardening straight -> softening optimized backbones --------
    def ssm_curve(A1, amax=0.30):
        d, s = _beam(max(A1, 1e-6))
        rho = np.linspace(1e-7, 4*amax*H0/abs(s.phi[s.ref_dof]), 300)
        rms = s.rms_amplitude(rho); om = (s.lambda1.imag + s.gamma.imag*rho**2/4)/d["omega1"]
        k = rms <= amax*H0; return om[k], rms[k]/H0

    def epmc_curve(A1, amax=0.30, H=5):
        d, s = _beam(A1); w1 = d["omega1"]; ref = d["ref"]
        a_end = np.log10(1.2*amax*H0*np.sqrt(2)/abs(s.phi[ref]))
        nm = nonlinear_mode(d["system"], H=H, mode_index=0, ref_dof=ref,
                            log_a_start=a_end-2.0, log_a_end=a_end, ds=0.05, ds_max=0.11, max_steps=400)
        rms = np.asarray(nm["phys_amplitude"])/np.sqrt(2); om = np.asarray(nm["omega"])/w1
        o = np.argsort(rms); k = rms[o] <= amax*H0
        return om[o][k], rms[o][k]/H0

    oi, ai = ssm_curve(1e-6); oS, aS_c = ssm_curve(aS); oE_s, aE_s = ssm_curve(aE)
    oE, aE_c = epmc_curve(aE)
    fig, a0 = plt.subplots(figsize=(3.45, 2.45))
    a0.plot(oi, ai, color=ps.GREY, lw=1.5, label="straight (hardening)")
    a0.plot(oS, aS_c, color=ps.GREEN, lw=1.7, label="SSM-optimized")
    a0.plot(oE, aE_c, "--", color=ps.RED, lw=1.3, label="EPMC-optimized")
    a0.plot(1.0 + TARGET, CAMP, "o", color=ps.INK, ms=5.5, mfc="none", mew=1.3, zorder=5, label="target")
    a0.set_xlabel(r"$\Omega/\omega_1$ (each design's own $\omega_1$)"); a0.set_ylabel(r"RMS mid-span $v/h_0$")
    ps.grid(a0); ps.despine(a0); ps.panel(a0, "a")
    a0.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.53, 1.0), fontsize=7, handlelength=1.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    ps.save(fig, (_ROOT + "/figs/ssm_vs_epmc_opt"))
    ratio = int(round((round(tE, 1) / nE) / (round(tS, 3) / max(nS, 1)), -1))    # from the displayed values
    with open((_ROOT + "/docs/paper/svse_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}lrr@{}}\n\\toprule\n & SSM & EPMC\\\\ \\midrule\n")
        fh.write(f"optimal $A_1$ (mm) & {aS*1e3:.2f} & {aE*1e3:.2f}\\\\\n")
        fh.write(f"detuning at $0.2h_0$ & ${dS*100:+.2f}\\%$ & ${dE*100:+.2f}\\%$\\\\\n".replace("+", "{+}").replace("-", "$-$").replace("$$-$", "$-"))
        fh.write(f"backbone evaluations & {nS} & {nE}\\\\\n")
        fh.write(f"wall time (s) & {tS:.3f} & {tE:.1f}\\\\\n")
        fh.write(f"cost per backbone & $1\\times$ & $\\approx{ratio}\\times$\\\\\n\\bottomrule\n\\end{{tabular}}\n")
    print("wrote docs/paper/svse_table.tex")


if __name__ == "__main__":
    main()
