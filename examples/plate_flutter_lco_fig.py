r"""Post-flutter LCO figure: limit-cycle amplitude vs dynamic pressure for the
import os
uniform vs steered panel (the nonlinear flutter bifurcation), + time traces at a
flight condition between the two boundaries (baseline flutters, steered is quiet)."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys; sys.path.insert(0, (_WAVETOPO + ""))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
import plate_flutter as F
import plate_flutter_lco as L

rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                 "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
                 "font.family": "DejaVu Sans"})
UNI, STE, AMB, GRID = "#5b6470", "#12813f", "#b3760a", "#dcdfda"
UNI_TH = np.full(F.P.M, np.deg2rad(30))
STE_TH = np.load(os.path.join(os.path.dirname(__file__),"..","figs","flutter_steered_30.npy"))


def _lco_task(args):
    th_hat, q = args
    pl = L.build_nl(th_hat); G = F.aero_G(pl); T3 = L.relaxed_cubic_tensor(pl)
    a, f1, _ = L.lco(pl, G, q, T3, ic=0.12, periods=550)
    return q, a


def trace(th_hat, q, periods=260):
    pl = L.build_nl(th_hat); G = F.aero_G(pl); T3 = L.relaxed_cubic_tensor(pl)
    sw = pl._sw; Mw = np.array(pl.M[sw, sw]); Kw = np.array(pl.K[sw, sw]); Minv = np.linalg.inv(Mw)
    Cw = F.C0*Mw; Kae = Kw + F.BETA*q*G
    cen = np.array([np.sin(m*np.pi*0.5)*np.sin(n*np.pi*0.5) for (m, n) in pl.w_modes])
    w1 = np.sqrt(np.linalg.eigvalsh(np.linalg.solve(Mw, Kw)).min()); Tp = 2*np.pi/w1

    def rhs(t, yy):
        qw, vw = yy[:pl.nw], yy[pl.nw:]
        return np.concatenate([vw, -Minv@(Cw@vw + Kae@qw + np.einsum('ijkl,j,k,l->i', T3, qw, qw, qw))])
    y0 = np.zeros(2*pl.nw); y0[:pl.nw] = 0.02*F.H*np.ones(pl.nw)/np.sqrt(pl.nw)
    te = np.linspace(0, periods*Tp, periods*30)
    sol = solve_ivp(rhs, (0, periods*Tp), y0, t_eval=te, method="RK45", rtol=1e-7, atol=1e-10)
    return sol.t*1e3, (sol.y[:pl.nw].T @ cen)/F.H       # ms, a/h


if __name__ == "__main__":
    plu = L.build_nl(UNI_TH); qcu = F.qcr(plu, F.aero_G(plu))
    pls = L.build_nl(STE_TH); qcs = F.qcr(pls, F.aero_G(pls))
    print(f"q_cr: uniform {qcu/1e3:.0f} kPa, steered {qcs/1e3:.0f} kPa")
    facs = [0.95, 1.0, 1.06, 1.15, 1.28, 1.42]
    tasks = [(UNI_TH, f*qcu) for f in facs] + [(STE_TH, f*qcs) for f in facs]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_lco_task, tasks))
    nu = len(facs); uni = sorted(res[:nu]); ste = sorted(res[nu:])
    qu = np.array([r[0] for r in uni])/1e3; au = np.array([r[1] for r in uni])
    qs = np.array([r[0] for r in ste])/1e3; as_ = np.array([r[1] for r in ste])
    qop = 0.5*(qcu+qcs)                                  # flight condition between the two
    tu, wu = trace(UNI_TH, qop); ts, ws = trace(STE_TH, qop)

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for a in ax:
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in a.spines.values(): s.set_color("#9aa0a6")
    # (a) LCO bifurcation
    axa = ax[0]
    axa.axvspan(qcu/1e3, qcs/1e3, color=STE, alpha=0.08)
    axa.plot([qu.min(), qcu/1e3], [0, 0], "-", color=UNI, lw=2.4)
    axa.plot(qu[au > 1e-3], au[au > 1e-3], "-o", color=UNI, lw=2.4, ms=6, label="uniform")
    axa.plot([qs.min(), qcs/1e3], [0, 0], "-", color=STE, lw=2.4)
    axa.plot(qs[as_ > 1e-3], as_[as_ > 1e-3], "-o", color=STE, lw=2.4, ms=6, label="steered")
    axa.axvline(qcu/1e3, color=UNI, ls=":", lw=1.3); axa.axvline(qcs/1e3, color=STE, ls=":", lw=1.3)
    axa.axvline(qop/1e3, color=AMB, ls="--", lw=1.6)
    axa.text(qop/1e3, 0.60, "flight\ncondition", color=AMB, fontsize=8.5, ha="center", fontweight="bold")
    axa.text(0.5*(qcu+qcs)/1e3, 0.05, "uniform flutters,\nsteered quiet", color=STE, fontsize=8.6,
             ha="center", fontweight="bold")
    axa.set_xlabel("dynamic pressure  $q$  [kPa]"); axa.set_ylabel("limit-cycle amplitude  $a/h$")
    axa.set_title("(a)  post-flutter limit cycle vs dynamic pressure", loc="left", fontweight="bold")
    axa.legend(loc="upper left", frameon=False); axa.set_ylim(-0.03, 0.7)
    # (b) time traces at the flight condition
    axb = ax[1]
    axb.plot(tu, wu, color=UNI, lw=1.2, label=f"uniform ($q_{{cr}}$={qcu/1e3:.0f})")
    axb.plot(ts, ws, color=STE, lw=1.5, label=f"steered ($q_{{cr}}$={qcs/1e3:.0f})")
    axb.set_xlabel("time  [ms]"); axb.set_ylabel("centre deflection  $a/h$")
    axb.set_title(f"(b)  response at $q$={qop/1e3:.0f} kPa (between the boundaries)", loc="left", fontweight="bold")
    axb.legend(loc="upper left", frameon=False)
    axb.text(0.97, 0.06, "steering removes the flutter", transform=axb.transAxes, ha="right",
             color=STE, fontsize=9, fontweight="bold")

    fig.suptitle("Steering the tows does not just delay flutter --- it keeps the panel quiet where the "
                 "baseline is in limit-cycle flutter", fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig((_ROOT + "/figs/plate_flutter_lco.png"), dpi=150)
    print(f"wrote figs/plate_flutter_lco.png; uniform LCO@qop={np.interp(qop/1e3,qu,au):.2f}, steered=0")
