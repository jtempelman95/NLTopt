r"""Generate docs/paper/numbers.tex: EVERY number quoted in the manuscript, read from
the results files, so text, tables and figures cannot disagree.

    python examples/paper_numbers.py
"""
import os
import numpy as np

R = "results"
OUT = "docs/paper/numbers.tex"
macros = {}


def put(name, val, fmt="{:.2f}"):
    macros[name] = fmt.format(val) if not isinstance(val, str) else val


def load(f):
    p = os.path.join(R, f)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def pct(x):           # reduction in percent from a ratio
    return 100.0 * (1.0 - x)


# ---------------------------------------------------------------- plate: levers
L = load("levers.npz")
if L is not None:
    w_ud, J_ud = float(L["w_ud"]), float(L["J_ud"])
    put("nUDf", w_ud / 2 / np.pi, "{:.1f}")
    h = 0.002
    put("nEvBetaUD", 0.375 * J_ud * h ** 2, "{:.3f}")
    put("nUDdet", 100 * 0.375 * J_ud * h ** 2 * 0.16, "{:.2f}")
    for d in L["F"]:
        if d is not None and abs(d["omega1"] / w_ud - 1) < 1e-3:
            put("nEvBetaOpt", 0.375 * d["J"] * h ** 2, "{:.3f}")
    fU, JU = L["U_f"] / w_ud, L["U_J"] / J_ud
    k = int(np.argmin(JU))
    put("nUfmax", fU.max()); put("nUJmin", JU[k]); put("nUthJmin", np.rad2deg(L["U_theta"][k]), "{:.0f}")
    put("nUfatJmin", fU[k])
    S = L["S"]; fS, JS = S[:, 4] / w_ud, S[:, 5] / J_ud
    put("nSnum", f"{len(S):,}".replace(",", "{,}"))
    put("nSfmin", fS.min(), "{:.3f}")
    m = np.abs(fS - 1) < 0.01
    put("nSheld", JS[m].min(), "{:.3f}")
    kg = int(np.argmin(JS)); put("nSmin", JS[kg]); put("nSfatmin", fS[kg])
    put("nSminname", "[" + "/".join(f"{a:+.0f}".replace("+", "") for a in np.rad2deg(S[kg, :4][::-1])) + "]_s", "{}")
    tg = L["targets"] / w_ud
    names = {0.9: "Nine", 1.0: "Ten", 1.1: "Eleven", 1.2: "Twelve", 1.3: "Thirteen"}
    for t, d, ds in zip(tg, L["F"], L["SF"]):
        key = names[round(float(t), 2)]
        mm = np.abs(fS - t) < 0.01
        if mm.any():
            put(f"nS{key}", JS[mm].min())
        if d is not None:
            put(f"nF{key}", d["J"] / J_ud, "{:.3f}" if key == "Ten" else "{:.2f}")
            put(f"nF{key}Red", pct(d["J"] / J_ud), "{:.0f}")
        if ds is not None:
            put(f"nSF{key}", ds["J"] / J_ud)
            put(f"nSF{key}stack", "[" + "/".join(f"{a:.0f}" for a in np.rad2deg(ds["stack"][::-1])) + "]_s", "{}")

# ---------------------------------------------------------------- continuous-stack competitor
SC = load("stack_cont.npz")
if SC is not None:
    for t, Jc, Js, Jg in zip(SC["targets"], SC["J_cont"], SC["J_single"], SC["J_grid"]):
        key = {1.0: "Ten", 1.1: "Eleven", 1.2: "Twelve", 1.3: "Thirteen"}[round(float(t), 2)]
        put(f"nCS{key}", min(Jc, Js))                      # best uniform: continuous stack or single angle
        put(f"nG{key}", Jg)                                # 15-degree grid envelope
        put(f"nU{key}", Js)                                # continuous single angle

# ---------------------------------------------------------------- mechanism
M = load("mechanism.npz")
if M is not None:
    for k, nm in (("knl", "knl"), ("relax", "relax"), ("gamma", "gamma"), ("om2", "omsq"),
                  ("mid", "mid"), ("J", "J")):
        put(f"nD{nm}", float(M[f"opt_{k}"]) / float(M[f"ud_{k}"]))
    put("nRelaxUD", 100 * float(M["ud_relax"]) / float(M["ud_knl"]), "{:.0f}")
    put("nRelaxOpt", 100 * float(M["opt_relax"]) / float(M["opt_knl"]), "{:.0f}")
    put("nStarJ", float(M["star_J"]) / float(M["J_ud"]))
    put("nStarF", float(M["star_f"]) / float(M["w_ud"]), "{:.3f}")
    put("nStarDev", float(M["star_vs_opt_deg"]), "{:.0f}")
    # P4 detail: angle difference where the energies are large, and at the energy peak
    _w = M["ud_eK"] / M["ud_eK"].sum() + M["ud_eG"] / M["ud_eG"].sum(); _wn = _w / _w.max()
    _d = np.rad2deg(np.abs(np.angle(np.exp(2j * (M["theta_opt"] - M["theta_star"])))) / 2)
    _hi = _wn > 0.5
    put("nStarDevHigh", float(np.sqrt(np.sum(_w[_hi] * _d[_hi] ** 2) / _w[_hi].sum())), "{:.0f}")
    _k = int(np.argmax(M["ud_eK"]))
    put("nOptAtPeak", float(np.rad2deg(np.angle(np.exp(2j * M["theta_opt"][_k])) / 2)), "{:.0f}")
    from nltopt.relaxed_plate import RelaxedPlate
    P = RelaxedPlate()
    eK, eG = M["ud_eK"], M["ud_eG"]
    # Lambda < 0 at the held frequency: the frequency-favored direction is ACROSS the curvature
    # (phi_K + 90), the stretch-favored one across the stretch (phi_G + 90) -> conflict = |phi_K - phi_G|
    conflict = np.rad2deg(np.abs(np.angle(np.exp(2j * (M["ud_phiK"] - M["ud_phiG"])))) / 2)
    ML = load("mechanism_lambda.npz")
    if ML is not None:
        put("nLambda", float(ML["lam"]), "{:.2f}"); put("nFreeFreq", float(ML["free_freq"]), "{:.2f}")
    wgt = eK / eK.sum() + eG / eG.sum()
    put("nConflict", float(np.sum(wgt * conflict) / wgt.sum()), "{:.0f}")
    rname = {8: "Eight", 12: "Twelve", 20: "Twenty", 30: "Thirty", 50: "Fifty", 80: "Eighty",
             150: "OneFifty", 400: "FourHundred"}
    for R_, J_ in zip(M["front_R"], M["front_J"]):
        put(f"nFront{rname[int(round(1000 * R_))]}", J_, "{:.2f}")
    if "dose" in M.files:
        for row in M["dose"]:
            s, ratio, Jopt = row[0], row[1], row[2]
            tag = {0.1: "A", 0.25: "B", 0.5: "C", 0.75: "D", 1.0: "E", 1.1: "F"}[round(float(s), 2)]
            put(f"nDose{tag}", Jopt); put(f"nDoseRatio{tag}", ratio, "{:.1f}")

C = load("ceiling.npz")
if C is not None:
    put("nCeil", float(C["ceiling_J"]), "{:.2f}")
    for k in C.files:
        if k.startswith("res_"):
            _, dx, r = k.split("_")
            put(f"nRes{ {'50': 'A', '33': 'B', '25': 'C'}[dx] }{ 'Inf' if r == 'inf' else 'Twenty' }",
                float(C[k]), "{:.3f}")

# ---------------------------------------------------------------- beam
Bm = load("beam_symmetric.npz")
if Bm is not None:
    put("nBUDf", float(Bm["base_w1"]) / 2 / np.pi, "{:.1f}")
    put("nBUDrms", 100 * float(Bm["base_rms"]), "{:.2f}")
    put("nBUnif", float(Bm["uniform45_ratio"]), "{:.4f}")
    for tag, name in (("k900_f98", "NineHeld"), ("k900_f90", "NineNinety"),
                      ("k1800_f98", "EighteenHeld"), ("k1800_f90", "EighteenNinety"),
                      ("k99999_f98", "InfHeld"), ("k99999_f90", "InfNinety")):
        if tag + "_coeff" in Bm.files:
            put(f"nB{name}", float(Bm[tag + "_coeff"]), "{:.3f}")
            put(f"nB{name}Red", pct(float(Bm[tag + "_coeff"])), "{:.0f}")
            put(f"nB{name}f", float(Bm[tag + "_f"]), "{:.3f}")
            put(f"nB{name}Cond", float(Bm[tag + "_Jcond"]), "{:.3f}")
Bl = load("results_steered_beam.npz")
if Bl is not None:
    put("nBBred", pct(float(Bl["drift1"]) / float(Bl["drift0"])), "{:.0f}")
    put("nBBf", float(Bl["f1_opt"]) / float(Bl["f1_base"]), "{:.2f}")

# ---------------------------------------------------------------- confirmation
Cf = load("confirm.npz")
if Cf is not None:
    for d in ("ud", "opt"):
        D = d.upper() if d == "ud" else "Opt"
        put(f"nC{D}f", float(Cf[f"{d}_ssm_w1"]) / 2 / np.pi, "{:.1f}")
        put(f"nC{D}betaSSM", float(Cf[f"{d}_ssm_beta"]), "{:.3f}")
        put(f"nC{D}ac", float(Cf[f"{d}_ac"]), "{:.3f}")
        if f"{d}_epmc_beta" in Cf.files:
            put(f"nC{D}betaEPMC", float(Cf[f"{d}_epmc_beta"]), "{:.3f}")
            put(f"nC{D}epmcSec", float(Cf[f"{d}_epmc_seconds"]), "{:.0f}")
        if f"{d}_sweep_up_o" in Cf.files:          # hysteresis gap from Omega-sorted sweeps
            ou, au = Cf[f"{d}_sweep_up_o"], Cf[f"{d}_sweep_up_a"]
            od, ad = Cf[f"{d}_sweep_down_o"], Cf[f"{d}_sweep_down_a"]
            iu, idn = np.argsort(ou), np.argsort(od)
            ou, au, od, ad = ou[iu], au[iu], od[idn], ad[idn]
            gap = np.nanmax(np.abs(au - np.interp(ou, od, ad)))
            put(f"nC{D}gap", gap, "{:.3f}")
        if f"{d}_modal_beta_r6" in Cf.files:
            put(f"nC{D}rsix", float(Cf[f"{d}_modal_beta_r6"]), "{:.3f}")
            put(f"nC{D}rten", float(Cf[f"{d}_modal_beta_r10"]), "{:.3f}")
    put("nEPMCH", str(int(Cf["opt_epmc_H"])) if "opt_epmc_H" in Cf.files else "5", "{}")
    rs = float(Cf["opt_ssm_beta"]) / float(Cf["ud_ssm_beta"])
    put("nCratioSSM", rs, "{:.3f}"); put("nCredSSM", pct(rs), "{:.0f}")
    if "opt_epmc_beta" in Cf.files:
        re_ = float(Cf["opt_epmc_beta"]) / float(Cf["ud_epmc_beta"])
        put("nCratioEPMC", re_, "{:.3f}"); put("nCredEPMC", pct(re_), "{:.0f}")
    put("nCacRatio", float(Cf["opt_ac"]) / float(Cf["ud_ac"]), "{:.2f}")
    FS = load("confirm_fine_sweep.npz")
    if FS is not None:
        gap = np.abs(FS["ud_up_a"] - FS["ud_down_a"])
        put("nFineSteps", len(FS["ud_up_a"]), "{:d}"); put("nFineGap", float(np.nanmax(gap)), "{:.3f}")
        put("nFineHyst", int(np.sum(gap > 0.01)), "{:d}")
        put("nFineOptHyst", int(np.sum(np.abs(FS["opt_up_a"] - FS["opt_down_a"]) > 0.01)), "{:d}")
    # Prop. 3 absolute prediction (peak amplitude), zeta = 2%, with the full-model SSM beta
    put("nCacFormUD", np.sqrt(8 * 0.02 / (3 * np.sqrt(3) * float(Cf["ud_ssm_beta"]))), "{:.3f}")
    put("nCacFormOpt", np.sqrt(8 * 0.02 / (3 * np.sqrt(3) * float(Cf["opt_ssm_beta"]))), "{:.3f}")
    put("nCacTheory", rs ** -0.5, "{:.2f}")
    put("nCFratio", float(Cf["opt_Fc"]) / float(Cf["ud_Fc"]), "{:.2f}")

Fe = load("fe_design.npz")
if Fe is not None:
    nx = max(int(k.split("_")[1]) for k in Fe.files if k.startswith("ud_"))
    put("nFEmesh", f"{nx}\\times{nx*3//4}", "{}")
    put("nFEUDf", float(Fe[f"ud_{nx}_f"]), "{:.1f}"); put("nFEOptf", float(Fe[f"opt_{nx}_f"]), "{:.1f}")
    put("nFEUDbeta", float(Fe[f"ud_{nx}_beta"]), "{:.3f}"); put("nFEOptbeta", float(Fe[f"opt_{nx}_beta"]), "{:.3f}")
    put("nFEratio", float(Fe[f"opt_{nx}_beta"]) / float(Fe[f"ud_{nx}_beta"]), "{:.3f}")

# any macro used in the manuscript but not (yet) computed is shown as ?? in bold
# ---------------------------------------------------------------- previously hand-typed results
from nltopt.laminate import Qbar as _Qb
for _t, _nm in ((20, "Twenty"), (45, "FortyFive")):           # uniform single-ply beam: f ~ sqrt(Qbar_11)
    put(f"nBUnifF{_nm}", float(np.sqrt(_Qb(np.deg2rad(_t))[0, 0] / _Qb(0.0)[0, 0])), "{:.3f}")
_CT = load("conv_table.npy")
if _CT is not None:
    _CT = np.asarray(_CT, float); _base = _CT[(_CT[:, 0] == 8) & (_CT[:, 1] == 6)][0]
    put("nConvThree", float(_CT[(_CT[:, 0] == 8) & (_CT[:, 1] == 3)][0][4]), "{:.2f}")
    put("nConvBase", float(_base[4]), "{:.2f}")
    put("nConvFreqDrop", 100 * (1 - _CT[:, 3].min() / _base[3]), "{:.1f}")
    _fe_dev = abs(1 - float(macros.get("nFEOptf", 0)) / float(macros.get("nFEUDf", 1))) if "nFEOptf" in macros else 0
    put("nHeldTol", 100 * max(1 - _CT[:, 3].min() / _base[2], _fe_dev), "{:.1f}")
_Bm2 = load("beam_symmetric.npz")
if _Bm2 is not None:
    put("nBeamFreqBand", 100 * (1 - min(float(_Bm2[f"{k}_f98_f"]) for k in ("k900", "k1800", "k99999"))), "{:.1f}")
_SC2 = load("stack_cont.npz"); _Lv = load("levers.npz")
if _SC2 is not None and _Lv is not None:
    _w, _J = float(_Lv["w_ud"]), float(_Lv["J_ud"])
    _best = np.minimum(_SC2["J_cont"], _SC2["J_single"]); _ratios = []
    for _t, _bu in zip(_SC2["targets"], _best):
        if _t > 1.01:
            _d = [d for d in _Lv["F"] if d is not None and abs(d["omega1"] / _w - _t) < 1e-3]
            if _d:
                _ratios.append(_d[0]["J"] / _J / _bu)
    put("nSteerVsUniformMax", 100 * max(_ratios), "{:.0f}")
    _S = _Lv["S"]; _fS, _JS = _S[:, 4] / _w, _S[:, 5] / _J; _m = np.abs(_fS - 1) < 0.01
    put("nGridStackDiff", 100 * (1 - _JS[_m].min()), "{:.1f}")
_Cf2 = load("confirm.npz")
if _Cf2 is not None:
    _errs = []
    for _d in ("ud", "opt"):
        _ae, _oe = _Cf2[_d + "_epmc_amp"] / 0.002, _Cf2[_d + "_epmc_om"] / float(_Cf2[_d + "_epmc_w1"])
        _a, _o = _Cf2[_d + "_ssm_amp"] / 0.002, _Cf2[_d + "_ssm_om"] / float(_Cf2[_d + "_ssm_w1"])
        _k = np.argsort(_a); _oi = np.interp(_ae, _a[_k], _o[_k]); _sel = (_ae > 0.05) & (_ae <= _a.max())
        _errs.append(np.max(np.abs((_oi[_sel] - 1) - (_oe[_sel] - 1)) / np.abs(_oe[_sel] - 1)))
    put("nEPMCAgree", 100 * max(_errs), "{:.0f}")
_FC = load("plate_fe_fig_cache.npz")
if _FC is not None:
    _rows = [("UD $[0]$ (homogeneous)", r"FE ($28\times21$)", _FC["fe_ud"], None),
             ("", "assumed-modes", _FC["mod_ud"][2], _FC["fe_ud"]),
             ("steered (strong)", r"FE ($28\times21$)", _FC["fe_st"], None),
             ("", "assumed, $8$ modes", _FC["mod_st"][2], _FC["fe_st"]),
             ("", "assumed, $24$ modes", _FC["mod_st"][5], _FC["fe_st"])]
    with open("docs/paper/fe_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{llccc}\n\\toprule\nfield & model & $f_1$ [Hz] & $\\gamma$ & detuning @ $a/h{=}0.4$\\\\\n\\midrule\n")
        for i, (fld, mdl, r, ref) in enumerate(_rows):
            if i == 2:
                fh.write("\\midrule\n")
            ex = int(np.floor(np.log10(r[1])))
            err = "" if ref is None else " \\;($" + f"{100*abs(r[3]/ref[3]-1):.0f}" + "\\%$)"
            fh.write(f"{fld} & {mdl} & ${r[0]:.1f}$ & ${r[1]/10**ex:.3f}\\!\\times\\!10^{{{ex}}}$ & ${16*r[3]:.2f}\\%${err}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    put("nFEtabUDdet", 16 * float(_FC["mod_ud"][2][3]), "{:.2f}")
    put("nFEselfUD", float(_FC["fe_ud"][0]), "{:.1f}")
    put("nFEtabUDagree", 100 * abs(float(_FC["mod_ud"][2][3]) / float(_FC["fe_ud"][3]) - 1), "{:.0f}")
    for _i, _nm in ((2, "Eight"), (5, "TwentyFour")):
        put(f"nFEtabFreqErr{_nm}", 100 * (float(_FC["mod_st"][_i][0]) / float(_FC["fe_st"][0]) - 1), "{:.0f}")
        put(f"nFEtabDetErr{_nm}", 100 * abs(float(_FC["mod_st"][_i][3]) / float(_FC["fe_st"][3]) - 1), "{:.0f}")

_pairs = [("nBNineHeldCond", "nBNineHeld"), ("nBNineNinetyCond", "nBNineNinety"), ("nBEighteenHeldCond", "nBEighteenHeld"),
          ("nBEighteenNinetyCond", "nBEighteenNinety"), ("nBInfHeldCond", "nBInfHeld"), ("nBInfNinetyCond", "nBInfNinety")]
if all(a in macros and b in macros for a, b in _pairs):
    put("nBeamCondFE", max(abs(float(macros[a]) - float(macros[b])) for a, b in _pairs), "{:.3f}")
if _Cf2 is not None and "nEvBetaUD" in macros:
    _L2 = load("levers.npz"); _Jud = float(_L2["J_ud"]); _bud = 0.375 * _Jud * 0.002 ** 2
    _bop = [0.375 * d["J"] * 0.002 ** 2 for d in _L2["F"] if d is not None and abs(d["omega1"] / float(_L2["w_ud"]) - 1) < 1e-3][0]
    put("nSSMvsEval", 100 * max(abs(float(_Cf2["ud_ssm_beta"]) / _bud - 1), abs(float(_Cf2["opt_ssm_beta"]) / _bop - 1)), "{:.1f}")

if _Cf2 is not None:
    put("nEPMCmaxUD", float(np.nanmax(_Cf2["ud_epmc_amp"])) / 0.002, "{:.2f}")
    put("nEPMCmaxOpt", float(np.nanmax(_Cf2["opt_epmc_amp"])) / 0.002, "{:.2f}")

_MR = load("mode_ratios.npz")
if _MR is not None:
    _r = _MR["opt_ratios"]; put("nModeRatios", "1{:}" + "{:}".join(f"{v:.2f}" for v in _r[1:]), "{}")
    put("nDetThreeOne", 100 * float(_MR["opt_det31"]), "{:.0f}")
    put("nFin", float(_MR["ud_w_in"]) / 2 / np.pi / 1000, "{:.1f}")
    _rho = 4 * float(_MR["ud_w1"]) ** 2 / float(_MR["ud_w_in"]) ** 2
    put("nRhoIn", f"{_rho*1e3:.1f}\\times10^{{-3}}", "{}")
from nltopt.laminate import lamina_Q as _lQ
_Q = _lQ(); _U2 = 0.5 * (_Q[0, 0] - _Q[1, 1]); _U3 = (_Q[0, 0] + _Q[1, 1] - 2 * _Q[0, 1] - 4 * _Q[2, 2]) / 8
put("nUtwoUthree", _U2 / _U3, "{:.1f}")
if _Bm2 is not None:
    put("nBeamMaxAngle", float(np.rad2deg(np.max(np.abs(_Bm2["k99999_f98_c"])))), "{:.0f}")

# ---------------------------------------------------------------- Revision R1
_M1 = load("r1_matched.npz")
if _M1 is not None:
    put("nRefEvalJ", float(_M1["ref_eval_J"]), "{:.3f}")
    put("nRefEvalF", float(_M1["ref_eval_f"]), "{:.3f}")
    put("nRefHeldJ", float(_M1["ref_held_J"]), "{:.3f}")
    put("nFEasisF", float(_M1["fe_asis_f"]), "{:.3f}")
    put("nFEasisJ", float(_M1["fe_asis_J"]), "{:.3f}")
    put("nFEmatchedJ", float(_M1["fe_matched_J"]), "{:.3f}")
    put("nFEslope", -float(_M1["fe_slope"]), "{:.2f}")
_GEN = {n: load(f"r1_generality_{n}.npz") for n in ("ss_43", "ss_11", "ss_21", "cl_43", "mv_43")}
if all(v is not None for v in _GEN.values()):
    _r = lambda n: float(_GEN[n]["J_steer"] / _GEN[n]["J_ud"])
    put("nGenBaseJ", _r("ss_43"), "{:.3f}")
    put("nGenSqJ", _r("ss_11"), "{:.2f}")
    put("nGenTwoJ", _r("ss_21"), "{:.2f}")
    put("nGenClJ", _r("cl_43"), "{:.2f}")
    put("nGenMvJ", _r("mv_43"), "{:.2f}")
    _cl = _GEN["cl_43"]
    put("nGenClUnif", min(float(_cl["J_stack"]), float(_cl["J_single"])) / float(_cl["J_ud"]), "{:.2f}")
    put("nGenClMinF", float(_cl["U_f"].min() / _cl["w_ud"]), "{:.3f}")
    put("nGenClMinAng", abs(float(np.rad2deg(_cl["U_theta"][np.argmin(_cl["U_f"])]))), "{:.0f}")
    put("nGenClf", float(_cl["w_ud"]) / 2 / np.pi, "{:.0f}")
    _mv = _GEN["mv_43"]
    put("nGenMvRatio", float(_GEN["ss_43"]["J_ud"] / _mv["J_ud"]), "{:.0f}")
    put("nGenMvRelaxUD", 100 * float(_mv["relax_ud"]), "{:.0f}")
    put("nGenMvRelaxOpt", 100 * float(_mv["relax_steer"]), "{:.0f}")
    put("nGenTwoR", 1000 * float(_GEN["ss_21"]["Rmin_used"]), "{:.0f}")
_CF = load("r1_conflict.npz")
if _CF is not None:
    _n = list(_CF["names"]); _c = _CF["rows"][:, 0]
    for key, mac in (("ss_43", "nChiBase"), ("ss_11", "nChiSq"), ("ss_21", "nChiTwo"), ("cl_43", "nChiCl"), ("mv_43", "nChiMv")):
        put(mac, float(_c[_n.index(key)]), "{:.0f}")
_DP = load("r1_damping.npz")
if _DP is not None:
    _d = _DP["rows"]; _m = _d[:, 1] == 0
    put("nDampRatioMin", float(_d[_m, 4].min()), "{:.3f}")
    put("nDampRatioMax", float(_d[_m, 4].max()), "{:.3f}")
    put("nDampStiffRatio", float(_d[~_m, 4][0]), "{:.2f}")
    put("nDampAcLo", float(_d[_m, 2][0]), "{:.3f}")
    put("nDampAcHi", float(_d[_m, 2][-1]), "{:.3f}")
    put("nDampScale", float(np.polyfit(np.log(_d[_m, 0]), np.log(_d[_m, 2]), 1)[0]), "{:.2f}")
_AB = load("r1_asbuilt.npz")
if _AB is not None:
    _k = list(_AB["res_keys"]); _R = _AB["res"]
    _t = _AB["tau"]
    put("nTauMin", float(_t.min()), "{:.2f}")
    put("nTauMax", float(_t.max()), "{:.0f}")
    put("nTauGapArea", 100 * float(np.mean(_t < 0.95)), "{:.0f}")
    for N in (4, 8, 16):
        g, o = _R[_k.index(f"GAP{N}")], _R[_k.index(f"OVL{N}")]
        w = {4: "Four", 8: "Eight", 16: "Sixteen"}[N]
        put(f"nGap{w}J", float(g[0]), "{:.3f}"); put(f"nGap{w}F", float(g[1]), "{:.3f}"); put(f"nGap{w}M", 100 * (float(g[2]) - 1), "{:+.1f}")
        put(f"nOvl{w}J", float(o[0]), "{:.3f}"); put(f"nOvl{w}F", float(o[1]), "{:.3f}"); put(f"nOvl{w}M", 100 * (float(o[2]) - 1), "{:+.1f}")
    _Js = [float(_R[_k.index(f"{t}{N}")][0]) for t in ("GAP", "OVL") for N in (4, 8, 16)]
    put("nAsBuiltJmin", min(_Js), "{:.2f}"); put("nAsBuiltJmax", max(_Js), "{:.2f}")
    put("nAsBuiltMaxDev", max(abs(j - float(_R[_k.index("ideal")][0])) for j in _Js), "{:.2f}")
    _MC = _AB["MC"]
    put("nAngJmean", float(_MC[:, 0].mean()), "{:.3f}"); put("nAngJstd", float(_MC[:, 0].std()), "{:.3f}")
    put("nAngFstd", 100 * float(_MC[:, 1].std()), "{:.1f}")
    _cts = _R[_k.index("CTS")]
    put("nCtsF", float(_cts[1]), "{:.2f}"); put("nCtsM", float(_cts[2]), "{:.2f}")
_VF = load("r1_verification.npz")
if _VF is not None:
    _g = _VF["grad_rows"]; _nd = int(_VF["ndirs"])
    _worst = max(min(_g[(_g[:, 1] == j) & (_g[:, 0] == i), 3]) for j in range(4) for i in range(_nd))
    _e = int(np.ceil(np.log10(_worst)))
    put("nGradErr", f"10^{{{_e}}}", "{}")
    put("nGradDirs", _nd, "{:d}")
    put("nRminFine", 1000 / float(_VF["kmax_fine"]), "{:.1f}")
    put("nKappaOver", 100 * (float(_VF["kmax_fine"]) / float(_VF["zall"]) - 1), "{:.1f}")
    _ms = _VF["ms"]; _b = _ms[:, 0].min()
    put("nMSgood", int(np.sum(_ms[:, 0] <= 1.05 * _b)), "{:d}")
    put("nMSworst", float(_ms[:, 0].max()), "{:.2f}")
    put("nMSspread", 100 * (float(np.sort(_ms[:, 0])[int(np.sum(_ms[:, 0] <= 1.05 * _b)) - 1]) / _b - 1), "{:.0f}")
    def _sci(x):
        if x == 0:
            return "0"
        e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
        return f"{m:.0f}\\times10^{{{e}}}"
    _tabrows = []
    for lab, r in zip(_VF["ms_labels"], _ms):
        _tabrows.append(f"{lab} & ${r[0]:.4f}$ & ${_sci(r[1])}$ & ${_sci(r[2])}$ & {int(r[4])}\\\\")
    with open("docs/paper/ms_table.tex", "w") as fh:
        fh.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nstart & $J/J_{\\UD}$ & $\\omega_1/\\omega_\\star-1$ & $\\min_e c_e$ & iterations\\\\\n\\midrule\n")
        fh.write("\n".join(_tabrows) + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

import glob, re
used = set()
for f in glob.glob("docs/paper/sec_*.tex"):
    used |= set(re.findall(r"\\(n[A-Z][A-Za-z]*)", open(f).read()))
missing = sorted(u for u in used if u not in macros)
for u in missing:
    macros[u] = r"\textbf{??}"
if missing:
    print("MISSING (shown as ??):", " ".join(missing))

with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by examples/paper_numbers.py -- do not edit by hand.\n")
    for k in sorted(macros):
        fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(f"wrote {OUT}: {len(macros)} macros")
