r"""Basis convergence of the REPORTED held-frequency design (Table: tab:conv).

Evaluates the steered design of design_source.heldfreq_design() and the UD plate on
transverse bases m,n <= M and in-plane bases i,j <= N (same 96x72 quadrature grid).

    python examples/conv_table.py -> docs/paper/conv_table.tex
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import numpy as np
sys.path.insert(0, (_ROOT + "/examples"))
import levers_study as LS
from design_source import heldfreq_design
from nltopt.relaxed_plate import RelaxedPlate, SteeredPlate

th_hat = heldfreq_design()
rows = []
for M, N in [(8, 3), (8, 4), (8, 5), (8, 6), (8, 8), (12, 6), (12, 8), (16, 8)]:
    P = RelaxedPlate(mmax=M, n_in=N)
    PHI = LS.csrbf(P)[0]
    sp = SteeredPlate(P)
    ud, st = sp.forward(0.0, full=False), sp.forward(PHI @ th_hat, full=False)
    rows.append((M, N, ud["omega1"] / 2 / np.pi, st["omega1"] / 2 / np.pi, st["J"] / ud["J"],
                 100 * ud["relax"] / ud["knl"], 100 * st["relax"] / st["knl"]))
    print("M=%2d N=%d  f_UD=%.1f f_st=%.1f J/J_UD=%.3f share UD %.0f%% steered %.0f%%" % rows[-1], flush=True)
with open("docs/paper/conv_table.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\n"
             "transverse $m,n\\le$ & in-plane $i,j\\le$ & $f_{1,\\UD}$ (Hz) & $f_1$ (Hz) & $J/J_{\\UD}$ & "
             "relaxed share (UD) & relaxed share (steered)\\\\ \\midrule\n")
    for M, N, fu, fs, j, su, ss in rows:
        b = "\\bf " if (M, N) == (8, 6) else ""
        fh.write(f"{b}{M} & {b}{N} & {b}{fu:.1f} & {b}{fs:.1f} & {b}{j:.3f} & {b}{su:.0f}\\% & {b}{ss:.0f}\\%\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
np.save("results/conv_table.npy", np.array(rows))
print("wrote docs/paper/conv_table.tex")
