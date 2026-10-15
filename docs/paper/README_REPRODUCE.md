# Reproducing the NLTopt paper

All numbers in `numbers.tex` and the generated tables (`conv_table.tex`, `fe_table.tex`,
`svse_table.tex`) are written by scripts from the files in `results/`. Run from the repository root
with a Python that has numpy/scipy/matplotlib (+ a LaTeX install for the pgf figures). Some scripts
use the CS-RBF map of the companion repository `wavetopo` (set `WAVETOPO_DIR`, default: sibling
directory `../wavetopo`).

Order (each step writes into `results/`; see Table "Result → script map" in the appendix):
1. `examples/levers_study.py`, `examples/stack_continuous.py`, `examples/mechanism_study.py`,
   `examples/mechanism_lambda.py`, `examples/ceiling_study.py`, `examples/beam_symmetric_steering.py`
2. `examples/confirm_heldfreq.py`, then `examples/epmc_check.py` (merges EPMC into `confirm.npz`),
   `examples/confirm_fine_sweep.py`, `examples/fe_check_design.py`, `examples/mode_ratios.py`,
   `examples/conv_table.py`
3. figures: `examples/fig_*.py`, `examples/plate_fe_fig.py`, calibration scripts listed in the appendix
4. `examples/paper_numbers.py` → `docs/paper/numbers.tex` and generated tables; then `pdflatex nltopt_paper`.

Unit tests: `pytest tests/`. Scripts in `examples/` not listed above belong to earlier, archived studies.
