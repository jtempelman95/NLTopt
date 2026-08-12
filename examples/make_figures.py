r"""Regenerate every figure into ``figs/`` and sync the paper subset to
``docs/paper/figs/``.

Each figure has an owning example script (which writes its PNG into ``figs/``).
This driver runs them, times them, and copies the ``paper`` ones to the
manuscript's figure directory. Run all, or a subset by name:

    python examples/make_figures.py                 # all
    python examples/make_figures.py fig2 fig3       # by tag
    python examples/make_figures.py --list

Some figures embed optimizations and take minutes; ``kind`` marks the fast
verification plots vs. the slower design results.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
PAPER_FIGS = os.path.join(ROOT, "docs", "paper", "figs")

# tag -> (script, output png in figs/, kind, needs_wavetopo, in_paper)
REGISTRY = {
    "duffing":  ("duffing_ssm.py",               "duffing_ssm_backbone.png",          "verify",  False, False),
    "vkbeam":   ("vonkarman_beam_ssm.py",        "vonkarman_beam_ssm_backbone.png",   "verify",  False, False),
    "fig2":     ("coupled_osc_sensitivity.py",   "coupled_osc_sensitivity.png",       "paper",   False, True),
    "fig3":     ("beam_backbone_opt.py",         "beam_backbone_opt.png",             "paper",   False, True),
    "steered":  ("steered_beam_amplitude_robust.py", "steered_beam_amplitude_robust.png", "result", False, True),
    "plate":    ("plate_framework_prototype.py", "plate_framework_prototype.png",     "result", True,  True),
}


def run_one(tag):
    script, png, kind, needs_wt, in_paper = REGISTRY[tag]
    env = dict(os.environ)
    pp = [ROOT] + ([os.path.join(ROOT, "..", "wavetopo")] if needs_wt else [])
    env["PYTHONPATH"] = os.pathsep.join(pp + [env.get("PYTHONPATH", "")])
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join("examples", script)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    out = os.path.join(FIGS, png)
    ok = r.returncode == 0 and os.path.exists(out)
    if ok and in_paper:
        os.makedirs(PAPER_FIGS, exist_ok=True)
        shutil.copy2(out, os.path.join(PAPER_FIGS, png))
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {tag:8s} {kind:6s} {dt:6.1f}s  -> figs/{png}"
          + ("  (+paper)" if ok and in_paper else ""))
    if not ok:
        print("    " + (r.stderr.strip().splitlines() or ["<no stderr>"])[-1])
    return ok


def main(argv):
    os.makedirs(FIGS, exist_ok=True)
    if "--list" in argv:
        for tag, (s, p, k, w, ip) in REGISTRY.items():
            print(f"  {tag:8s} {k:6s} {'[paper]' if ip else '       '} {s}")
        return
    tags = [a for a in argv if a in REGISTRY] or list(REGISTRY)
    print(f"regenerating {len(tags)} figure(s) into figs/  (paper subset -> docs/paper/figs/)")
    n_ok = sum(run_one(t) for t in tags)
    print(f"done: {n_ok}/{len(tags)} ok")


if __name__ == "__main__":
    main(sys.argv[1:])
