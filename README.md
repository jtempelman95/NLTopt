# NLTopt — manufacturable design of *nonlinear* structural dynamics

NLTopt opens a new research line by **coupling two existing tools**:

- [**PyHB**](../PyHB) — harmonic balance, continuation, shooting and EPMC
  nonlinear modal analysis for second-order mechanical systems;
- [**wavetopo**](../wavetopo) — curvature-feasible **fiber-orientation design**
  (CS-RBF tow paths, tow-curvature manufacturability constraints, MMA), so far
  applied only to *linear* wave/structural problems.

The goal: **manufacturable fiber-steering design of the nonlinear vibration
response of composite components** — tailoring **backbone** and **forced-response
curves** (hardening/softening, peak amplitude, bifurcation spacing) of laminated
CFRP beams/plates, subject to the same tow-curvature manufacturability
constraints that anchor the wavetopo papers.

It keeps `wavetopo` lean: NLTopt lives in its own repository and *pulls* from
both dependencies rather than growing either one.

## Why this is an open contribution

Reduction on **spectral submanifolds (SSMs)** makes nonlinear-dynamic
optimization tractable — the backbone/FRC *and its adjoint sensitivity* are
available analytically, at a cost independent of the number of design variables.
The recent literature (see [`docs/literature`](docs/literature)) established this
with **shape / size / density** design variables:

- Pozzi, Marconi, Jain, Li & Braghin (2025) — *adjoint sensitivities of the SSM
  backbone* (`pozzi2025_ssm_adjoint.pdf`);
- Liang, Pozzi, Marconi, Jain & Li (2025) — *topology optimization of nonlinear
  forced-response curves via SSM* (`arxiv_2510.07900v2.pdf`);
- Xiao (Bristol PhD, 2026) — *ROM-based optimization of nonlinear dynamics*
  (`xiao_2026_phd_thesis.pdf`).

**None of them use a continuous fiber-orientation field as the design variable,
and none impose manufacturability constraints.** That is the gap NLTopt targets:

> The first manufacturability-constrained, **fiber-steered** design of a
> composite's nonlinear backbone / forced response, via SSM adjoint sensitivity.

Three things make it more than gluing two codes together:
1. **A new design-variable class for SSM optimization** — the tow-path field
   `θ(x)` is *many* variables, exactly the regime where the SSM adjoint (cost
   independent of design count) pays off.
2. **θ steers the nonlinearity itself** — in the von Kármán model the cubic
   membrane–bending coupling is built from the laminate `A/B/D` matrices, which
   rotate with `θ`; fiber orientation controls not just `ω` but the
   hardening/softening slope.
3. **Manufacturability closes the loop** — the optimized `θ(x)` must be a
   printable tow path (`|ζ| ≤ 1/R_tow`), inherited from wavetopo.

## What's here now (v0.0.1 — the scaffold)

A **cubic-order, 2-D spectral submanifold** and its **physical backbone curve**,
built directly from a `pyhb.MechanicalSystem` and verified three ways.

```python
from pyhb import MechanicalSystem, BeamModel
from pyhb.nonlinear_elements import VonKarmanBeamForce
from nltopt import ssm_from_system
import numpy as np

beam = BeamModel(n_elem=12, length=1.0, EI=1.0, rhoA=1.0, EA=1.0).clamped_clamped()
vk   = VonKarmanBeamForce(beam.Kg, beam.EA / (2 * beam.length))
sys  = MechanicalSystem(beam.M, np.zeros_like(beam.M), beam.K, [vk])

ssm  = ssm_from_system(sys, mode=0, ref_dof=beam.free_index_of(beam.node_dof(6, "w")))
print(ssm.gamma, ssm.hardening)        # reduced cubic coeff; hardening/softening
bb = ssm.backbone(np.linspace(1e-5, 0.03, 200))   # {omega, amplitude, damping, ...}
```

- `nltopt/nonlinearity.py` — extracts the **symmetric multilinear forms**
  (`B2`, `T3`) of a system's nonlinearity by polarization, so the SSM and PyHB's
  harmonic-balance/EPMC model consume *identical* nonlinearities.
- `nltopt/ssm.py` — the 2-D SSM to **arbitrary polynomial order** (`ssm_from_system(
  …, order=k)`): solves the homological equations order-by-order in the first-order
  state space, extracts the reduced coefficients `γ₃, γ₅, …` via the left-eigenvector
  (adjoint) solvability condition, and reads the backbone back in physical coordinates
  (fundamental & RMS read-outs, amplitude→`ρ` inversion). Order 3 gives a monotone
  (soften **or** harden) parabola; **order ≥ 5 captures soften-to-harden backbones**
  (`γ₅` supplies the hardening that turns a softening backbone around).
- `nltopt/beam.py` — a **curved (shallow-arch) von Kármán beam**: an initial shape
  `w0` introduces the *quadratic* nonlinearity that lets the backbone soften
  (`w0=0` recovers the straight hardening beam exactly). The model behind the
  Fig. 3 optimization; reuses PyHB's `Kg`.

### Verification (`tests/`, all passing)

| Check | What it exercises | Result |
|---|---|---|
| Duffing (1 DOF) vs **analytic** `Ω = ω₀(1 + 3knl A²/8ω₀²)` | reduced cubic coeff + amplitude map | exact to 6 digits |
| Duffing vs **EPMC** | independent full-order method | agree |
| Quadratically-coupled 2-DOF vs **EPMC** | order-2 **slaving** (manifold curvature) | agree (softening) |
| Clamped–clamped **von Kármán beam** vs **EPMC** | distributed cubic, multi-DOF | `dΩ/dA² = 0.503` both |
| Curved-beam straight limit vs `VonKarmanBeamForce` | quadratic model reduces correctly | exact |
| Initial curvature → **softening** | the curvature-quadratic mechanism | `γ` flips sign |
| **Backbone target optimization** | the whole design loop | hits target |
| Coupled oscillators vs **paper eigenvalue** | damped 2-DOF benchmark | `λ=−0.0191±i0.6177` exact |
| **Backbone-sensitivity prediction** | `∂Ω/∂μ` predicts perturbation | matches to `~1e-5` |
| Duffing **order-5** `A⁴` coefficient | arbitrary-order machinery | `−21/256` (5% ) |
| **Soften-to-harden** needs order ≥ 5 | quintic turn-back | order-7 turns back |

Examples reproduce the backbone figures:

```bash
pip install -e ../PyHB           # sibling dependency
pip install -e .                 # NLTopt
python examples/duffing_ssm.py           # SSM vs analytic vs EPMC
python examples/vonkarman_beam_ssm.py    # clamped-clamped beam, SSM vs EPMC
python examples/beam_backbone_opt.py     # Fig. 3 replication: optimize to a softening backbone
python examples/coupled_osc_sensitivity.py  # Fig. 2 replication: backbone-sensitivity validation
pytest                                    # the verification suite
```

## Roadmap

- [x] **(a)** cubic-order 2-D SSM + physical backbone, verified vs analytic & EPMC.
- [x] **(a2)** **backbone-shape optimization** replicating Pozzi et al. (2025) Fig. 3:
  a curved von Kármán beam optimized (SLSQP over shape/geometry, SSM backbone) to a
  prescribed *softening* target, verified against full-order EPMC (`<0.1%` at both
  target points). See `nltopt/beam.py`, `examples/beam_backbone_opt.py`. This proves
  the full optimization loop end-to-end; it uses FD-through-SSM gradients (fine for a
  handful of variables) — the analytic adjoint **(c)** is what makes it scale to a
  fiber *field*.
- [x] **(a3)** **backbone-sensitivity validation** replicating Pozzi et al. (2025) Fig. 2:
  two coupled oscillators (Eq. 33; our SSM reproduces the paper's eigenvalue
  `λ=−0.0191±i0.6177` exactly), with `∂Ω/∂μ` for `μ∈{m,k,k₂,k₃}` used to predict the
  perturbed backbones — predictions match the recomputed backbones to `~1e-5`, over the
  **soften-to-harden** "backward-C" backbone (captured with an **order-7** SSM; the cubic
  SSM can't turn back). See `nltopt/benchmarks.py`, `examples/coupled_osc_sensitivity.py`.
  (FD sensitivity through the SSM; the analytic adjoint **(c)** computes the same at cost
  independent of #params.)
- [~] **(b)** *laminated* von Kármán beam — the **novel** problem: steer the tow path
  `θ(x)` to place the nonlinear backbone under a manufacturability constraint. The lever
  is the laminate **bending–extension coupling `B(θ)`** (single-angle steering only scales
  `EI,EA` together and can't tune the hardening; an *unsymmetric* layup's `B` feeds a
  tunable quadratic → softening to balance the cubic hardening → **iso-frequency**).
  - [x] **b.1** classical laminate theory `A,B,D(θ)` from wavetopo's CFRP lamina
    (`nltopt/laminate.py`): symmetric⇒`B=0`, `[θ/0]` steering sweeps the dimensionless
    coupling `0→0.75` — the verified design lever.
  - [ ] **b.2** laminated beam FE (`u,w` DOF, von Kármán + `B`-coupling) → `M,K,Q,C`.
  - [ ] **b.3** SSM backbone as a function of `θ(x)`.
  - [ ] **b.4** optimize `θ(x)` to flatten `Ω(a)` under the tow-curvature constraint.
- [ ] **(c)** **adjoint sensitivity** `dγ/dθ`, `dΩ/dθ` of the backbone w.r.t. the
  fiber-orientation field; FD-verified against direct differentiation
  (paper App. E) before large-scale optimization.
- [ ] **(d)** MMA optimization of `θ(x)` under wavetopo's tow-curvature
  constraint to place a target backbone (iso-frequency / prescribed
  hardening/softening).
- [ ] **(e)** forced-response curves (Liang et al. 2025): peak-amplitude and
  saddle-node-spacing objectives.
- [ ] **(f)** graduate from beam to a 2-D dolfinx von Kármán plate.

## Layout

```
nltopt/          package: nonlinearity, ssm, beam, laminated_beam, laminate, benchmarks
tests/           verification suite (analytic + EPMC cross-checks, optimization, sensitivity)
examples/        runnable demos; make_figures.py regenerates every figure
figs/            generated figures (scratch, git-ignored)
results/         generated data .npz (scratch, git-ignored)
docs/paper/      nltopt_paper.tex + figs/ (the curated paper figures, tracked)
docs/literature/ the SSM-optimization papers this program builds on (untracked)
```

**Figures.** Every figure is owned by an example that writes into `figs/`;
`python examples/make_figures.py [tags…]` regenerates them and syncs the paper
subset to `docs/paper/figs/`. Tags: `duffing`, `vkbeam` (fast verification);
`fig2`, `fig3` (paper replications); `steered`, `plate` (design results, slow).
The `plate` prototype couples wavetopo's CS-RBF + curl + MMA to the SSM backbone
(needs `../wavetopo` on the path).
