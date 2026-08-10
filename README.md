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
- `nltopt/ssm.py` — the cubic-order 2-D SSM: solves the homological equations in
  the first-order state space, extracts the reduced cubic coefficient `γ` via the
  left-eigenvector (adjoint) solvability condition, and reads the backbone back
  in physical coordinates.

### Verification (`tests/`, all passing)

| Check | What it exercises | Result |
|---|---|---|
| Duffing (1 DOF) vs **analytic** `Ω = ω₀(1 + 3knl A²/8ω₀²)` | reduced cubic coeff + amplitude map | exact to 6 digits |
| Duffing vs **EPMC** | independent full-order method | agree |
| Quadratically-coupled 2-DOF vs **EPMC** | order-2 **slaving** (manifold curvature) | agree (softening) |
| Clamped–clamped **von Kármán beam** vs **EPMC** | distributed cubic, multi-DOF | `dΩ/dA² = 0.503` both |

Examples reproduce the backbone figures:

```bash
pip install -e ../PyHB           # sibling dependency
pip install -e .                 # NLTopt
python examples/duffing_ssm.py           # SSM vs analytic vs EPMC
python examples/vonkarman_beam_ssm.py    # clamped-clamped beam, SSM vs EPMC
pytest                                    # the verification suite
```

## Roadmap

- [x] **(a)** cubic-order 2-D SSM + physical backbone, verified vs analytic & EPMC.
- [ ] **(b)** *laminated* von Kármán beam: `EI(θ)`, `EA(θ)` from a rotated
  laminate (the 1-D analogue of wavetopo's `D_f(θ)`) — makes the backbone a
  function of the fiber angle.
- [ ] **(c)** **adjoint sensitivity** `dγ/dθ`, `dΩ/dθ` of the backbone w.r.t. the
  fiber-orientation field; FD-verified against direct differentiation
  (paper App. E) before any optimization.
- [ ] **(d)** MMA optimization of `θ(x)` under wavetopo's tow-curvature
  constraint to place a target backbone (iso-frequency / prescribed
  hardening/softening).
- [ ] **(e)** forced-response curves (Liang et al. 2025): peak-amplitude and
  saddle-node-spacing objectives.
- [ ] **(f)** graduate from beam to a 2-D dolfinx von Kármán plate.

## Layout

```
nltopt/          package: nonlinearity.py, ssm.py
tests/           verification suite (analytic + EPMC cross-checks)
examples/        runnable backbone demos
docs/literature/ the SSM-optimization papers this program builds on (untracked)
```
