r"""Curved (shallow-arch) von Karman beam -- the model behind a *tunable*
hardening/softening backbone.

A *straight* von Karman beam can only harden: its mid-plane-stretching
nonlinearity is purely cubic (``f = (EA/2L)(w^T Kg w) Kg w``, hardening).  An
**initial curvature** ``w0(x)`` breaks that symmetry -- it introduces a
**quadratic** nonlinearity that can make the first-mode backbone *soften*.  This
is exactly the mechanism the optimization in Pozzi, Marconi, Jain, Li & Braghin
(2025, Sec. 5.2 / Fig. 3) exploits: it shapes an initially curved beam so its
backbone follows a prescribed softening target.

Model (condensed, axially-immovable ends -- a shallow-arch reduction of the full
3-DOF/node von Karman element).  With additional transverse motion ``w`` on top
of a fixed initial shape ``w0``, the uniform membrane strain is
``eps = (1/L)[w0^T Kg w + 1/2 w^T Kg w]`` and the membrane force ``N = EA eps``.
The elastic membrane energy ``U = 1/2 EA L eps^2`` gives, on differentiation,

* a linear **initial-curvature stiffening**  ``dK = (EA/L)(Kg w0)(Kg w0)^T``,
* a **quadratic** force  ``Q(w) = (EA/L)[ 1/2 (w^T Kg w) Kg w0 + (w0^T Kg w) Kg w ]``,
* the usual **cubic** force ``C(w) = (EA/2L)(w^T Kg w) Kg w``.

``w0 = 0`` recovers PyHB's straight :class:`~pyhb.nonlinear_elements.VonKarmanBeamForce`
exactly.  We reuse PyHB's ``BeamModel`` for ``M``, ``K``, ``Kg`` (Euler-Bernoulli
bending DOFs ``[w, theta]`` per node), add ``dK``, and attach ``Q + C`` as a
single distributed element.  Both the NLTopt SSM and PyHB's EPMC then consume the
identical nonlinearity, so the model can be cross-checked (see
``examples/curved_beam_softening.py`` / tests).
"""

from __future__ import annotations

import numpy as np

from pyhb import MechanicalSystem, BeamModel
from pyhb.nonlinear_elements import GeneralNonlinearElement


class CurvedVKForce(GeneralNonlinearElement):
    r"""Distributed nonlinearity of a shallow-arch (curved) von Karman beam.

    ``f(w) = (EA/L)[ 1/2 (w^T Kg w) Kg w0 + (w0^T Kg w) Kg w ] + (EA/2L)(w^T Kg w) Kg w``.

    Parameters
    ----------
    Kg : ndarray (n, n)
        Geometric-stiffness matrix in the reduced (free) DOFs.
    w0 : ndarray (n,)
        Initial-shape nodal vector in the same reduced DOFs (``w`` and
        ``theta`` entries).  ``w0 = 0`` gives the straight-beam cubic force.
    ea_over_l : float
        ``EA / L``.
    """

    def __init__(self, Kg, w0, ea_over_l):
        self.Kg = 0.5 * (np.asarray(Kg, float) + np.asarray(Kg, float).T)
        self.w0 = np.asarray(w0, float).ravel()
        self.eal = float(ea_over_l)
        self.n = self.Kg.shape[0]
        self.Kgw0 = self.Kg @ self.w0                 # constant vector

    def forces(self, q_time, u_time):
        N = q_time.shape[0]
        eal = self.eal
        Kg = self.Kg
        Kgw0 = self.Kgw0
        v = q_time @ Kg                                # (N, n) = Kg w per sample
        s_ww = np.einsum("ni,ni->n", q_time, v)        # w^T Kg w
        s_w0w = q_time @ Kgw0                          # w0^T Kg w   (N,)

        # force = quadratic (curvature) + cubic (stretching)
        f = eal * (0.5 * s_ww[:, None] * Kgw0[None, :] + s_w0w[:, None] * v)
        f += 0.5 * eal * s_ww[:, None] * v

        # tangent df/dw
        Kt = np.empty((N, self.n, self.n))
        vv = np.einsum("ni,nj->nij", v, v)             # (Kg w)(Kg w)^T
        v_Kgw0 = np.einsum("ni,j->nij", v, Kgw0)       # (Kg w)(Kg w0)^T
        Kgw0_v = np.einsum("i,nj->nij", Kgw0, v)       # (Kg w0)(Kg w)^T
        Kt[:] = eal * (Kgw0_v + v_Kgw0 + s_w0w[:, None, None] * Kg[None])
        Kt += eal * vv + 0.5 * eal * s_ww[:, None, None] * Kg[None]
        Ct = np.zeros((N, self.n, self.n))
        return f, Kt, Ct


def _initial_shape_full(beam, A1, A2, length):
    """Full-DOF initial-shape vector ``w0`` for ``y = A1 sin(pi x/L)+A2 sin(2pi x/L)``.

    Fills both the transverse (``w``) and rotation (``theta = dy/dx``) DOFs so the
    geometric coupling ``Kg w0`` sees the correct slope.
    """
    x = beam.x
    L = length
    w0 = np.zeros(beam.ndof_full)
    y = A1 * np.sin(np.pi * x / L) + A2 * np.sin(2 * np.pi * x / L)
    dy = (A1 * (np.pi / L) * np.cos(np.pi * x / L)
          + A2 * (2 * np.pi / L) * np.cos(2 * np.pi * x / L))
    w0[0::2] = y
    w0[1::2] = dy
    return w0


def curved_vk_beam(A1, A2, h, length, E=90e9, rho=7850.0, n_elem=10,
                   width=1.0):
    r"""Build a clamp-clamp curved von Karman beam as a :class:`pyhb.MechanicalSystem`.

    Rectangular cross-section (width ``width``, thickness ``h``); the width
    cancels in the natural frequency and the normalized backbone, so its value
    is immaterial.  Lengths in metres, ``E`` in Pa, ``rho`` in kg/m^3.

    Returns
    -------
    dict with keys
        ``system``   -- the MechanicalSystem (K already includes dK),
        ``beam``     -- the underlying BeamModel,
        ``ref``      -- reduced-DOF index of the mid-span transverse motion,
        ``omega1``   -- first linear natural frequency of this configuration,
        ``r_gyr``    -- gyration radius sqrt(EI/EA) = h/sqrt(12).
    """
    b = width
    EI = E * b * h ** 3 / 12.0
    EA = E * b * h
    rhoA = rho * b * h

    beam = BeamModel(n_elem=n_elem, length=length, EI=EI, rhoA=rhoA, EA=EA)
    beam.clamped_clamped()

    w0_full = _initial_shape_full(beam, A1, A2, length)
    w0 = w0_full[beam.free]                             # reduced DOFs

    eal = EA / length
    dK = eal * np.outer(beam.Kg @ w0, beam.Kg @ w0)     # initial-curvature stiffening
    K_eff = beam.K + dK

    element = CurvedVKForce(beam.Kg, w0, eal)
    sys = MechanicalSystem(beam.M, np.zeros_like(beam.M), K_eff, [element])

    ref = beam.free_index_of(beam.node_dof(n_elem // 2, "w"))
    omega1 = float(beam.natural_frequencies()[0])       # NOTE: without dK
    # recompute omega1 with dK included (the actual configuration)
    from scipy.linalg import eigh
    omega1 = float(np.sqrt(np.clip(eigh(K_eff, beam.M, eigvals_only=True)[0], 0, None)))
    r_gyr = np.sqrt(EI / EA)
    return {"system": sys, "beam": beam, "ref": ref, "omega1": omega1,
            "r_gyr": r_gyr, "EI": EI, "EA": EA, "rhoA": rhoA}
