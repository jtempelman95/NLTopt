r"""Classical laminate theory (CLT) for fiber-steered composite beams.

Gives the laminate stiffnesses ``A`` (extensional), ``B`` (bending-extension
coupling) and ``D`` (bending) from the ply angles, using wavetopo's CFRP lamina
properties.  ``B`` is the lever this whole line of work turns on: for a plain
beam, rotating the fibers scales the axial modulus and therefore ``A`` and ``D``
*together*, leaving the gyration radius (and thus the hardening) unchanged -- so
single-angle steering only moves the linear frequency.  An **unsymmetric** layup
has ``B != 0``: transverse motion's membrane strain ``1/2 (w')^2`` feeds, through
``B``, a **quadratic** transverse force (softening or hardening by the sign of
``B``).  Steering the ply angle changes ``B`` largely independently of ``A``, so
the tow path can balance ``B``-softening against ``A``-hardening -> an amplitude-
invariant ("iso-frequency") backbone.

Conventions: plane-stress reduced stiffness ``Q`` (on-axis), rotated to ``Qbar``
by ply angle ``theta`` (radians, measured from the beam axis).  A laminate is a
bottom-to-top stack of equal- or unequal-thickness plies; ``z`` runs from
``-h/2`` to ``+h/2``.  For a (narrow) beam we use the axial ``11`` components
``A11, B11, D11`` -- the standard laminated-beam reduction.
"""

from __future__ import annotations

import numpy as np

# wavetopo CFRP lamina (GPa, and density kg/m^3) -- see wavetopo/CLAUDE.md
E1, E2, G12, NU12 = 131e9, 9e9, 5e9, 0.27
RHO = 1600.0


def lamina_Q(E1=E1, E2=E2, G12=G12, nu12=NU12):
    """On-axis plane-stress reduced stiffness matrix ``Q`` (3x3, Voigt 1,2,6)."""
    nu21 = nu12 * E2 / E1
    d = 1.0 - nu12 * nu21
    return np.array([[E1 / d,        nu12 * E2 / d, 0.0],
                     [nu12 * E2 / d, E2 / d,        0.0],
                     [0.0,           0.0,           G12]])


def Qbar(theta, Q=None):
    """Reduced stiffness rotated to fiber angle ``theta`` (radians)."""
    if Q is None:
        Q = lamina_Q()
    m, n = np.cos(theta), np.sin(theta)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    Qb = np.empty((3, 3))
    Qb[0, 0] = Q11 * m4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n4
    Qb[1, 1] = Q11 * n4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m4
    Qb[0, 1] = Qb[1, 0] = (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m4 + n4)
    Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m4 + n4)
    Qb[0, 2] = Qb[2, 0] = ((Q11 - Q12 - 2 * Q66) * m2 * m * n
                           + (Q12 - Q22 + 2 * Q66) * m * n2 * n)
    Qb[1, 2] = Qb[2, 1] = ((Q11 - Q12 - 2 * Q66) * m * n2 * n
                           + (Q12 - Q22 + 2 * Q66) * m2 * m * n)
    return Qb


def laminate_ABD(angles, thickness, Q=None):
    """Full 3x3 ``A, B, D`` for a stack of equal-thickness plies.

    Parameters
    ----------
    angles : sequence of float
        Ply angles (radians), listed bottom -> top.
    thickness : float
        Total laminate thickness (each ply is ``thickness/len(angles)``).
    """
    if Q is None:
        Q = lamina_Q()
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    nply = angles.size
    tply = thickness / nply
    z = -0.5 * thickness + tply * np.arange(nply + 1)      # ply interfaces
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, th in enumerate(angles):
        Qb = Qbar(th, Q)
        A += Qb * (z[k + 1] - z[k])
        B += Qb * 0.5 * (z[k + 1] ** 2 - z[k] ** 2)
        D += Qb * (1.0 / 3.0) * (z[k + 1] ** 3 - z[k] ** 3)
    return A, B, D


def beam_abd(angles, thickness, width=1.0, Q=None):
    """Axial (``11``) laminated-beam stiffnesses ``(A11, B11, D11)`` and mass/length.

    Returns ``(A11, B11, D11, rhoA)`` for a beam of the given ``width``, using the
    laminate's 11 components (narrow/wide-strip reduction).  ``rhoA`` is the mass
    per unit length.
    """
    A, B, D = laminate_ABD(angles, thickness, Q)
    A11 = width * A[0, 0]
    B11 = width * B[0, 0]
    D11 = width * D[0, 0]
    rhoA = RHO * width * thickness
    return A11, B11, D11, rhoA
