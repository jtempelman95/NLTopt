"""Multilinear extraction of the polynomial nonlinearity of a PyHB system.

A :class:`pyhb.MechanicalSystem` carries a list of local nonlinear elements.
For spectral-submanifold (SSM) reduction we do not want the *time-history*
force those elements evaluate during harmonic balance -- we want the
**symmetric multilinear forms** of the underlying polynomial nonlinearity:

* the quadratic (bilinear) form  ``B2(a, b)``  with  ``Q(x) = B2(x, x)``,
* the cubic  (trilinear) form    ``T3(a, b, c)`` with ``C(x) = T3(x, x, x)``,

so the total displacement-dependent restoring nonlinearity is
``f_nl(x) = Q(x) + C(x)`` (velocity-independent elements only; friction and
other dissipative elements are not part of the conservative SSM backbone).

We obtain these *without any hand-coded per-element algebra*: every element in
PyHB can be evaluated at a single displacement sample, so we

1. read the total nonlinear force ``F(x) = f_nl(x)`` at an arbitrary ``x``,
2. split it into its even (degree-2) and odd (degree-3) homogeneous parts via
   ``Q(x) = (F(x) + F(-x)) / 2``,  ``C(x) = (F(x) - F(-x)) / 2``,
3. recover the symmetric multilinear forms by polarization.

Because ``Q`` and ``C`` are *exactly* homogeneous of degree 2 and 3 for the
supported elements (polynomial springs, von Karman beam force), the
polarization identities are exact.  The returned forms are complex-linear
extensions of the real multilinear forms, so the SSM machinery may feed them
complex manifold coefficients.

This keeps the SSM model and the PyHB harmonic-balance / EPMC model *bit-for-
bit consistent*: they consume the same element definitions.
"""

from __future__ import annotations

import itertools

import numpy as np

from pyhb.nonlinear_elements import GeneralNonlinearElement


# --------------------------------------------------------------------------- #
# Total displacement-dependent nonlinear force of a system at a single sample
# --------------------------------------------------------------------------- #
def nonlinear_force(system, x):
    """Total displacement-dependent nonlinear force ``f_nl(x)`` (velocity 0).

    ``x`` is a real length-``n`` displacement vector.  Only displacement-
    dependent elements contribute; velocity-dependent (friction) elements are
    ignored -- they are not part of the conservative backbone.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = system.n
    f = np.zeros(n)
    zero1 = np.zeros((1, n))
    for el in system.nonlinear_elements:
        if isinstance(el, GeneralNonlinearElement):
            fe, _, _ = el.forces(x[None, :], zero1)
            f += fe[0]
        else:
            g = float(el.w @ x)
            fe, _, dfdgd = el.evaluate(np.array([g]), np.array([0.0]))
            # skip purely velocity-dependent elements (f(g=., gdot=0) may be 0
            # but a friction element contributes nothing to the backbone anyway)
            f += el.w * fe[0]
    return f


def _even_odd(system):
    """Return the degree-2 (even) and degree-3 (odd) homogeneous force parts."""
    def Q(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (nonlinear_force(system, x) + nonlinear_force(system, -x))

    def C(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (nonlinear_force(system, x) - nonlinear_force(system, -x))

    return Q, C


# --------------------------------------------------------------------------- #
# Polarization -> symmetric multilinear forms (complex-linear extensions)
# --------------------------------------------------------------------------- #
def _bilinear_real(Q, a, b):
    """Symmetric bilinear form of a homogeneous quadratic force ``Q``."""
    return 0.5 * (Q(a + b) - Q(a) - Q(b))


def _trilinear_real(C, a, b, c):
    """Symmetric trilinear form of a homogeneous cubic force ``C``."""
    return (C(a + b + c) - C(a + b) - C(a + c) - C(b + c)
            + C(a) + C(b) + C(c)) / 6.0


def _complexify(real_form, degree):
    """Lift a real ``degree``-linear form to its complex-linear extension.

    ``real_form(*real_vectors) -> real vector``.  The returned callable accepts
    complex vectors and expands over real/imaginary parts, contributing a
    factor ``i`` for every imaginary part chosen.
    """
    def form(*vecs):
        vecs = [np.asarray(v) for v in vecs]
        if all(np.isrealobj(v) or not np.any(v.imag) for v in vecs):
            return real_form(*[v.real for v in vecs]).astype(complex)
        parts = [(v.real, v.imag) for v in vecs]
        out = None
        for choice in itertools.product((0, 1), repeat=degree):
            args = [parts[k][c] for k, c in enumerate(choice)]
            coeff = (1j) ** sum(choice)
            term = coeff * real_form(*args)
            out = term if out is None else out + term
        return out
    return form


def multilinear_forms(system, tol=1e-13):
    """Return ``(bilinear, trilinear)`` symmetric forms for ``system``.

    Each is either a callable (the complex-linear extension of the symmetric
    multilinear form) or ``None`` if that degree is absent -- e.g. a purely
    cubic (von Karman / Duffing) nonlinearity returns ``(None, trilinear)``.
    ``bilinear(a, b)`` satisfies ``Q(x) = bilinear(x, x)`` and
    ``trilinear(a, b, c)`` satisfies ``C(x) = trilinear(x, x, x)``.
    """
    Q, C = _even_odd(system)
    n = system.n
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(n)

    bilinear = None
    if np.max(np.abs(Q(probe))) > tol:
        bilinear = _complexify(lambda a, b: _bilinear_real(Q, a, b), 2)

    trilinear = None
    if np.max(np.abs(C(probe))) > tol:
        trilinear = _complexify(lambda a, b, c: _trilinear_real(C, a, b, c), 3)

    return bilinear, trilinear
