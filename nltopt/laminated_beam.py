r"""Fiber-steered laminated von Karman beam (membrane + bending, B-coupling).

A clamp-clamp beam with axial ``u`` and transverse ``w`` (+ slope) DOF, whose
per-element laminate stiffnesses ``A(theta), B(theta), D(theta)`` come from the
local fiber angle ``theta(x)`` (see :mod:`nltopt.laminate`).  The von Karman
membrane strain ``eps = u' + 1/2 (w')^2`` and curvature ``kappa = -w''`` combine
with the laminate constitutive law ``N = A eps + B kappa``, ``M = B eps + D
kappa`` to give the strain energy

    U = 1/2 integral ( A eps^2 + 2 B eps kappa + D kappa^2 ) dx,

whose gradient supplies the linear stiffness (with bending-extension coupling),
a **quadratic** nonlinear force (the ``B`` lever -- activated by *steering*,
``B'(x)``, once the axial DOF are slaved) and a **cubic** hardening force (the
usual membrane stretching, ``~A``).  Everything is assembled by Gauss quadrature
so spatially-varying ``A,B,D`` are handled exactly.

DOF ordering: 3 per node ``[u, w, w']``; a 2-node element has
``[u_a, w_a, w'_a, u_b, w_b, w'_b]``.  The nonlinear element consumed by
:class:`pyhb.MechanicalSystem` works in the reduced (free) DOFs.
"""

from __future__ import annotations

import numpy as np

from pyhb import MechanicalSystem
from pyhb.nonlinear_elements import GeneralNonlinearElement

from .laminate import beam_abd, RHO

# 4-point Gauss on [-1, 1]
_GX = np.array([-0.8611363115940526, -0.3399810435848563,
                0.3399810435848563, 0.8611363115940526])
_GW = np.array([0.3478548451374538, 0.6521451548625461,
                0.6521451548625461, 0.3478548451374538])


def _shape_derivs(Le):
    """Return per-Gauss-point (B_u, B_w, B_ww, N_u, N_w) shape rows for a 2-node
    element of length ``Le`` (6-DOF ordering [u_a,w_a,w'_a,u_b,w_b,w'_b])."""
    rows = []
    for xi in _GX:
        s = 0.5 * (xi + 1.0)                      # s in [0,1]
        # axial (linear)
        Nu = np.array([1 - s, 0, 0, s, 0, 0])
        Bu = np.array([-1 / Le, 0, 0, 1 / Le, 0, 0])
        # transverse (Hermite): values map to [w_a, w'_a, w_b, w'_b]
        N1 = 1 - 3 * s ** 2 + 2 * s ** 3
        N2 = Le * (s - 2 * s ** 2 + s ** 3)
        N3 = 3 * s ** 2 - 2 * s ** 3
        N4 = Le * (-s ** 2 + s ** 3)
        Nw = np.array([0, N1, N2, 0, N3, N4])
        # w' = dN/dx
        d1 = (-6 * s + 6 * s ** 2) / Le
        d2 = 1 - 4 * s + 3 * s ** 2
        d3 = (6 * s - 6 * s ** 2) / Le
        d4 = -2 * s + 3 * s ** 2
        Bw = np.array([0, d1, d2, 0, d3, d4])
        # w'' = d2N/dx2
        e1 = (-6 + 12 * s) / Le ** 2
        e2 = (-4 + 6 * s) / Le
        e3 = (6 - 12 * s) / Le ** 2
        e4 = (-2 + 6 * s) / Le
        Bww = np.array([0, e1, e2, 0, e3, e4])
        rows.append((Bu, Bw, Bww, Nu, Nw))
    return rows


def _elem_internal(qe, A, B, D, Le, rows):
    """Element internal force and tangent from the von Karman laminate energy."""
    f = np.zeros(6)
    Kt = np.zeros((6, 6))
    J = 0.5 * Le
    for (Bu, Bw, Bww, _Nu, _Nw), wg in zip(rows, _GW):
        up = Bu @ qe
        wp = Bw @ qe
        wpp = Bww @ qe
        eps = up + 0.5 * wp ** 2
        kap = -wpp
        N = A * eps + B * kap
        M = B * eps + D * kap
        deps = Bu + wp * Bw                       # d eps / dq
        dkap = -Bww
        wgt = wg * J
        f += wgt * (N * deps + M * dkap)
        Kt += wgt * (np.outer(deps, A * deps + B * dkap)
                     + np.outer(dkap, B * deps + D * dkap)
                     + N * np.outer(Bw, Bw))       # d2 eps / dq2 = Bw (x) Bw
    return f, Kt


def _elem_mass(rhoA, Le, rows):
    Me = np.zeros((6, 6))
    J = 0.5 * Le
    for (_Bu, _Bw, _Bww, Nu, Nw), wg in zip(rows, _GW):
        Me += wg * J * rhoA * (np.outer(Nu, Nu) + np.outer(Nw, Nw))
    return Me


class LaminatedBeam:
    """Clamp-clamp fiber-steered laminated von Karman beam."""

    def __init__(self, n_elem, length, thickness, width=1.0, base_angle=0.0):
        self.n_elem = int(n_elem)
        self.length = float(length)
        self.thickness = float(thickness)
        self.width = float(width)
        self.base_angle = float(base_angle)         # bottom ply angle (rad)
        self.Le = length / n_elem
        self.n_node = n_elem + 1
        self.ndof = 3 * self.n_node
        self.x = np.linspace(0.0, length, self.n_node)
        self.xe = 0.5 * (self.x[:-1] + self.x[1:])  # element centres
        self._rows = _shape_derivs(self.Le)
        # clamp both ends: u,w,w' = 0 at first and last node
        fixed = [0, 1, 2, self.ndof - 3, self.ndof - 2, self.ndof - 1]
        self.free = np.array([d for d in range(self.ndof) if d not in fixed])
        self.set_angles(np.zeros(self.n_elem))

    # -- laminate assembly -------------------------------------------------- #
    def set_ply_angles(self, ply_angles):
        """Set every ply's per-element angle field: ``ply_angles`` is
        ``(n_ply, n_elem)`` (rad), bottom ply first.

        This is the full laminate design variable -- each ply is steered
        independently along the beam, so ``A(x), B(x), D(x)`` are all shaped by
        the tow layout, not just a single steered ply over a fixed base.
        """
        pa = np.atleast_2d(np.asarray(ply_angles, float))
        if pa.shape[1] != self.n_elem:
            pa = np.broadcast_to(pa, (pa.shape[0], self.n_elem))
        self.ply_angles = pa.copy()
        self.A = np.empty(self.n_elem)
        self.B = np.empty(self.n_elem)
        self.D = np.empty(self.n_elem)
        rhoA = None
        for e in range(self.n_elem):
            A11, B11, D11, rhoA = beam_abd(pa[:, e], self.thickness, self.width)
            self.A[e], self.B[e], self.D[e] = A11, B11, D11
        self.rhoA = rhoA
        self._assemble_linear()
        return self

    def set_angles(self, theta):
        """Convenience: steer the top ply by ``theta`` over the fixed base ply."""
        theta = np.broadcast_to(np.asarray(theta, float), (self.n_elem,))
        self.theta = theta.copy()
        base = np.full(self.n_elem, self.base_angle)
        return self.set_ply_angles(np.vstack([theta, base]))

    def _assemble_linear(self):
        n = self.ndof
        M = np.zeros((n, n))
        K = np.zeros((n, n))
        for e in range(self.n_elem):
            d = np.array([3 * e, 3 * e + 1, 3 * e + 2,
                          3 * e + 3, 3 * e + 4, 3 * e + 5])
            ix = np.ix_(d, d)
            _f, Kt0 = _elem_internal(np.zeros(6), self.A[e], self.B[e],
                                     self.D[e], self.Le, self._rows)
            K[ix] += Kt0                            # linear stiffness = tangent at 0
            M[ix] += _elem_mass(self.rhoA, self.Le, self._rows)
        self.M_full, self.K_full = M, K

    # -- internal nonlinear force (full DOF) -------------------------------- #
    def internal_force(self, q_full):
        f = np.zeros(self.ndof)
        for e in range(self.n_elem):
            d = np.array([3 * e, 3 * e + 1, 3 * e + 2,
                          3 * e + 3, 3 * e + 4, 3 * e + 5])
            fe, _ = _elem_internal(q_full[d], self.A[e], self.B[e],
                                   self.D[e], self.Le, self._rows)
            f[d] += fe
        return f

    def free_index_of(self, global_dof):
        pos = np.where(self.free == global_dof)[0]
        if pos.size == 0:
            raise ValueError(f"dof {global_dof} is constrained")
        return int(pos[0])

    def midspan_w_dof(self):
        """Reduced-DOF index of the mid-span transverse displacement."""
        node = self.n_elem // 2
        return self.free_index_of(3 * node + 1)

    # -- PyHB system -------------------------------------------------------- #
    def to_system(self):
        """Return a :class:`pyhb.MechanicalSystem` in the reduced (free) DOFs."""
        ix = np.ix_(self.free, self.free)
        M = self.M_full[ix].copy()
        K = self.K_full[ix].copy()
        element = _LaminatedBeamForce(self)
        return MechanicalSystem(M, np.zeros_like(M), K, [element])


class _LaminatedBeamForce(GeneralNonlinearElement):
    """Nonlinear (beyond-linear) internal force of the laminated beam, reduced DOFs."""

    def __init__(self, beam: LaminatedBeam):
        self.beam = beam
        self.free = beam.free
        self.Kfree = beam.K_full[np.ix_(beam.free, beam.free)]
        self.n = beam.free.size

    def forces(self, q_time, u_time):
        beam = self.beam
        N = q_time.shape[0]
        nf = self.n
        f = np.zeros((N, nf))
        Kt = np.zeros((N, nf, nf))
        qfull = np.zeros(beam.ndof)
        for s in range(N):
            qfull[:] = 0.0
            qfull[self.free] = q_time[s]
            # full internal force & tangent
            fint = np.zeros(beam.ndof)
            Ktan = np.zeros((beam.ndof, beam.ndof))
            for e in range(beam.n_elem):
                d = np.array([3 * e, 3 * e + 1, 3 * e + 2,
                              3 * e + 3, 3 * e + 4, 3 * e + 5])
                fe, Kte = _elem_internal(qfull[d], beam.A[e], beam.B[e],
                                         beam.D[e], beam.Le, beam._rows)
                fint[d] += fe
                Ktan[np.ix_(d, d)] += Kte
            ff = fint[self.free]
            Kf = Ktan[np.ix_(self.free, self.free)]
            qf = q_time[s]
            f[s] = ff - self.Kfree @ qf              # nonlinear part only
            Kt[s] = Kf - self.Kfree
        return f, Kt, np.zeros((N, nf, nf))
