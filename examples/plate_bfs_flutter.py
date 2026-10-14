r"""Panel flutter on the CONFORMING BFS FE (correct for bend-twist-coupled laminates):
the flutter boundary q_cr reduced on the FE's own modes, and its analytic adjoint
dq_cr/dtheta_hat (implicit eigenvalue sensitivity, element-curvature contraction).
This replaces the assumed-modes flutter, which is wrong once D16,D26 are nonzero."""
import numpy as np
from scipy.linalg import eigh, eig
from wavetopo.cfrp import CSRBFMapping
from nltopt.laminate import Qbar, RHO
import plate_bfs as BFS
import plate_flutter as PF

A, B, H, CB, Q0 = BFS.A, BFS.B, BFS.H, BFS.CB, BFS.Q0
BETA, C0 = BFS.BETA, PF.C0


def dQbar_dtheta(theta, Q=Q0):
    m, n = np.cos(theta), np.sin(theta)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    m2, n2 = m*m, n*n; m4, n4 = m2*m2, n2*n2
    dm2n2 = 2*m*n*(m2-n2); dm4 = -4*m2*m*n; dn4 = 4*n2*n*m; dm4n4 = dm4+dn4
    d = np.zeros((3, 3))
    d[0, 0] = Q11*dm4 + 2*(Q12+2*Q66)*dm2n2 + Q22*dn4
    d[1, 1] = Q11*dn4 + 2*(Q12+2*Q66)*dm2n2 + Q22*dm4
    d[0, 1] = d[1, 0] = (Q11+Q22-4*Q66)*dm2n2 + Q12*dm4n4
    d[2, 2] = (Q11+Q22-2*Q12-2*Q66)*dm2n2 + Q66*dm4n4
    dm3n = m4-3*m2*n2; dmn3 = 3*m2*n2-n4
    d[0, 2] = d[2, 0] = (Q11-Q12-2*Q66)*dm3n + (Q12-Q22+2*Q66)*dmn3
    d[1, 2] = d[2, 1] = (Q11-Q12-2*Q66)*dmn3 + (Q12-Q22+2*Q66)*dm3n
    return d


class Mesh:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny; self.lx, self.ly = A/nx, B/ny
        nnode = (nx+1)*(ny+1); self.ndof = 4*nnode
        nid = lambda ix, iy: iy*(nx+1)+ix
        self.edofs = []; cx = []; cy = []
        for iy in range(ny):
            for ix in range(nx):
                nodes = [nid(ix, iy), nid(ix+1, iy), nid(ix+1, iy+1), nid(ix, iy+1)]
                self.edofs.append(np.array([[4*n, 4*n+1, 4*n+2, 4*n+3] for n in nodes]).ravel())
                cx.append((ix+0.5)*self.lx); cy.append((iy+0.5)*self.ly)
        self.edofs = np.array(self.edofs); self.nel = len(self.edofs)
        # gauss templates (constant across the uniform mesh)
        BBg = []; Nwg = []; Nwxg = []; wqg = []
        for i, s in enumerate(BFS._S):
            vx, d1x, d2x = BFS._herm(s, self.lx)
            for j, t in enumerate(BFS._S):
                vy, d1y, d2y = BFS._herm(t, self.ly)
                wqg.append(BFS._W[i]*BFS._W[j]*self.lx*self.ly)
                Nwg.append([vx[a]*vy[b] for (a, b) in BFS._PAIRS])
                Nwxg.append([d1x[a]*vy[b] for (a, b) in BFS._PAIRS])
                BBg.append(np.array([[d2x[a]*vy[b], vx[a]*d2y[b], 2*d1x[a]*d1y[b]] for (a, b) in BFS._PAIRS]).T)
        self.BBg = np.array(BBg); self.Nwg = np.array(Nwg); self.Nwxg = np.array(Nwxg); self.wqg = np.array(wqg)
        fix = []
        for ix in range(nx+1):
            for iy in range(ny+1):
                n = nid(ix, iy)
                if ix in (0, nx) or iy in (0, ny):
                    fix.append(4*n)
                    if ix in (0, nx): fix.append(4*n+2)
                    if iy in (0, ny): fix.append(4*n+1)
        self.free = np.setdiff1d(np.arange(self.ndof), np.unique(fix))
        self.f2f = -np.ones(self.ndof, int); self.f2f[self.free] = np.arange(len(self.free))
        self.Me = np.einsum('g,gi,gj->ij', self.wqg*RHO*H, self.Nwg, self.Nwg)
        self.Ge = np.einsum('g,gi,gj->ij', self.wqg, self.Nwg, self.Nwxg)
        Mf = np.zeros((self.ndof, self.ndof)); Gf = np.zeros_like(Mf)
        for ed in self.edofs:
            Mf[np.ix_(ed, ed)] += self.Me; Gf[np.ix_(ed, ed)] += self.Ge
        self.M = Mf[np.ix_(self.free, self.free)]; self.G = Gf[np.ix_(self.free, self.free)]
        class DM: pass
        dm = DM(); dm.cx = np.array(cx); dm.cy = np.array(cy); dm.N = self.nel
        self.PhiD = CSRBFMapping(dm, (PF.P.rbf.sx, PF.P.rbf.sy), r_s=PF.P.rbf.r_s).Phi   # (nel,M)

    def theta_e(self, th_hat):
        return self.PhiD @ th_hat

    def K(self, theta_e):
        Kf = np.zeros((self.ndof, self.ndof))
        for e, ed in enumerate(self.edofs):
            D = CB*Qbar(theta_e[e], Q0)
            Ke = np.einsum('g,gai,ab,gbj->ij', self.wqg, self.BBg, D, self.BBg)
            Kf[np.ix_(ed, ed)] += Ke
        return Kf[np.ix_(self.free, self.free)]


def build_basis(mesh, th_ref, nr=40):
    """Fixed reduced basis: lowest nr FE modes of the reference design (M is
    theta-independent, so Phi stays M-orthonormal => Mr=I for any design)."""
    w2, Phi = eigh(mesh.K(mesh.theta_e(th_ref)), mesh.M); Phi = Phi[:, :nr]
    return {"Phi": Phi, "Gr": Phi.T @ mesh.G @ Phi}


def flutter(mesh, th_hat, basis, grad=True):
    Phi = basis["Phi"]; Gr = basis["Gr"]; nr = Phi.shape[1]
    te = mesh.theta_e(th_hat); K = mesh.K(te)
    Kr = Phi.T @ K @ Phi; Cr = C0*np.eye(nr)
    f1 = np.sqrt(np.linalg.eigvalsh(Kr).min())/(2*np.pi)       # Mr=I
    Z, I = np.zeros((nr, nr)), np.eye(nr)
    mre = lambda q: eig(np.block([[Z, I], [-(Kr+BETA*q*Gr), -Cr]]), right=False).real.max()
    hi = 1e3
    while mre(hi) < 0:
        hi *= 2
    lo = 0.0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        if mre(mid) > 0: hi = mid
        else: lo = mid
    qc = 0.5*(lo+hi)
    if not grad:
        return qc, f1
    Ar = np.block([[Z, I], [-(Kr+BETA*qc*Gr), -Cr]])
    w, vl, vr = eig(Ar, left=True, right=True)
    c = int(np.argmax(w.real)); vv = vr[:, c]; uu = vl[:, c]
    uu = uu/np.conjugate(np.vdot(uu, vv))                       # uu* vv = 1 (Br=I)
    v1 = vv[:nr]; u2 = uu[nr:]
    phi_v = Phi @ v1; phi_u = Phi @ u2
    dsdq = -BETA*np.vdot(u2, Gr @ v1)                           # u2*(-beta Gr) v1
    gfield = np.zeros(mesh.nel, complex)
    for e, ed in enumerate(mesh.edofs):
        fe = mesh.f2f[ed]; mm = fe >= 0
        du = np.zeros(16, complex); dv = np.zeros(16, complex)
        du[mm] = np.conjugate(phi_u[fe[mm]]); dv[mm] = phi_v[fe[mm]]
        ku = np.einsum('gai,i->ga', mesh.BBg, du); kv = np.einsum('gai,i->ga', mesh.BBg, dv)
        gfield[e] = np.einsum('g,ga,ab,gb->', mesh.wqg, ku, CB*dQbar_dtheta(te[e]), kv)
    grad_hat = mesh.PhiD.T @ gfield                            # phi_u* dK/dtheta_hat phi_v
    dsdtheta = -grad_hat                                        # u2*(-dKr/dtheta)v1
    dqc = -(dsdtheta.real)/(dsdq.real)
    return qc, f1, dqc


def w1sq_and_grad(mesh, th_hat, basis):
    Phi = basis["Phi"]; te = mesh.theta_e(th_hat)
    Kr = Phi.T @ mesh.K(te) @ Phi
    vals, vecs = eigh(Kr, np.eye(Phi.shape[1]))
    phi = Phi @ vecs[:, 0]
    gfield = np.zeros(mesh.nel)
    for e, ed in enumerate(mesh.edofs):
        fe = mesh.f2f[ed]; mm = fe >= 0
        d = np.zeros(16); d[mm] = phi[fe[mm]]
        kap = np.einsum('gai,i->ga', mesh.BBg, d)
        gfield[e] = np.einsum('g,ga,ab,gb->', mesh.wqg, kap, CB*dQbar_dtheta(te[e]), kap)
    return vals[0], mesh.PhiD.T @ gfield


def uniform_front(mesh, degs, nr=40):
    out = []
    for d in degs:
        th = np.zeros(PF.P.M); th[:] = 0.0
        te = np.full(mesh.nel, np.deg2rad(d))
        w2, Phi = eigh(mesh.K(te), mesh.M); Phi = Phi[:, :nr]
        basis = {"Phi": Phi, "Gr": Phi.T @ mesh.G @ Phi}
        # flutter with this design as its own basis: reduce K(te) again (=diag)
        Kr = np.diag(w2[:nr]); Cr = C0*np.eye(nr); Gr = basis["Gr"]
        Z, I = np.zeros((nr, nr)), np.eye(nr)
        mre = lambda q: eig(np.block([[Z, I], [-(Kr+BETA*q*Gr), -Cr]]), right=False).real.max()
        hi = 1e3
        while mre(hi) < 0: hi *= 2
        lo = 0.0
        for _ in range(50):
            mid = 0.5*(lo+hi); lo, hi = (lo, mid) if mre(mid) > 0 else (mid, hi)
        out.append((d, np.sqrt(w2[0])/(2*np.pi), 0.5*(lo+hi)))
    return out


if __name__ == "__main__":
    mesh = Mesh(12, 9)
    rng = np.random.default_rng(1)
    th = np.deg2rad(25.0)*np.ones(PF.P.M) + np.deg2rad(6.0)*rng.standard_normal(PF.P.M)
    basis = build_basis(mesh, th)
    qc, f1, g = flutter(mesh, th, basis)
    print(f"BFS flutter (fixed basis): q_cr={qc/1e3:.1f} kPa, f1={f1:.0f} Hz")
    h = 1e-4; errs = []
    print(f"  {'k':>3} {'analytic':>14} {'FD central':>14} {'rel err':>10}")
    for k in rng.choice(PF.P.M, 6, replace=False):
        e = np.zeros(PF.P.M); e[k] = h
        qp = flutter(mesh, th+e, basis, grad=False)[0]; qm = flutter(mesh, th-e, basis, grad=False)[0]
        fd = (qp-qm)/(2*h); err = abs(g[k]-fd)/(abs(fd)+1e-30); errs.append(err)
        print(f"  {k:>3} {g[k]:>14.4e} {fd:>14.4e} {err:>10.2e}")
    print(f"  max rel err = {max(errs):.2e}  {'PASS' if max(errs)<2e-3 else 'FAIL'}")
    # w1sq gradient FD check
    w1sq, gw = w1sq_and_grad(mesh, th, basis)
    ew = []
    for k in rng.choice(PF.P.M, 4, replace=False):
        e = np.zeros(PF.P.M); e[k] = h
        fd = (w1sq_and_grad(mesh, th+e, basis)[0] - w1sq_and_grad(mesh, th-e, basis)[0])/(2*h)
        ew.append(abs(gw[k]-fd)/(abs(fd)+1e-30))
    print(f"  w1sq grad FD max rel err = {max(ew):.2e}")
    print("\nBFS uniform front (deg, f1 Hz, q_cr kPa):")
    for d, f, q in uniform_front(mesh, range(0, 91, 10)):
        print(f"  {d:3d}  {f:6.0f}  {q/1e3:8.0f}")
