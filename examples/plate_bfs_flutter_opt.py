r"""Redo the flutter design on the CONFORMING BFS FE: steer the tow field to MAXIMIZE
the (correct) flutter boundary q_cr at held f1, curl-feasible -- recovering the flutter
margin the bend-twist coupling craters for a uniform layup."""
import sys; sys.path.insert(0, "/home/jrt/wavetopo")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from wavetopo.cfrp_optimizer import MMA
import plate_bfs_flutter as BF
import plate_flutter as PF

P = PF.P
MESH = None


def _mesh():
    global MESH
    if MESH is None:
        MESH = BF.Mesh(12, 9)
    return MESH


def optimize(w1sq_tgt, zeta_all, seed_deg, iters=45, nr=40):
    mesh = _mesh()
    th = np.full(P.M, np.deg2rad(seed_deg))
    basis = BF.build_basis(mesh, th, nr)
    q0 = BF.flutter(mesh, th, basis, grad=False)[0]
    lam = np.full(P.N, 12.0); lam_f = 0.0; mu = 5.0
    mma = MMA(np.full(P.M, -np.pi/2), np.full(P.M, np.pi/2))
    best = None
    for it in range(iters):
        basis = BF.build_basis(mesh, th, nr)                    # refresh basis at current design
        for _ in range(3):
            qc, f1, dq = BF.flutter(mesh, th, basis)
            w1sq, dw1 = BF.w1sq_and_grad(mesh, th, basis)
            gth = -(dq/q0)
            zeta = np.cos(P.Phi@th)*(P.Phi_x@th) + np.sin(P.Phi@th)*(P.Phi_y@th)
            gc = zeta**2 - zeta_all**2
            fac = (lam + mu*np.maximum(gc, -lam/mu))*(gc > -lam/mu)
            gth = gth + (1.0/P.N)*(P.curl_jac(th, P.Phi@th).T @ (2*zeta*fac))
            h = w1sq/w1sq_tgt - 1.0
            gth = gth + (lam_f + mu*h)*(dw1/w1sq_tgt)
            th = np.clip(mma.update(th, gth), -np.pi/2, np.pi/2)
        qc, f1, _ = BF.flutter(mesh, th, basis)
        w1sq = BF.w1sq_and_grad(mesh, th, basis)[0]
        zeta = np.cos(P.Phi@th)*(P.Phi_x@th) + np.sin(P.Phi@th)*(P.Phi_y@th)
        zmax = float(np.max(np.abs(zeta))); h = w1sq/w1sq_tgt - 1.0
        if (zmax/zeta_all < 1.02) and (abs(h) < 6e-3) and (best is None or qc > best[0]):
            best = (qc, th.copy(), zmax, np.sqrt(w1sq)/(2*np.pi))
        gc = zeta**2 - zeta_all**2
        lam = np.maximum(lam + mu*np.maximum(gc, -lam/mu), 0.0)
        lam_f = lam_f + mu*h
        if best is None or zmax/zeta_all >= 1.02 or abs(h) >= 6e-3:
            mu = min(1.3*mu, 1e7)
    return best


def _task(a):
    return optimize(*a)


if __name__ == "__main__":
    mesh = _mesh()
    front = BF.uniform_front(mesh, range(0, 91, 3))
    ff = np.array([p[1] for p in front]); fq = np.array([p[2] for p in front])
    for ftgt in (300.0, 320.0):
        bu = fq[np.abs(ff - ftgt) < 8].max()/1e3
        w1sq_t = (2*np.pi*ftgt)**2
        tasks = [(w1sq_t, 12.0, s) for s in (0, 20, 50)]
        with ProcessPoolExecutor(max_workers=6) as ex:
            res = [r for r in ex.map(_task, tasks) if r is not None]
        if res:
            b = max(res, key=lambda r: r[0])
            print(f"held f1={ftgt:.0f}Hz: best-uniform q_cr={bu:.0f}kPa | STEERED {b[0]/1e3:.0f}kPa "
                  f"@f1={b[3]:.0f} (curl {b[2]:.1f}) -> {b[0]/1e3/bu:.1f}x")
            np.save(f"bfs_steered_{int(ftgt)}.npy", b[1])
        else:
            print(f"held f1={ftgt:.0f}Hz: no feasible steered design")
