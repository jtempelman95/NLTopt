r"""How sensitive is the steered design to the way it would actually be laid?  (Reviewer R1, M1.)

The design is an IDEALIZED tow field: one angle per point, constant thickness, full
coverage.  Laying it with tows of constant width along the streamlines of the director
t = (cos theta, sin theta) changes the local tow density: neighbouring streamlines
spread or converge at the rate  d ln(spacing)/dl = div t,  so a tow course that starts
fully covering its inflow edge reaches the point x with relative areal density

    tau(x) = exp( - int_{inflow}^{x} div t  dl )        (integrated back along the streamline).

tau > 1 means converging tows (overlaps), tau < 1 diverging tows (gaps).  This is the
continuum (narrow-tow) limit of the defect-layer picture.  Three bounding as-built laminates
follow from tau, all with the same per-point fiber angle:

  CTS    all tows continue, thickness h tau (continuous-tow-shearing-like; mass varies)
  GAP    0 % coverage: tows are cut where they would overlap, gaps are resin-filled;
         constant thickness h, fiber fraction min(1, tau)
  OVL    100 % coverage: tows are added where they would gap, overlaps are kept;
         thickness h max(1, tau)

Each is evaluated (J, omega_1, mass) with the same evaluator and compared with UD (tau = 1).
The uncut (CTS) course is far from buildable for this field (tau spans more than two decades),
so the design REQUIRES tow cut/add.  With cut/add inside courses of N tows, the coverage
mismatch at a course boundary never exceeds one tow width and averages half of one, so the
defect area fraction is about 1/(2N) wherever at least one cut/add has occurred upstream:

  f_d(x) = (1/(2N)) min(1, |ln tau(x)| / ln(1 + 1/N)),

as resin-filled gaps (0 % coverage, constant thickness, fiber fraction 1 - f_d) or as
overlaps (100 % coverage, thickness 1 + f_d), for N = 4, 8, 16 tows per course.
A fourth test perturbs the laid angle by a smooth random error (sigma = 2 deg, correlation
length ~ 25 mm; 40 samples).

    python examples/r1_asbuilt.py       -> results/r1_asbuilt.npz
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_WAVETOPO = _os.environ.get("WAVETOPO_DIR", _os.path.join(_os.path.dirname(_ROOT), "wavetopo"))
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, _os.path.join(_ROOT, "examples")); sys.path.insert(0, _WAVETOPO)
import levers_study as LS
from design_source import heldfreq_design
from nltopt.relaxed_plate import SteeredPlate, rotate
from nltopt.laminate import lamina_Q, RHO
from wavetopo.cfrp import CSRBFMapping

P = LS.PLATE
Q = lamina_Q()
E_r, nu_r, RHO_R = 3.5e9, 0.35, 1200.0                         # epoxy in the gaps
Q_r = E_r / (1 - nu_r ** 2) * np.array([[1, nu_r, 0], [nu_r, 1, 0], [0, 0, (1 - nu_r) / 2]])
th_hat = heldfreq_design()

# ------------------------------------------------------------- director on a fine grid
NXF, NYF = 401, 301
xs = np.linspace(0, P.a, NXF); ys = np.linspace(0, P.b, NYF)
GX, GY = np.meshgrid(xs, ys)


class _D: pass


m = _D(); m.cx, m.cy, m.N = GX.ravel(), GY.ravel(), GX.size
SX, SY = np.meshgrid(np.linspace(0, P.a, 5), np.linspace(0, P.b, 4))
R = CSRBFMapping(m, (SX.ravel(), SY.ravel()), r_s=0.075)
TH = (R.Phi @ th_hat).reshape(NYF, NXF)
THX = (R.Phi_x @ th_hat).reshape(NYF, NXF); THY = (R.Phi_y @ th_hat).reshape(NYF, NXF)
DIV = -np.sin(TH) * THX + np.cos(TH) * THY
I_th = RegularGridInterpolator((ys, xs), TH)
I_div = RegularGridInterpolator((ys, xs), DIV)


def tau_at(px, py, ds=2.5e-4, Lmax=1.0):
    """Integrate div t back along the streamline from (px, py) to the inflow boundary."""
    x, y = px.copy(), py.copy()
    acc = np.zeros_like(x); alive = np.ones(x.size, bool); L = np.zeros_like(x)
    for _ in range(int(Lmax / ds)):
        if not alive.any():
            break
        pts = np.c_[y[alive], x[alive]]
        t = I_th(pts)
        xm = x[alive] - 0.5 * ds * np.cos(t); ym = y[alive] - 0.5 * ds * np.sin(t)
        xm = np.clip(xm, 0, P.a); ym = np.clip(ym, 0, P.b)
        tm = I_th(np.c_[ym, xm])
        acc[alive] += ds * I_div(np.c_[ym, xm])
        x[alive] = x[alive] - ds * np.cos(tm); y[alive] = y[alive] - ds * np.sin(tm)
        L[alive] += ds
        out = (x < 0) | (x > P.a) | (y < 0) | (y > P.b)
        alive &= ~out
        x = np.clip(x, 0, P.a); y = np.clip(y, 0, P.b)
    return np.exp(-acc), alive, L


tau, stuck, Lstream = tau_at(P.qx.copy(), P.qy.copy())
print(f"tow density tau on the plate: min {tau.min():.3f}  max {tau.max():.3f}  "
      f"area with gaps (tau<0.95) {100*np.mean(tau<0.95):.1f}%  overlaps (tau>1.05) {100*np.mean(tau>1.05):.1f}%  "
      f"untraced {stuck.sum()}", flush=True)

sp = SteeredPlate(P)
theta_q = LS.PHI @ th_hat
UD = sp.forward(0.0, full=False)
ideal = sp.forward(theta_q, full=False)
QB = rotate(Q, theta_q); QR = np.broadcast_to(Q_r, QB.shape)


def as_built(thick, phi):
    """thick: thickness factor; phi: fiber fraction of the material (rest resin)."""
    h = P.h * thick
    C = phi[:, None, None] * QB + (1 - phi)[:, None, None] * QR
    P.set_mass(h * (phi * RHO + (1 - phi) * RHO_R))
    st = P.evaluate((h ** 3 / 12)[:, None, None] * C, h[:, None, None] * C, full=False)
    mass = np.sum(P.w * h * (phi * RHO + (1 - phi) * RHO_R)) / (RHO * P.h * P.a * P.b)
    P.set_mass(None)
    return st["J"] / UD["J"], st["omega1"] / UD["omega1"], mass


one = np.ones_like(tau)
res = {"ideal": (ideal["J"] / UD["J"], ideal["omega1"] / UD["omega1"], 1.0),
       "CTS": as_built(tau, one),
       "GAP": as_built(one, np.minimum(1.0, tau)),
       "OVL": as_built(np.maximum(1.0, tau), one)}
for N in (4, 8, 16):
    fd = 0.5 / N * np.minimum(1.0, np.abs(np.log(tau)) / np.log(1 + 1.0 / N))
    res[f"GAP{N}"] = as_built(one, 1 - fd)
    res[f"OVL{N}"] = as_built(1 + fd, one)
for k, v in res.items():
    print(f"{k:5s}: J/J_UD={v[0]:.4f}  f/f_UD={v[1]:.4f}  mass/mass_UD={v[2]:.4f}", flush=True)

# ------------------------------------------------------------- random angle error
rng = np.random.default_rng(7)
MC = []
for k in range(40):
    n = gaussian_filter(rng.standard_normal((NYF, NXF)), sigma=25e-3 / (P.a / (NXF - 1)), mode="reflect")
    n *= np.deg2rad(2.0) / n.std()
    In = RegularGridInterpolator((ys, xs), n)
    st = sp.forward(theta_q + In(np.c_[P.qy, P.qx]), full=False)
    MC.append((st["J"] / UD["J"], st["omega1"] / UD["omega1"]))
MC = np.array(MC)
print(f"angle error 2 deg RMS (40 samples): J/J_UD {MC[:,0].mean():.4f} +- {MC[:,0].std():.4f} "
      f"(range {MC[:,0].min():.4f}-{MC[:,0].max():.4f}); f/f_UD {MC[:,1].mean():.4f} +- {MC[:,1].std():.4f}", flush=True)
np.savez(_os.path.join(_ROOT, "results", "r1_asbuilt.npz"), tau=tau, qx=P.qx, qy=P.qy, nx=P.nx, ny=P.ny,
         res_keys=list(res), res=np.array(list(res.values())), MC=MC, untraced=stuck.sum())
print("wrote results/r1_asbuilt.npz")
