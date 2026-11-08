"""Lab-frame aerodynamic force and torque (njit) - port of booster_forces_lab.aero_lab.

Conventions (booster_forces_lab.py):
    theta_lab  body axis from the upward vertical, positive toward +X
    Omega      velocity from the DOWNWARD vertical toward +X:  v_hat = (sin Om, -cos Om)
    theta = theta_lab + Omega   body vs flow;   alpha = gamma + Omega   fin vs flow
    delta = gamma - theta_lab   fin deflection;  rotational EOM  I thetaddot_lab = -tau_z
    u = -v_hat = (-sin Om, cos Om) flow direction,  n = (cos Om, sin Om) cross-flow
    F_B   = q A   (Cpm u + (Cpm - C_Ntheta) theta n)
    F_f   = n_f q A_f (CA0 u + (CA0 - CNa) alpha n),   CA0 = sigma Cpm,  CNa = (c/s) 4 / sqrt(M^2 - 1)
    tau_z = -q A r_a C_Ntheta theta + n_f q A_f r_f (CNa alpha - CA0 delta)
Extensions needed to fly the whole trajectory (forces_lab is valid for M >= 1.41, |theta| < 10 deg):
  * CNa uses max(M, M_clamp), M_clamp = sqrt(1 + (c/s)^2): bounded, constant below M_clamp (subsonic
    landing burn). This is an assumption, not a fin model.
  * |theta| > 90 deg (flying top-first: ascent after separation) -> axial drag only, no torque.
  * r_a = CM height above the engine face (from mass properties), r_f = h_fins - r_a.
"""
import math

from atmosphere import cp_max, density_dyn, sound_speed
from jit import njit
from params import A_FIN, C_NTH, C_OVER_S, D, GAMMA, H_FIN, M_CLAMP, N_FIN, SIGMA


@njit(cache=True)
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def aero_lab(th_lab, Om, gam, z, V, r_a, P):
    """(F_X, F_Y, tau_z, q, theta) with theta = wrapped body-vs-flow angle."""
    if V < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    q = 0.5 * density_dyn(z, P) * V * V
    M = V / sound_speed(P)
    Cpm = cp_max(M, P[GAMMA])
    A = 0.25 * math.pi * P[D] ** 2
    Af, nf = P[A_FIN], P[N_FIN]
    CA0 = P[SIGMA] * Cpm
    ux, uy = -math.sin(Om), math.cos(Om)
    nx, ny = math.cos(Om), math.sin(Om)
    theta = wrap(th_lab + Om)
    if abs(theta) > 0.5 * math.pi:                     # top-first: axial drag only
        Fa = q * (A * Cpm + nf * Af * CA0)
        return Fa * ux, Fa * uy, 0.0, q, theta
    alpha = wrap(gam + Om)
    delta = wrap(gam - th_lab)
    Mc = max(M, P[M_CLAMP])
    CNa = P[C_OVER_S] * 4.0 / math.sqrt(Mc * Mc - 1.0)
    CN = P[C_NTH]
    FBu, FBn = q * A * Cpm, q * A * (Cpm - CN) * theta
    Ffu, Ffn = nf * q * Af * CA0, nf * q * Af * (CA0 - CNa) * alpha
    Fx = (FBu + Ffu) * ux + (FBn + Ffn) * nx
    Fy = (FBu + Ffu) * uy + (FBn + Ffn) * ny
    r_f = P[H_FIN] - r_a
    tau = -q * A * r_a * CN * theta + nf * q * Af * r_f * (CNa * alpha - CA0 * delta)
    return Fx, Fy, tau, q, theta
