"""Planar (3-DOF) equations of motion with thrust, gravity, aerodynamics and LQR-controlled grid fins (njit).

State S = [x, y, theta_lab, xdot, ydot, thetadot_lab, m_fuel];  throttle beta in {0, 1}.
    m       = m_fuel + m_dry,   F_t = beta F_max  along the body axis a = (sin theta_lab, cos theta_lab)
    m xddot = F_t sin(theta_lab + tvc) + F_X
    m yddot = F_t cos(theta_lab + tvc) + F_Y - m g(y)
    I thetaddot_lab = -tau_z + r_a F_t sin(tvc)                 (sign convention of forces_lab; tvc from control)
    mdot_fuel = -F_t / v_ex
(F_X, F_Y, tau_z) from aero.aero_lab; fin angle gamma = theta_lab + delta; (delta, tvc) from control().

Attitude control (optimal control of a 2nd-order linear oscillator, gain-scheduled on the state)
----------------------------------------------------------------------------------------------
Actuators: grid-fin deflection delta (aero, authority ~ q) and engine gimbal tvc (only while burning).
Linearising aero_lab's torque in theta = theta_lab + Omega (body vs flow), alpha = theta + delta:
    tau_z = -q A r_a C_Ntheta theta + n_f q A_f r_f (CNa (theta + delta) - CA0 delta)
    thetaddot = -tau_z / I + r_a F_t tvc / I = a theta + b_f delta + b_t tvc,
    a   = q (A r_a C_Ntheta - n_f A_f r_f CNa) / I,   b_f = -q n_f A_f r_f (CNa - CA0) / I,
    b_t = r_a F_t / I                                  (a = b_f = 0 when flying top-first)
(Omega frozen over a control step). Reference attitude theta_lab,ref = -w Omega,
w = clip((V - v_vert) / (v_blend - v_vert), 0, 1): engine-first along the flow (retrograde, a gravity
turn) above v_blend, vertical below v_vert. With e = theta_lab - theta_lab,ref:
    eddot = a e + a (theta_lab,ref + Omega) + b_f delta + b_t tvc
LQR, J = int (q1 e^2 + q2 edot^2 + r_f delta^2 + r_t tvc^2) dt, Bryson weights q1 = 1/e_max^2,
q2 = 1/edot_max^2, r_f = 1/delta_max^2, r_t = 1/tvc_max^2. With B = [0 0; b_f b_t] and
s = b_f^2/r_f + b_t^2/r_t the Riccati equation A'P + PA - P B R^-1 B'P + Q = 0 is the scalar-input one:
    p12 = (a + sqrt(a^2 + s q1)) / s,   p22 = sqrt((2 p12 + q2) / s)
    delta = (b_f/r_f) v,  tvc = (b_t/r_t) v,  v = -(p12 e + p22 edot) - a (theta_lab,ref + Omega) / s
(the last term is the minimum-effort feed-forward cancelling the off-reference aero torque), and the
closed loop is the damped oscillator  eddot + 2 zeta w_n edot + w_n^2 e = 0 with
    w_n^2 = sqrt(a^2 + s q1),   2 zeta w_n = sqrt(2 (a + w_n^2) + s q2).
Bandwidth cap: Bryson weights alone make the loop as fast as the fins allow (w_n of several rad/s at
q ~ 50-90 kPa) - faster than real fin actuators, and it forces ~0.08 s RK steps (stiffness). So q1, q2
are reduced when needed so that w_n <= w_max and zeta <= zeta_max:
    q1 <- min(q1, (w_max^4 - a^2)/s),   q2 <- min(q2, (4 zeta_max^2 w_n^2 - 2(a + w_n^2))/s)
(from the two closed-loop formulas above; the result is still the LQR optimum for the reduced weights).
Both outputs are clipped to +-delta_max, +-tvc_max. tvc_max = 0 (default): fins only, no gimbal.
Above H_sep (climbing, or not yet fallen back through H_sep) the fins are stowed and the attitude is set by
the cold-gas RCS (rcs()): held while climbing, then flipped to the tangent attitude before the descent
starts. Without it, q ~ 50-120 Pa over ~100 s would turn the free body by radians.
Diagnostics per node: q, q|sin(alpha_b)| (alpha_b = angle between body axis and velocity line),
sensed acceleration |a - g| / g0, gamma, theta (body vs flow).
"""
import math

import numpy as np

from aero import aero_lab, wrap
from atmosphere import cp_max, density_dyn, gravity, sound_speed
from jit import njit
from params import (A_FIN, C_NTH, C_OVER_S, CTRL_ON, CTRL_Q1, CTRL_Q2, CTRL_R, CTRL_RT, CTRL_WMAX, CTRL_ZMAX, D, DELTA_MAX, ERR_STATE,
                    ERR_TANK, ERR_TUMBLE, FMAX, G0, GAMMA, H_FIN, H_SEP, M_CLAMP, MDRY, N_FIN, OK, Q_TUMBLE,
                    RCS_ALPHA, RCS_TARGET, RCS_WMAX, RCS_WN, SIGMA, TUMBLE, TVC_MAX, V_BLEND, V_VERT, VEX)
from vehicle import inertia, r_cm


@njit(cache=True)
def plant(S, P, I, r_a):
    """(a, b, Omega) of thetaddot = a theta + b delta at the current state."""
    xd, yd = S[3], S[4]
    V = math.sqrt(xd * xd + yd * yd)
    Om = math.atan2(xd, -yd)
    q = 0.5 * density_dyn(max(S[1], 0.0), P) * V * V
    M = V / sound_speed(P)
    CA0 = P[SIGMA] * cp_max(M, P[GAMMA])
    Mc = max(M, P[M_CLAMP])
    CNa = P[C_OVER_S] * 4.0 / math.sqrt(Mc * Mc - 1.0)
    A = 0.25 * math.pi * P[D] ** 2
    nAf, r_f = P[N_FIN] * P[A_FIN], P[H_FIN] - r_a
    a = q * (A * r_a * P[C_NTH] - nAf * r_f * CNa) / I
    b = -q * nAf * r_f * (CNa - CA0) / I
    return a, b, Om


@njit(cache=True)
def attitude_hold(S, P):
    """True until the booster falls back through H_sep: attitude set by the cold-gas RCS (rcs()),
    fins stowed. Below H_sep on the way down the fins (and gimbal) take over."""
    return S[1] >= P[H_SEP] or S[4] > 0.0


@njit(cache=True)
def rcs(S, P):
    """thetaddot from the RCS above H_sep (aero torque there neglected: q < ~120 Pa).
    Climbing (ydot > 0): hold, thetaddot = 0. After apex: slew to theta_target = P[RCS_TARGET]
    with |thetaddot| <= alpha:  commanded rate  w_c = -sign(e) min(0.9 sqrt(2 alpha |e|), w_n |e|, w_max)
    (the sqrt branch is the fastest rate from which alpha can still stop at e = 0 -> near bang-bang
    slew without overshoot; the linear branch is the final approach), thetaddot = clip(2 w_n (w_c - thetadot)).
    Near the target this is  e'' + 2 w_n e' + 2 w_n^2 e = 0  (zeta = 1/sqrt 2)."""
    if S[4] > 0.0:
        return 0.0
    e = wrap(S[2] - P[RCS_TARGET])
    al, wn = P[RCS_ALPHA], P[RCS_WN]
    wc = -math.copysign(min(0.9 * math.sqrt(2.0 * al * abs(e)), wn * abs(e), P[RCS_WMAX]), e)
    u = 2.0 * wn * (wc - S[5])
    return min(al, max(-al, u))


@njit(cache=True)
def control(S, beta, P, I, r_a):
    """(delta, tvc) [rad]: fin deflection (body frame) and engine gimbal angle, joint LQR (see top)."""
    if P[CTRL_ON] == 0.0 or attitude_hold(S, P):
        return 0.0, 0.0
    V = math.sqrt(S[3] * S[3] + S[4] * S[4])
    if V < 1e-6:
        return 0.0, 0.0
    a, bf, Om = plant(S, P, I, r_a)
    if abs(wrap(S[2] + Om)) > 0.5 * math.pi:          # top-first: no aero torque in the model
        a, bf = 0.0, 0.0
    bt = r_a * P[FMAX] * beta / I                      # gimbal: thetaddot = (r_a F_t / I) tvc
    rf, rt, q1, q2 = P[CTRL_R], P[CTRL_RT], P[CTRL_Q1], P[CTRL_Q2]
    s = bf * bf / rf + (bt * bt / rt if P[TVC_MAX] > 0.0 else 0.0)
    if s < 1e-18:
        return 0.0, 0.0
    w = min(1.0, max(0.0, (V - P[V_VERT]) / (P[V_BLEND] - P[V_VERT])))
    th_ref = -Om * w
    e = wrap(S[2] - th_ref)
    wm2 = P[CTRL_WMAX] * P[CTRL_WMAX]                  # bandwidth cap: w_n <= w_max, zeta <= zeta_max
    if a * a + s * q1 > wm2 * wm2:
        q1 = max(0.0, (wm2 * wm2 - a * a) / s)
    wn2 = math.sqrt(a * a + s * q1)
    q2 = min(q2, max(0.0, (4.0 * P[CTRL_ZMAX] ** 2 * wn2 - 2.0 * (a + wn2)) / s))
    p12 = (a + wn2) / s
    p22 = math.sqrt((2.0 * p12 + q2) / s)
    v = -(p12 * e + p22 * S[5]) - a * wrap(th_ref + Om) / s      # feedback + feed-forward, per unit b/r
    dm, tm = P[DELTA_MAX], P[TVC_MAX]
    delta = min(dm, max(-dm, bf / rf * v))
    tvc = min(tm, max(-tm, bt / rt * v)) if tm > 0.0 else 0.0
    return delta, tvc


@njit(cache=True)
def _forces(S, beta, P, I):
    th, xd, yd = S[2], S[3], S[4]
    V = math.sqrt(xd * xd + yd * yd)
    Om = math.atan2(xd, -yd)
    r_a = r_cm(S[6], P)
    delta, tvc = control(S, beta, P, I, r_a)
    gam = th + delta
    Fx, Fy, tau, q, theta = aero_lab(th, Om, gam, max(S[1], 0.0), V, r_a, P)   # sea-level air below y = 0
    return Fx, Fy, tau, q, theta, gam, r_a, V, tvc


@njit(cache=True)
def rhs(S, beta, P, dS):
    """Fills dS in place. Returns status code."""
    y = S[1]
    if y < -1e5 or y > 1e7:                      # no ground: shooting may overshoot y = 0
        return ERR_STATE
    for i in range(7):
        if not np.isfinite(S[i]):
            return ERR_STATE
    I, ok = inertia(S[6], P)
    if not ok:
        return ERR_TANK
    Fx, Fy, tau, q, theta, gam, r_a, V, tvc = _forces(S, beta, P, I)
    if q > P[Q_TUMBLE] and P[TUMBLE] < abs(theta) < math.pi - P[TUMBLE]:
        return ERR_TUMBLE
    mass = S[6] + P[MDRY]
    Ft = P[FMAX] * beta
    th = S[2]
    dS[0] = S[3]
    dS[1] = S[4]
    dS[2] = S[5]
    dS[3] = (Ft * math.sin(th + tvc) + Fx) / mass
    dS[4] = (Ft * math.cos(th + tvc) + Fy) / mass - gravity(y, P)
    dS[5] = rcs(S, P) if attitude_hold(S, P) else (-tau + r_a * Ft * math.sin(tvc)) / I
    dS[6] = -Ft / P[VEX]
    return OK


@njit(cache=True)
def diagnostics(S, beta, P, out):
    """out = [q, q|sin alpha_b|, a/g0, gamma, theta, tvc]."""
    I, ok = inertia(S[6], P)
    if not ok:
        I = 1.0
    Fx, Fy, tau, q, theta, gam, r_a, V, tvc = _forces(S, beta, P, I)
    th = S[2]
    sa = 0.0
    if V > 1e-9:                                   # |a x v_hat|, a = body axis
        sa = abs(math.sin(th) * (S[4] / V) - math.cos(th) * (S[3] / V))
    mass = S[6] + P[MDRY]
    Ft = P[FMAX] * beta
    ax = (Ft * math.sin(th + tvc) + Fx) / mass
    ay = (Ft * math.cos(th + tvc) + Fy) / mass
    out[0] = q
    out[1] = q * sa
    out[2] = math.sqrt(ax * ax + ay * ay) / P[G0]
    out[3] = gam
    out[4] = theta
    out[5] = tvc
