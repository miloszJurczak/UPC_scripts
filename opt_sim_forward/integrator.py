"""FORWARD-in-time RK45 (Fehlberg 4(5)) integration of the planar flight (njit).

Time runs forward from separation (t = 0), one segment per phase (fins: state feedback,
dynamics.control):
    [0, T1]         free flight   (beta = 0)
    [T1, T2]        entry burn    (beta = 1)
    [T2, T3]        coast         (beta = 0)
    [T3, T4]        landing burn  (beta = 1);  T4 = touchdown
    T1 = t_free, T2 = T1 + t_entry, T3 = T2 + t_coast, T4 = T3 + t_burn.
(simulate.py flips the buffer to the old backward layout: row 0 = touchdown, t <= 0.)
buf[n] = [t, x, y, theta, xdot, ydot, thetadot, m_fuel, q, beta, a/g0, q sin(alpha), gamma, theta_flow, tvc]
stats  = [accepted steps, rejected steps, min |h|, max |h|].
"""
import math

import numpy as np

from dynamics import diagnostics, rhs
from jit import njit
from params import (BURN_STEP, COAST_STEP, EPS, ERR_BUF, ERR_NO_SEP, ERR_SEP_EARLY,
                    ERR_STEP, ERR_UNSTABLE, H_SEP, OK, T_LIMIT)

NCOL = 15
B = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
              [2 / 9, 0.0, 0.0, 0.0, 0.0],
              [1 / 12, 1 / 4, 0.0, 0.0, 0.0],
              [69 / 128, -243 / 128, 135 / 64, 0.0, 0.0],
              [-17 / 12, 27 / 4, -27 / 5, 16 / 15, 0.0],
              [65 / 432, -5 / 16, 13 / 16, 4 / 27, 5 / 144]])
C4 = np.array([1 / 9, 0.0, 9 / 20, 16 / 45, 1 / 12, 0.0])
C5 = np.array([47 / 450, 0.0, 12 / 25, 32 / 225, 1 / 30, 6 / 25])


@njit(cache=True)
def _stages(S, h, beta, P, K, tmp):
    for s in range(6):
        for j in range(7):
            acc = S[j]
            for m in range(s):
                acc += B[s, m] * K[m, j]
            tmp[j] = acc
        st = rhs(tmp, beta, P, K[s])
        if st != OK:
            return st
        for j in range(7):
            K[s, j] *= h
    return OK


@njit(cache=True)
def _combine(S, K, out):
    for j in range(7):
        acc = S[j]
        for i in range(6):
            acc += C5[i] * K[i, j]
        out[j] = acc


@njit(cache=True)
def step_fixed(S, h, beta, P, K, tmp, out):
    st = _stages(S, h, beta, P, K, tmp)
    if st == OK:
        _combine(S, K, out)
    return st


@njit(cache=True)
def step_adaptive(S, h, beta, P, prec, max_step, K, tmp, out, stats):
    """One accepted RK45 step. Returns (status, h_next, h_used). Sign of h kept."""
    direction = np.sign(h) if h != 0.0 else -1.0
    while True:
        if abs(h) < 1e-9:
            return ERR_STEP, 0.0, 0.0
        st = _stages(S, h, beta, P, K, tmp)
        if st != OK:
            return st, 0.0, 0.0
        TE = 0.0
        for j in range(7):
            e = 0.0
            for i in range(6):
                e += (C5[i] - C4[i]) * K[i, j]
            TE += (e / prec[j]) ** 2
        TE = math.sqrt(TE)
        if not np.isfinite(TE):
            return ERR_UNSTABLE, 0.0, 0.0
        hnew = 0.9 * h * (P[EPS] / TE) ** 0.2 if TE != 0.0 else direction * max_step
        if abs(hnew) > max_step:
            hnew = direction * max_step
        if TE <= P[EPS]:
            _combine(S, K, out)
            stats[0] += 1
            stats[2] = min(stats[2], abs(h))
            stats[3] = max(stats[3], abs(h))
            return OK, hnew, h
        stats[1] += 1
        h = hnew


@njit(cache=True)
def _events(t, S, ev, y_prev, P):
    """ev 1: (backward) apogee passed - the booster was still climbing.
    ev 2: (backward) y crossed H_sep from above while climbing = separation.
    (sim1 only tested y < H_sep, so a trajectory whose apogee is BELOW H_sep was
    'separated' at its apogee altitude.)"""
    if ev == 0 and t < -10.0 and S[4] > 0.0:
        ev = 1
    if ev == 1 and y_prev >= P[H_SEP] and S[1] < P[H_SEP]:
        ev = 2
    return ev


@njit(cache=True)
def _record(buf, n, t, S, beta, P, dg):
    if n < 0 or n >= buf.shape[0]:
        return -1
    diagnostics(S, beta, P, dg)
    buf[n, 0] = t
    buf[n, 1:8] = S
    buf[n, 8] = dg[0]
    buf[n, 9] = beta
    buf[n, 10] = dg[2]
    buf[n, 11] = dg[1]
    buf[n, 12] = dg[3]
    buf[n, 13] = dg[4]
    buf[n, 14] = dg[5]
    return n + 1


@njit(cache=True)
def _segment(S, Ta, Tb, h, beta, max_step, ev, P, prec, buf, n, W, stats):
    """Integrate S in place from Ta to Tb. Returns (status, ev, n, h_next)."""
    K, tmp, out = W[:6], W[6], W[7]
    dg = np.zeros(7)
    direction = np.sign(Tb - Ta)
    if direction == 0.0:
        return OK, ev, n, h
    t = Ta
    h = direction * min(abs(h), max_step)
    while (Tb - (t + h)) * direction > 0.0:
        y_prev = S[1]
        st, h, h_used = step_adaptive(S, h, beta, P, prec, max_step, K, tmp, out, stats)
        if st != OK:
            return st, ev, n, h
        S[:] = out
        t += h_used
        n = _record(buf, n, t, S, beta, P, dg)
        if n < 0:
            return ERR_BUF, ev, 0, h
        ev = _events(t, S, ev, y_prev, P)
    h_last = Tb - t
    if h_last != 0.0:
        y_prev = S[1]
        st = step_fixed(S, h_last, beta, P, K, tmp, out)
        if st != OK:
            return st, ev, n, h
        S[:] = out
        n = _record(buf, n, Tb, S, beta, P, dg)
        if n < 0:
            return ERR_BUF, ev, 0, h
        ev = _events(Tb, S, ev, y_prev, P)
    return OK, ev, n, h


@njit(cache=True)
def run_sim(S0, t_free, t_entry, t_coast, t_burn, P, prec, buf, stats):
    """Full FORWARD simulation from separation (t = 0, state S0) to the end of the landing
    burn (= touchdown, t = T4). Returns (status, n_nodes). Touchdown state = buf[n-1].
    The ground is not an event: the end state is driven to y = 0, ydot = 0 by the optimiser.
    The fins are not inputs: dynamics.control sets them from the state at every RHS call."""
    W = np.zeros((9, 7))
    dg = np.zeros(7)
    S = S0.copy()
    stats[0], stats[1], stats[2], stats[3] = 0.0, 0.0, 1e300, 0.0
    T1 = t_free
    T2 = T1 + t_entry
    T3 = T2 + t_coast
    T4 = T3 + t_burn
    if T4 > P[T_LIMIT]:
        return ERR_NO_SEP, 0
    n = _record(buf, 0, 0.0, S, 0.0, P, dg)
    ev = 0
    h = 1e-4
    # (t_start, t_end, beta, max step): free flight, entry burn, coast, landing burn
    phases = ((0.0, T1, 0.0, P[COAST_STEP]), (T1, T2, 1.0, P[BURN_STEP]),
              (T2, T3, 0.0, P[COAST_STEP]), (T3, T4, 1.0, P[BURN_STEP]))
    for ph in phases:
        ta, tb, beta, mstep = ph
        st, ev, n, h = _segment(S, ta, tb, h, beta, mstep, ev, P, prec, buf, n, W, stats)
        if st != OK:
            return st, n
    return OK, n
