"""Python-side wrapper: one FORWARD simulation and the error function.

The flight starts at the separation state (fixed by the mission) and must land:
    S(0) = [x, y, theta, xdot, ydot, thetadot, m_fuel]
         = [0, H_sep, theta_sep, v0 sin a, v0 cos a, thetadot_sep, m_load]
    m_load = F_max (t_entry + t_burn) / v_ex      (all burns at full throttle -> m_fuel(touchdown) = 0,
                                                   exactly as the backward model's touchdown state)
Timing variables (seconds, all >= 0), forward in time:
    t_free  separation -> entry-burn ignition,  t_entry  entry burn,
    t_coast coast between burns,                t_burn   landing burn (ends at touchdown)
    Before touchdown (unchanged convention, used by the plots):
    T_burn = t_burn,  T2 = T_burn + t_coast,  T1 = T2 + t_entry
Error function, Psi precisions s_* from Params.psi, all evaluated AT TOUCHDOWN:
    E = (y_td/s_y)^2 + (ydot_td/s_ydot)^2                          must land (solved inside, optimizer.py)
      + (m_load/s_m)^2                                             fuel (the actual goal)
      + (tilt_td/s_tilt)^2 + (xdot_td/s_xtd)^2 + (thetadot_td/s_thtd)^2   touchdown not ideal
      + (min(0, min_t y)/s_y)^2                                    went below ground before touchdown
      + w_td sum [|tilt|/T, |xdot|/X, |thetadot|/W - 1]_+^2         outside TD_BOUNDS (hard-ish)
      + w_qa [max_t q|sin alpha| / qa_limit - 1]_+^2               [qa_limit > 0]
      + w_q  [max_t q / q_limit - 1]_+^2                           [q_limit > 0]
      + w_a  [max_t a / g_limit - 1]_+^2                           [g_limit > 0]
      + w_T  sum_sites [(T_max - T_melt + margin) / 100 K]_+^2     melting
The stored trajectory keeps the old layout: row 0 = touchdown (t = 0), last row = separation (t < 0).
"""
import math
import time

import numpy as np

from integrator import NCOL, run_sim
from params import OK, RCS_TARGET, STATUS_TEXT
from thermal import melt_excess, pack_thermal, site_temperatures

FAIL_COST = 1e30          # must exceed every valid cost


def deltas_to_times(t_burn, t_coast, t_entry):
    T_burn = t_burn
    T2 = T_burn + t_coast
    return T_burn, T2 + t_entry, T2


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def t_free_from_altitude(prm, h_ign):
    """Vacuum-ballistic time from separation to the DESCENDING crossing of altitude h_ign [m]
    (used only to seed t_free)."""
    vy = prm.v0 * math.cos(prm.sep_angle)
    return (vy + math.sqrt(vy * vy + 2.0 * prm.g0 * max(prm.H_sep - h_ign, 0.0))) / prm.g0


def _descending_crossing(traj, H):
    """Index (in the stored, touchdown-first layout) of the first node below H after falling
    through it, or None."""
    f = traj[::-1]
    down = np.nonzero((f[1:, 2] < H) & (f[:-1, 2] >= H) & (f[1:, 5] < 0))[0]
    return None if len(down) == 0 else len(f) - 1 - (down[0] + 1)


def crossing_state(traj, H):
    """State row interpolated linearly in y to the exact DESCENDING crossing y = H (None if none).
    Uses the last node above and the first node below H."""
    i = _descending_crossing(traj, H)
    if i is None or i + 1 >= len(traj):
        return None
    lo, hi = traj[i], traj[i + 1]                  # stored layout: i = first node below, i+1 = last above
    w = (hi[2] - H) / (hi[2] - lo[2])
    return hi + w * (lo - hi)


def rcs_target(sim, iters=6):
    """RCS target attitude: tangent to the flow, ENGINE FIRST (body vs flow theta = theta_lab + Omega = 0,
    Omega = atan2(xdot, -ydot)), at the moment the booster falls back through y = H_sep (ydot < 0).
    Fixed point: fly the no-burn reference with the RCS slewing to the current target, read Omega at
    the crossing, set target = -Omega (the attitude history changes the drag above H_sep only slightly).
    In vacuum it is exactly -sep_angle."""
    prm = sim.prm
    T = t_free_from_altitude(prm, prm.H_sep) + 5.0
    th = -prm.sep_angle
    for _ in range(iters):
        sim.P[RCS_TARGET] = th
        r = sim.run(T, 0.0, 0.0, 0.0, theta_sep=sim.theta_sep0(th))
        c = crossing_state(r.traj, prm.H_sep)
        if c is None:
            break
        th = -math.atan2(c[4], -c[5])
    sim.P[RCS_TARGET] = th
    return th


class SimResult:
    def __init__(self, status, n, buf, stats, timings, t_free, theta_sep, wall, H_sep):
        self.status, self.message = status, STATUS_TEXT[status]
        self.ok = status == OK
        self.timings = timings                     # (t_burn, t_coast, t_entry)
        self.t_free, self.theta_sep = t_free, theta_sep
        fwd = buf[:n]
        self.traj = fwd[::-1].copy()               # row 0 = last integrated node (touchdown)
        if n:
            self.traj[:, 0] -= fwd[-1, 0]          # time to touchdown, <= 0
        self.touchdown = self.traj[0, [3, 4, 6]].copy() if n else np.zeros(3)   # theta, xdot, thetadot
        self.touchdown[0] = wrap(self.touchdown[0])
        self.stats = stats.copy()
        self.wall_time = wall
        good = self.ok and n > 0
        self.S_sep = self.traj[-1, 1:8] if good else None
        self.S_td = self.traj[0, 1:8] if good else None
        self.t_sep = self.traj[-1, 0] if good else np.nan
        col = (lambda k: float(self.traj[:, k].max())) if n else (lambda k: np.nan)
        self.q_max, self.a_max, self.qa_max = col(8), col(10), col(11)
        self.y_min = float(self.traj[1:, 2].min()) if n > 1 else np.nan   # before touchdown
        c = crossing_state(self.traj, H_sep) if n > 1 else None
        self.theta_flow_Hsep = float(wrap(c[3] + math.atan2(c[4], -c[5]))) if c is not None else np.nan  # body vs flow
        self.thdot_Hsep = float(c[6]) if c is not None else np.nan                                      # pitch rate
        self.delta = wrap(self.traj[:, 12] - self.traj[:, 3]) if n else np.zeros(0)  # fin deflection history
        self.sites = None
        self.cost = None


class Simulator:
    def __init__(self, prm):
        self.set_params(prm)

    def set_params(self, prm):
        self.prm = prm
        self.P, self.prec = prm.pack()
        self.TP = pack_thermal(prm)
        self.buf = np.zeros((prm.n_buffer, NCOL))
        self.stats = np.zeros(4)
        self.n_sims = 0
        self.sim_time = 0.0
        self.theta_sep = self.theta_sep0(-prm.sep_angle)
        self.theta_target = rcs_target(self)
        self.theta_sep = self.theta_sep0(self.theta_target)
        self._descent_profile()

    def _descent_profile(self):
        """No-burn reference flight from separation to the ground. Until the descending H_sep
        crossing nothing depends on the burn timing (attitude held, no burn allowed), so
        t_cross = time of that crossing is exact; t_ground (where the no-burn flight would hit
        y = 0) bounds the entry ignition from above. desc_t, desc_y: the descent below H_sep."""
        prm = self.prm
        r = self.run(t_free_from_altitude(prm, 0.0) + 300.0, 0.0, 0.0, 0.0)
        f = r.traj[::-1]
        t, y = f[:, 0] - f[0, 0], f[:, 2]
        i = _descending_crossing(r.traj, prm.H_sep)
        if i is None or len(f) < 3:                      # fallback: vacuum
            self.t_cross = t_free_from_altitude(prm, prm.H_sep)
            self.t_ground = t_free_from_altitude(prm, 0.0)
            self.desc_t = np.array([self.t_cross, self.t_ground])
            self.desc_y = np.array([prm.H_sep, 0.0])
            return
        k = len(f) - 1 - i                                # forward index of the first node below H_sep
        tc = t[k - 1] + (t[k] - t[k - 1]) * (y[k - 1] - prm.H_sep) / (y[k - 1] - y[k])
        g = np.nonzero(y[k:] <= 0.0)[0]
        kg = k + (g[0] if len(g) else len(y[k:]) - 1)
        tg = t[kg - 1] + (t[kg] - t[kg - 1]) * y[kg - 1] / (y[kg - 1] - y[kg]) if len(g) else t[kg]
        self.t_cross, self.t_ground = float(tc), float(tg)
        self.desc_t = np.r_[tc, t[k:kg], tg]
        self.desc_y = np.r_[prm.H_sep, y[k:kg], 0.0]

    def t_at_altitude(self, h):
        """Time after separation at which the no-burn flight falls through altitude h [m]."""
        return float(np.interp(h, self.desc_y[::-1], self.desc_t[::-1]))

    def altitude_at(self, t_free):
        """Altitude [m] of the no-burn flight at time t_free after separation (= entry ignition altitude)."""
        return float(np.interp(t_free, self.desc_t, self.desc_y))

    def theta_sep0(self, target):
        """Attitude at separation per Params.theta_sep ("prograde" / "tangent" = target / number)."""
        ts = self.prm.theta_sep
        if ts == "prograde":
            return self.prm.sep_angle                   # body axis (sin th, cos th) along the ascent velocity
        if ts == "tangent" or ts is None:
            return target
        return float(ts)

    def theta_default(self):
        """Attitude at separation used by run() [rad]."""
        return self.theta_sep

    def initial_state(self, theta_sep, t_entry, t_burn):
        prm = self.prm
        m_load = prm.Fmax * (t_entry + t_burn) / prm.v_exhaust
        return np.array([0.0, prm.H_sep, theta_sep, prm.v0 * math.sin(prm.sep_angle),
                         prm.v0 * math.cos(prm.sep_angle), prm.thetadot_sep, m_load])

    # ------------------------------------------------------------------ one sim
    def run(self, t_free, t_entry, t_coast, t_burn, theta_sep=None, thermal=False):
        th = self.theta_default() if theta_sep is None else float(theta_sep)
        S0 = self.initial_state(th, t_entry, t_burn)
        t0 = time.perf_counter()
        st, n = run_sim(S0, float(t_free), float(t_entry), float(t_coast), float(t_burn),
                        self.P, self.prec, self.buf, self.stats)
        dt = time.perf_counter() - t0
        self.n_sims += 1
        self.sim_time += dt
        res = SimResult(int(st), int(n), self.buf, self.stats, (t_burn, t_coast, t_entry),
                        t_free, th, dt, self.prm.H_sep)
        if thermal and res.ok:
            res.sites = site_temperatures(res.traj, self.prm, self.TP)
        return res

    # ------------------------------------------------------- error function
    def cost_terms(self, res):
        prm, s = self.prm, self.prm.psi
        if res is None or not res.ok:
            return dict(total=FAIL_COST)
        if res.sites is None:
            res.sites = site_temperatures(res.traj, prm, self.TP)
        S = res.S_td
        c = dict(y=(S[1] / s["y"]) ** 2, ydot=(S[4] / s["ydot"]) ** 2,
                 fuel=(res.S_sep[6] / s["m_fuel"]) ** 2,
                 touchdown=(res.touchdown[0] / s["tilt_td"]) ** 2 + (res.touchdown[1] / s["xdot_td"]) ** 2
                 + (res.touchdown[2] / s["thdot_td"]) ** 2,
                 ground=(min(0.0, res.y_min) / s["y"]) ** 2,
                 td_tol=prm.w_td * sum(max(0.0, abs(v) / (b * (1 - prm.limit_margin)) - 1) ** 2 for v, b in
                                       zip((math.degrees(res.touchdown[0]), res.touchdown[1],
                                            math.degrees(res.touchdown[2])), prm.td_bounds)),
                 qalpha=prm.w_qa * max(0.0, res.qa_max / (prm.qa_limit * (1 - prm.limit_margin)) - 1) ** 2 if prm.qa_limit > 0 else 0.0,
                 q=prm.w_q * max(0.0, res.q_max / (prm.q_limit * (1 - prm.limit_margin)) - 1) ** 2 if prm.q_limit > 0 else 0.0,
                 accel=prm.w_accel * max(0.0, res.a_max / (prm.g_limit * (1 - prm.limit_margin)) - 1) ** 2 if prm.g_limit > 0 else 0.0,
                 thermal=prm.w_thermal * melt_excess(res.sites) / 100.0 ** 2)
        c["total"] = float(sum(c.values()))
        res.cost = c
        return c
