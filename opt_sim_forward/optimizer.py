"""Minimisation of the error function E (simulate.py), FORWARD shooting.

The flight starts at the fixed separation state (attitude: Params.theta_sep, default tangent to
the flow at the descending H_sep) and must land: y = 0 and ydot = 0 at the end of the landing
burn. Those two conditions form a very narrow valley, so they are solved INSIDE every
evaluation: least squares on (t_coast, t_burn) -> (y_td, ydot_td).
The fins are NOT optimised: they are a state-feedback LQR loop (dynamics.control).
Outer variables (Nelder-Mead):
    t_delay  time from the DESCENDING crossing of H_sep to entry-burn ignition [s],
             0 <= t_delay <= t_ground - t_cross (no-burn flight would hit the ground)
             -> t_free = t_cross + t_delay (separation -> ignition), see Simulator._descent_profile
    t_entry  entry-burn duration [s]
so the whole optimisation is over the entry-burn and landing-burn timing, and the entry burn can
only start once the booster is falling back below H_sep.
Multi-start: the cost landscape has several local minima (e.g. "short burn high" vs "long burn
low"), so Nelder-Mead (a local method) is run once from every Params.starts point
(entry-ignition altitude on the way down [km], t_entry [s]) - "start 1, start 2, ..." in the
printout - and the best result is kept.
History rows: (stage = 1, start, E, t_free, t_entry, t_coast, t_burn, t_delay).
"""
import math
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from simulate import FAIL_COST, t_free_from_altitude


def landing_seed(sim, tf, te):
    """(t_coast, t_burn) guess: coast without a landing burn, then ignite where the
    constant-thrust stopping distance v^2 / (2 (F/m - g)) equals the altitude."""
    prm = sim.prm
    r = sim.run(tf, te, prm.inner_bounds[0][1], 0.0)
    if not r.ok:
        return np.array([40.0, 7.0])
    f = r.traj[::-1]                                      # forward in time
    t = f[:, 0] - f[0, 0]
    T2 = tf + te
    m = f[:, 7] + prm.mdry
    acc = prm.Fmax / m - prm.g0
    d = np.where(f[:, 5] < 0, f[:, 5] ** 2 / (2 * np.maximum(acc, 1e-3)), 0.0)
    ok = (t >= T2) & (f[:, 2] <= d) & (f[:, 5] < 0)
    if not ok.any():
        return np.array([prm.inner_bounds[0][1], 7.0])
    i = int(np.argmax(ok))
    return np.array([max(t[i] - T2, 0.0), -f[i, 5] / acc[i]])


def solve_landing(sim, tf, te, guess=None):
    """Inner solve: (t_coast, t_burn) so that the touchdown state has y = 0, ydot = 0.
    Returns (t_coast, t_burn, SimResult)."""
    prm = sim.prm
    (lo0, hi0), (lo1, hi1) = prm.inner_bounds
    lo, hi = [lo0, lo1], [hi0, hi1]
    s = prm.psi
    big = 1e6

    def resid(u):
        r = sim.run(tf, te, u[0], u[1])
        if not r.ok:
            return np.array([big, big])
        return np.array([r.S_td[1] / s["y"], r.S_td[4] / s["ydot"]])

    def ls(u0, ds):
        u0 = np.clip(np.asarray(u0, float), np.add(lo, 1e-9), np.subtract(hi, 1e-9))
        return least_squares(resid, u0, bounds=(lo, hi), method="trf", diff_step=ds,
                             xtol=1e-12, ftol=1e-12, max_nfev=60)

    # diff_step 1e-5 (relative): above the integration noise of the saturating fin loop
    sol = ls(landing_seed(sim, tf, te) if guess is None else guess, 1e-5)
    for u0, ds in (([] if guess is None else [(landing_seed(sim, tf, te), 1e-5)]) + [(sol.x, 1e-4)]):
        if np.max(np.abs(sol.fun)) <= 1.0:
            break
        sol2 = ls(u0, ds)
        if sol2.cost < sol.cost:
            sol = sol2
    return sol.x[0], sol.x[1], sim.run(tf, te, sol.x[0], sol.x[1])


def _bounds(sim):
    return [(0.0, max(sim.t_ground - sim.t_cross, 1.0)), tuple(sim.prm.bounds[1])]


def _run_start(sim, z0, sid, history, maxiter):
    bounds = _bounds(sim)
    last = [None]

    def obj(z):
        tf = sim.t_cross + z[0]
        tc, tb, res = solve_landing(sim, tf, z[1], last[0])
        c = sim.cost_terms(res) if res.ok else {"total": FAIL_COST}
        if res.ok:
            last[0] = np.array([tc, tb])
        history.append((1, sid, c["total"], tf, z[1], tc, tb, z[0]))
        return c["total"]

    z0 = np.clip(np.asarray(z0, float), [b[0] for b in bounds], [b[1] for b in bounds])
    minimize(obj, z0, method="Nelder-Mead", bounds=bounds,
             options={"xatol": sim.prm.nm_xatol, "fatol": sim.prm.nm_fatol, "maxiter": 10 * maxiter, "adaptive": True})
    return min((r for r in history if r[1] == sid), key=lambda r: r[2])


def optimize(sim, verbose=True, extra_starts=()):
    """extra_starts: additional points (t_delay, t_entry), e.g. the optimum of a neighbouring sweep point."""
    prm = sim.prm
    t0, n0 = time.perf_counter(), sim.n_sims
    history = []
    z0s = [[sim.t_at_altitude(1e3 * h_km) - sim.t_cross, te0] for h_km, te0 in prm.starts] + \
          [list(z)[:2] for z in extra_starts]
    if verbose:
        print(f"  falls back through H_sep at t = {sim.t_cross:.1f} s after separation; "
              f"no-burn ground impact at t = {sim.t_ground:.1f} s")
    best = None
    for sid, z0 in enumerate(z0s):
        b = _run_start(sim, z0, sid, history, prm.maxiter)
        if verbose:
            print(f"  start {sid + 1} (ignite at {sim.altitude_at(sim.t_cross + z0[0]) / 1e3:.0f} km, t_entry {z0[1]:.1f} s)"
                  f" -> E = {b[2]:.5g}: ignition {b[7]:.2f} s after H_sep ({sim.altitude_at(b[3]) / 1e3:.1f} km), "
                  f"t_entry {b[4]:.2f}, t_coast {b[5]:.2f}, t_burn {b[6]:.3f}")
        if best is None or b[2] < best[2]:
            best = b
    if best is None or best[2] >= FAIL_COST:
        raise RuntimeError("no landing trajectory found from any start - check starts/bounds/settings")
    res = sim.run(best[3], best[4], best[5], best[6], thermal=True)
    sim.cost_terms(res)
    return dict(best=best, res=res, history=history, wall_time=time.perf_counter() - t0,
                n_sims=sim.n_sims - n0)
