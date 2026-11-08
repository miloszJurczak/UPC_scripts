"""Propellant vs separation-velocity angle: one full optimisation (entry + landing burn timing) per angle.

Every point: attitude at separation per Params.theta_sep (default prograde), RCS flip after apex to be
exactly tangent to the flow (engine first) at the descending H_sep, LQR fins below (dynamics.control). Points are independent, so they can
run in parallel processes (workers > 1); with workers = 1 each point also starts from the
previous point's optimum.
"""
import copy
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from optimizer import optimize
from simulate import Simulator


def sweep_point(prm0, a, extra=()):
    """One angle [deg from vertical] -> result row (dict)."""
    np.seterr(all="ignore")
    prm = copy.deepcopy(prm0)
    prm.sep_angle = math.radians(a)
    t0 = time.perf_counter()
    row = dict(angle_from_vertical=float(a), elevation=90.0 - float(a), mode="tangent")  # tangent at descending H_sep (RCS)
    try:
        sim = Simulator(prm)
        out = optimize(sim, verbose=False, extra_starts=extra)
        res, b = out["res"], out["best"]
        ok_land = abs(res.S_td[1]) < prm.psi["y"] and abs(res.S_td[4]) < prm.psi["ydot"]
        td = (math.degrees(abs(res.touchdown[0])), abs(res.touchdown[1]), math.degrees(abs(res.touchdown[2])))
        ok_td = all(v <= bd for v, bd in zip(td, prm.td_bounds))
        lim = lambda v, L: L <= 0 or v <= L  # noqa: E731
        feasible = bool(ok_land and ok_td and lim(res.qa_max, prm.qa_limit) and lim(res.q_max, prm.q_limit)
                        and lim(res.a_max, prm.g_limit))
        row.update(m_fuel=float(res.S_sep[6]), E=float(b[2]), feasible=feasible,
                   theta_sep_deg=math.degrees(res.theta_sep), theta_flow_Hsep_deg=math.degrees(res.theta_flow_Hsep),
                   t_free=float(b[3]), t_delay=float(b[7]), h_ignition_km=sim.altitude_at(b[3]) / 1e3,
                   t_entry=float(b[4]), t_coast=float(b[5]), t_burn=float(b[6]),
                   qa_max=float(res.qa_max), tilt_td_deg=math.degrees(res.touchdown[0]),
                   xdot_td=float(res.touchdown[1]), thdot_td_degps=math.degrees(res.touchdown[2]),
                   x_td_km=float(res.S_td[0] / 1e3))
    except RuntimeError as e:
        row.update(m_fuel=np.nan, E=np.nan, feasible=False, error=str(e))
    row["wall_s"] = time.perf_counter() - t0
    return row


def _print(row):
    print(f"  {row['angle_from_vertical']:5.1f} deg from vertical: m_fuel = {row['m_fuel'] / 1e3:7.2f} t, "
          f"theta_sep = {row.get('theta_sep_deg', np.nan):+7.2f} deg, tilt_td = {row.get('tilt_td_deg', np.nan):+6.2f} deg, "
          f"E = {row['E']:.4g}, {'ok' if row['feasible'] else 'INFEASIBLE'}  ({row['wall_s']:.0f} s)", flush=True)


def _serial(prm0, angles, verbose):
    rows, prev = [], None
    for a in angles:
        rows.append(sweep_point(prm0, a, [prev] if prev else []))
        r = rows[-1]
        if np.isfinite(r["m_fuel"]):
            prev = (r["t_delay"], r["t_entry"])
        if verbose:
            _print(r)
    return rows


def _parallel(prm0, angles, workers, verbose):
    """Process pool. Must run in a process whose main module has an  if __name__ == "__main__"  guard
    (Windows starts workers by re-importing the main module) - see sweep_propellant."""
    from concurrent.futures.process import BrokenProcessPool
    Simulator(copy.deepcopy(prm0))                  # compile + write the numba cache once, before the workers
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(sweep_point, prm0, a) for a in angles]
            for f in futs:
                rows.append(f.result())
                if verbose:
                    _print(rows[-1])
    except BrokenProcessPool as e:
        print(f"  process pool failed ({e}); finishing the remaining angles serially", flush=True)
        rows += _serial(prm0, angles[len(rows):], verbose)
    return rows


def sweep_propellant(prm0, angles_from_vertical_deg, workers=1, verbose=True):
    """One optimisation per angle [deg from vertical]. workers > 1: run in parallel.
    The parallel sweep is launched as a SEPARATE python process (python sweep.py job.pkl): in Jupyter /
    VS Code interactive on Windows, worker processes re-import the calling script (run_interactive.py,
    no __main__ guard), which kills them ("BrokenProcessPool"). sweep.py has the guard, so it is safe."""
    angles = [float(a) for a in angles_from_vertical_deg]
    if workers <= 1:
        return _serial(prm0, angles, verbose)
    import pickle
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        job, out = os.path.join(tmp, "job.pkl"), os.path.join(tmp, "rows.pkl")
        with open(job, "wb") as fh:
            pickle.dump((prm0, angles, workers), fh)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), job, out]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        log = []
        for line in p.stdout:                       # stream progress into the notebook
            log.append(line)
            if verbose:
                print(line, end="", flush=True)
        p.wait()
        if p.returncode == 0 and os.path.isfile(out):
            with open(out, "rb") as fh:
                return pickle.load(fh)
    print(f"  parallel sweep process failed (exit code {p.returncode}); running serially instead", flush=True)
    if not verbose:
        print("".join(log[-20:]))
    return _serial(prm0, angles, verbose)


def save_sweep(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(path, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def new_sweep_dir(runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    nums = [int(d[6:]) for d in os.listdir(runs_dir) if d.startswith("sweep_") and d[6:].isdigit()]
    p = os.path.join(runs_dir, f"sweep_{max(nums, default=0) + 1:04d}")
    os.makedirs(p)
    return p


if __name__ == "__main__":                          # parallel worker host: python sweep.py job.pkl rows.pkl
    import multiprocessing
    import pickle
    import sys
    multiprocessing.set_start_method("spawn", force=True)   # same on every OS (Windows only has spawn)
    with open(sys.argv[1], "rb") as fh:
        _prm0, _angles, _workers = pickle.load(fh)
    _rows = _parallel(_prm0, _angles, _workers, verbose=True)
    with open(sys.argv[2], "wb") as fh:
        pickle.dump(_rows, fh)
