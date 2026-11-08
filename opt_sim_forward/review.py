"""Run bookkeeping and the human-readable review of a run.

Every run gets a number and a folder  Opt_sim/runs/run_NNNN/  with
    review.txt, summary.json, trajectory.npz (columns below), fig_*.pdf/png, animation.gif
"""
import datetime as dt
import json
import os

import numpy as np

from jit import HAVE_NUMBA
from simulate import deltas_to_times

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
COLUMNS = "t x y theta xdot ydot thetadot m_fuel q beta a_g0 q_sin_alpha gamma theta_flow tvc"


def new_run_dir():
    os.makedirs(RUNS_DIR, exist_ok=True)
    nums = [int(d[4:]) for d in os.listdir(RUNS_DIR) if d.startswith("run_") and d[4:].isdigit()]
    n = max(nums, default=0) + 1
    path = os.path.join(RUNS_DIR, f"run_{n:04d}")
    os.makedirs(path)
    return n, path


def _lim(val, lim, unit, fmt="{:.2f}"):
    if lim <= 0:
        return f"{fmt.format(val)} {unit}   (no limit)"
    return f"{fmt.format(val)} {unit}   limit {fmt.format(lim)} {unit}   [{'PASS' if val <= lim else 'FAIL'}]"


def build_review(sim, res, run_no, mode, extra=None):
    """Return (text, summary dict) for one SimResult."""
    prm = sim.prm
    extra = extra or {}
    s = {"run": run_no, "mode": mode, "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
    bar = "=" * 76
    L = [bar, f" RUN #{run_no:04d}   mode: {mode}   {s['timestamp']}", bar]
    tb, tc, te = res.timings
    T_burn, T1, T2 = deltas_to_times(tb, tc, te)
    s["timings"] = dict(t_free=res.t_free, t_entry=te, t_coast=tc, t_burn=tb)
    L.append(f" Timings   t_free = {res.t_free:8.3f} s   t_entry = {te:8.3f} s   t_coast = {tc:8.3f} s   t_burn = {tb:8.3f} s")
    L.append(f"           entry burn {T1:7.2f} -> {T2:7.2f} s before touchdown; landing burn last {T_burn:.2f} s")
    if hasattr(sim, "t_cross"):
        L.append(f"           booster falls back through H_sep {sim.t_cross:.2f} s after separation; entry ignition "
                 f"{res.t_free - sim.t_cross:+.2f} s after that, at y = {sim.altitude_at(res.t_free) / 1e3:.2f} km")
        s["timings"].update(t_cross=sim.t_cross, t_delay=res.t_free - sim.t_cross)
    dmax = float(np.degrees(np.abs(res.delta)).max()) if len(res.traj) else 0.0
    tmax = float(np.degrees(np.abs(res.traj[:, 14])).max()) if len(res.traj) else 0.0
    L.append(f" Control   LQR: e_max {prm.e_max_deg:g} deg, edot_max {prm.edot_max_degps:g} deg/s; "
             f"w_n <= {prm.ctrl_wmax:g} rad/s; fins |delta| <= {prm.fin_bound_deg:g} deg (max used {dmax:.1f})"
             + (f", gimbal |tvc| <= {prm.tvc_max_deg:g} deg (max used {tmax:.1f})" if prm.tvc_max_deg > 0 else ", no gimbal")
             + f"; vertical below {prm.v_vert:g} m/s, retrograde above {prm.v_blend:g} m/s")
    if not res.ok:
        L += [f" STATUS    FAILED: {res.message}", bar]
        s.update(status=res.message, verdict="FAILED")
        return "\n".join(L), s
    S0, S = res.S_sep, res.S_td
    errs = dict(y=S[1], ydot=S[4])
    L.append(f" Flight    {-res.t_sep:.2f} s from separation to touchdown (t_free = {res.t_free:.2f} s); "
             f"landing site x = {S[0] / 1e3:+.3f} km from separation")
    L.append(" -- separation state (initial condition) " + "-" * 35)
    L.append(f"   y = {S0[1]:.0f} m   xdot = {S0[3]:+.2f} m/s   ydot = {S0[4]:+.2f} m/s   "
             f"thetadot = {S0[5]:+.3g} rad/s")
    L.append(f"   theta_lab at separation = {np.degrees(S0[2]):+.2f} deg ({prm.theta_sep}); RCS: hold while climbing, "
             f"flip after apex to {np.degrees(sim.theta_target):+.2f} deg (|alpha| <= {prm.rcs_alpha_degps2:g} deg/s^2, "
             f"|omega| <= {prm.rcs_wmax_degps:g} deg/s)")
    L.append(f"   at the descending H_sep crossing: body vs flow = {np.degrees(res.theta_flow_Hsep):+.4f} deg "
             f"(0 = tangent, engine first), thetadot = {np.degrees(res.thdot_Hsep):+.4f} deg/s")
    L.append(" -- touchdown: landing error / precision " + "-" * 35)
    for k, unit in (("y", "m"), ("ydot", "m/s")):
        L.append(f"   {k:6s} = {errs[k]:+.3e} {unit:6s}  precision {prm.psi[k]:.3g}  -> ({errs[k] / prm.psi[k]:+.2e})^2")
    L.append(f"   lowest altitude before touchdown = {res.y_min:.2f} m")
    td = res.touchdown
    T, X, W = prm.td_bounds
    td_ok = abs(np.degrees(td[0])) <= T and abs(td[1]) <= X and abs(np.degrees(td[2])) <= W
    L.append(" -- touchdown attitude/drift (soft cost vs psi; tolerance = pass/fail) " + "-" * 6)
    L.append(f"   tilt = {np.degrees(td[0]):+.2f} deg (tol {T:g}, psi {np.degrees(prm.psi['tilt_td']):g})   "
             f"xdot = {td[1]:+.3f} m/s (tol {X:g}, psi {prm.psi['xdot_td']:g})   "
             f"thetadot = {np.degrees(td[2]):+.3f} deg/s (tol {W:g}, psi {np.degrees(prm.psi['thdot_td']):g})")
    cap = prm.tank_capacity()
    L.append(" -- propellant " + "-" * 61)
    L.append(f"   m_fuel loaded at separation = {S0[6]:10.1f} kg  ({100 * S0[6] / cap:.1f} % of tank capacity {cap:.0f} kg);"
             f" left at touchdown {S[6]:.2f} kg")
    L.append(" -- loads " + "-" * 66)
    i = int(np.argmax(res.traj[:, 11]))
    L.append(f"   q sin(alpha) max = " + _lim(res.qa_max, prm.qa_limit, "Pa", "{:.0f}")
             + f"   at y = {res.traj[i, 2] / 1e3:.1f} km")
    i = int(np.argmax(res.traj[:, 8]))
    L.append(f"   q max            = " + _lim(res.q_max / 1e3, prm.q_limit / 1e3, "kPa", "{:.1f}")
             + f"   at y = {res.traj[i, 2] / 1e3:.1f} km")
    L.append(f"   a max            = " + _lim(res.a_max, prm.g_limit, "g0"))
    th = np.degrees(np.abs(res.traj[:, 13]))
    eng_first = th < 90
    dense = res.traj[:, 8] > prm.q_tumble
    valid = np.mean(th[eng_first & dense] < 10) if np.any(eng_first & dense) else 1.0
    L.append(f"   |body - flow| < 10 deg (forces_lab validity) on {100 * valid:.0f} % of nodes with q > {prm.q_tumble:g} Pa")
    melt = False
    for name, st in (res.sites or {}).items():
        ok = st["margin"] >= 0
        melt |= (not ok) and st["active"]
        L.append(f"   T_max[{name:6s}] = {st['T_max']:6.0f} K  {st['material']:13s} d={st['delta'] * 1e3:.1f} mm "
                 f"R_n={st['R_n']:g} m  T_melt={st['T_melt']:.0f} K  margin {st['margin']:+.0f} K  [{'OK' if ok else 'MELTS'}]")
    if res.cost:
        L.append(" -- error function E " + "-" * 55)
        L.append("   " + "  ".join(f"{k}={v:.3g}" for k, v in res.cost.items()))
    st = res.stats
    L.append(" -- numerics " + "-" * 63)
    L.append(f"   nodes {len(res.traj)}, steps {int(st[0])} (+{int(st[1])} rejected), |h| in [{st[2]:.1e}, {st[3]:.2f}] s;"
             f" {res.wall_time * 1e3:.2f} ms/sim; numba {'ON' if HAVE_NUMBA else 'OFF'}")
    for k, v in extra.items():
        L.append(f"   {k}: {v}")
    land_ok = abs(errs["y"]) < prm.psi["y"] and abs(errs["ydot"]) < prm.psi["ydot"] and res.y_min > -prm.psi["y"]
    bad = [n for n, b in (("landing", not land_ok), ("touchdown attitude/drift", not td_ok),
                          ("q sin(alpha)", prm.qa_limit > 0 and res.qa_max > prm.qa_limit),
                          ("q", prm.q_limit > 0 and res.q_max > prm.q_limit),
                          ("accel", prm.g_limit > 0 and res.a_max > prm.g_limit), ("melting", melt)) if b]
    verdict = "ALL CONSTRAINTS MET" if not bad else "VIOLATED: " + ", ".join(bad)
    L += [bar, f" VERDICT: {verdict}", bar]
    s.update(status="ok", verdict=verdict, m_fuel=float(S0[6]), theta_sep_deg=float(np.degrees(S0[2])), q_max=res.q_max, qa_max=res.qa_max,
             a_max=res.a_max, errors={k: float(v) for k, v in errs.items()},
             touchdown_deg_mps_degps=[float(np.degrees(td[0])), float(td[1]), float(np.degrees(td[2]))],
             theta_flow_Hsep_deg=float(np.degrees(res.theta_flow_Hsep)), cost={k: float(v) for k, v in (res.cost or {}).items()},
             extra={k: str(v) for k, v in extra.items()})
    return "\n".join(L), s


def save_run(path, text, summary, res=None, params=None):
    with open(os.path.join(path, "review.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    if params is not None:
        summary["params"] = dict(params.editable(), sites=params.sites, psi=params.psi)
    with open(os.path.join(path, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    if res is not None and len(res.traj):
        np.savez_compressed(os.path.join(path, "trajectory.npz"), traj=res.traj,
                            timings=np.array(res.timings), t_free=res.t_free, theta_sep=res.theta_sep,
                            columns=COLUMNS)


# ------------------------------------------------------------- notebook review
def _sci(v, d=2):
    """LaTeX scientific notation: -1.66e-10 -> -1.66 \\times 10^{-10}."""
    if v == 0 or not np.isfinite(v):
        return f"{v:.{d}f}"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 4:
        return f"{v:.{max(d, -e + 1)}f}" if e < 0 else f"{v:.{d}f}"
    return f"{v / 10 ** e:.{d}f} \\times 10^{{{e}}}"


def _join(blocks):
    """Blank line between blocks, none between consecutive table rows."""
    out = blocks[0]
    for prev, b in zip(blocks, blocks[1:]):
        out += ("\n" if prev.startswith("|") and b.startswith("|") else "\n\n") + b
    return out


def _ok(b):
    return "✅ pass" if b else "❌ **fail**"


def review_markdown(sim, res, summary):
    """Markdown + LaTeX version of build_review (for IPython.display.Markdown)."""
    prm = sim.prm
    s = summary
    tb, tc, te = res.timings
    T_burn, T1, T2 = deltas_to_times(tb, tc, te)
    M = [f"## Run \\#{s['run']:04d} &nbsp;·&nbsp; mode `{s['mode']}` &nbsp;·&nbsp; {s['timestamp']}"]
    v = s.get("verdict", "")
    M.append(f"> **Verdict:** {'✅' if v == 'ALL CONSTRAINTS MET' else '❌'} {v}")
    M += ["### Timings",
          "| $t_{\\mathrm{free}}$ | $t_{\\mathrm{entry}}$ | $t_{\\mathrm{coast}}$ | $t_{\\mathrm{burn}}$ | flight |",
          "|---:|---:|---:|---:|---:|",
          f"| ${res.t_free:.3f}$ s | ${te:.3f}$ s | ${tc:.3f}$ s | ${tb:.3f}$ s | "
          f"${-res.t_sep if res.ok else res.t_free + te + tc + tb:.2f}$ s |",
          f"Entry burn from $t = -{T1:.2f}$ s to $-{T2:.2f}$ s; landing burn the last ${T_burn:.2f}$ s "
          f"($t = 0$ is touchdown)."
          + (f" The booster falls back through $H_{{\\mathrm{{sep}}}}$ ${sim.t_cross:.2f}$ s after separation; the entry burn "
             f"ignites $t_{{\\mathrm{{delay}}}} = {res.t_free - sim.t_cross:.2f}$ s later, at $y = {sim.altitude_at(res.t_free) / 1e3:.2f}$ km."
             if hasattr(sim, "t_cross") else "")]
    dmax = float(np.degrees(np.abs(res.delta)).max()) if len(res.traj) else 0.0
    tmax = float(np.degrees(np.abs(res.traj[:, 14])).max()) if len(res.traj) else 0.0
    M += ["### Attitude control: fins only, LQR (gain-scheduled, bandwidth-capped)",
          "$$\\ddot e = a\\,e + b_f\\,\\delta,\\qquad "
          "J = \\int\\Big(\\frac{e^2}{e_{\\max}^2} + \\frac{\\dot e^2}{\\dot e_{\\max}^2} + "
          "\\frac{\\delta^2}{\\delta_{\\max}^2}\\Big)dt,\\qquad \\omega_n\\le\\omega_{\\max},\\ \\zeta\\le1$$",
          "| $e_{\\max}$ | $\\dot e_{\\max}$ | $\\omega_{\\max}$ | fins $\\delta_{\\max}$ (max used) | reference |",
          "|---:|---:|---:|---:|---|",
          f"| ${prm.e_max_deg:g}^\\circ$ | ${prm.edot_max_degps:g}^\\circ$/s | ${prm.ctrl_wmax:g}$ rad/s | "
          f"${prm.fin_bound_deg:g}^\\circ$ (${dmax:.1f}^\\circ$) | retrograde above ${prm.v_blend:g}$ m/s, vertical below ${prm.v_vert:g}$ m/s |"]
    if not res.ok:
        M.append(f"**Simulation failed:** {res.message}")
        return _join(M)
    S0, S = res.S_sep, res.S_td
    M += ["### Separation state (initial condition)",
          "| $y_0$ | $\\dot x_0$ | $\\dot y_0$ | $\\theta_0$ | $\\dot\\theta_0$ |", "|---:|---:|---:|---:|---:|",
          f"| ${S0[1] / 1e3:.1f}$ km | ${S0[3]:+.2f}$ m/s | ${S0[4]:+.2f}$ m/s | "
          f"${np.degrees(S0[2]):+.2f}^\\circ$ ({prm.theta_sep}) | "
          f"${np.degrees(S0[5]):+.2f}^\\circ$/s |",
          f"**RCS flip:** attitude held while climbing, then slewed after apex to "
          f"$\\theta_{{\\mathrm{{target}}}} = {np.degrees(sim.theta_target):+.2f}^\\circ$ "
          f"($|\\ddot\\theta| \\le {prm.rcs_alpha_degps2:g}^\\circ/\\mathrm{{s}}^2$, $|\\dot\\theta| \\le {prm.rcs_wmax_degps:g}^\\circ$/s). "
          f"At the descending $H_{{\\mathrm{{sep}}}}$ crossing: body vs flow ${np.degrees(res.theta_flow_Hsep):+.4f}^\\circ$ "
          f"(0 = tangent, engine first), $\\dot\\theta = {np.degrees(res.thdot_Hsep):+.4f}^\\circ$/s. Fins (LQR) below."]
    td = res.touchdown
    T, X, W = prm.td_bounds
    ps = prm.psi
    rows = [("$y_{\\mathrm{td}}$", _sci(S[1]), "m", f"${ps['y']:g}$ m", "—", abs(S[1]) < ps["y"]),
            ("$\\dot y_{\\mathrm{td}}$", _sci(S[4]), "m/s", f"${ps['ydot']:g}$ m/s", "—", abs(S[4]) < ps["ydot"]),
            ("tilt $\\theta_{\\mathrm{td}}$", f"{np.degrees(td[0]):+.2f}", "^\\circ",
             f"${np.degrees(ps['tilt_td']):g}^\\circ$", f"${T:g}^\\circ$", abs(np.degrees(td[0])) <= T),
            ("$\\dot x_{\\mathrm{td}}$", f"{td[1]:+.3f}", "m/s", f"${ps['xdot_td']:g}$ m/s", f"${X:g}$ m/s",
             abs(td[1]) <= X),
            ("$\\dot\\theta_{\\mathrm{td}}$", f"{np.degrees(td[2]):+.3f}", "^\\circ/\\mathrm{s}",
             f"${np.degrees(ps['thdot_td']):g}^\\circ$/s", f"${W:g}^\\circ$/s", abs(np.degrees(td[2])) <= W),
            ("$\\min_t y$ before touchdown", f"{res.y_min:.2f}", "m", "—", "$\\ge 0$", res.y_min > -ps["y"])]
    M += [f"### Touchdown &nbsp; (landing site $x = {S[0] / 1e3:+.3f}$ km from separation)",
          "| quantity | value | precision $\\psi$ | tolerance | |", "|---|---:|---:|---:|---|"]
    for name, val, unit, p, tol, ok in rows:
        u = f"\\,\\mathrm{{{unit}}}" if unit[0].isalpha() else unit
        M.append(f"| {name} | ${val}{u}$ | {p} | {tol} | {_ok(ok)} |")
    cap = prm.tank_capacity()
    M += ["### Propellant",
          f"$$m_{{\\mathrm{{f}}}}(0) = \\frac{{F_{{\\max}}\\,(t_{{\\mathrm{{entry}}}} + t_{{\\mathrm{{burn}}}})}}{{v_{{\\mathrm{{ex}}}}}}"
          f" = {f'{S0[6]:,.1f}'.replace(',', '{,}')}\\ \\mathrm{{kg}}"
          + f"\\quad({100 * S0[6] / cap:.1f}\\,\\%\\ \\text{{of tank}}),"
          f"\\qquad m_{{\\mathrm{{f}}}}(t_{{\\mathrm{{td}}}}) = {abs(S[6]):.2f}\\ \\mathrm{{kg}}$$"]

    def lim(val, limit, unit, fmt):
        if limit <= 0:
            return f"${fmt.format(val)}$ {unit} | — | —"
        return f"${fmt.format(val)}$ {unit} | ${fmt.format(limit)}$ {unit} | {_ok(val <= limit)}"
    i_qa, i_q = int(np.argmax(res.traj[:, 11])), int(np.argmax(res.traj[:, 8]))
    M += ["### Loads", "| quantity | max | limit | | at altitude |", "|---|---:|---:|---|---:|",
          f"| $q\\,\\lvert\\sin\\alpha\\rvert$ | " + lim(res.qa_max, prm.qa_limit, "Pa", "{:.0f}") + f" | ${res.traj[i_qa, 2] / 1e3:.1f}$ km |",
          f"| $q = \\tfrac12\\rho v^2$ | " + lim(res.q_max / 1e3, prm.q_limit / 1e3, "kPa", "{:.1f}") + f" | ${res.traj[i_q, 2] / 1e3:.1f}$ km |",
          f"| $a/g_0$ (sensed) | " + lim(res.a_max, prm.g_limit, "", "{:.2f}") + " | |"]
    if res.sites:
        M += ["### Heating", "| site | material | $\\delta$ | $R_n$ | $T_{\\max}$ | $T_{\\mathrm{melt}}$ | margin | |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
        for name, st in res.sites.items():
            M.append(f"| {name} | {st['material']} | ${st['delta'] * 1e3:.1f}$ mm | ${st['R_n']:g}$ m | "
                     f"${st['T_max']:.0f}$ K | ${st['T_melt']:.0f}$ K | ${st['margin']:+.0f}$ K | {_ok(st['margin'] >= 0)} |")
    if res.cost:
        sym = {"y": "(y_{\\mathrm{td}}/\\psi_y)^2", "ydot": "(\\dot y_{\\mathrm{td}}/\\psi_{\\dot y})^2",
               "fuel": "(m_{\\mathrm{f}}/\\psi_m)^2", "touchdown": "\\text{touchdown}",
               "ground": "\\text{ground}", "td_tol": "w_{\\mathrm{td}}[\\cdot]_+^2", "qalpha": "w_{q\\alpha}[\\cdot]_+^2", "q": "w_q[\\cdot]_+^2",
               "accel": "w_a[\\cdot]_+^2", "thermal": "w_T[\\cdot]_+^2"}
        terms = [(k, v) for k, v in res.cost.items() if k != "total"]
        M += ["### Error function",
              "$$E = " + " + ".join(sym.get(k, k) for k, _ in terms) + "$$",
              "| " + " | ".join(f"${sym.get(k, k)}$" for k, _ in terms) + " | $E$ |",
              "|" + "---:|" * (len(terms) + 1),
              "| " + " | ".join(f"${_sci(v, 3)}$" for _, v in terms) + f" | $\\mathbf{{{_sci(res.cost['total'], 4)}}}$ |"]
    st = res.stats
    extra = s.get("extra", {})
    M += ["<sub>" + f"nodes {len(res.traj)}, steps {int(st[0])} (+{int(st[1])} rejected), "
          f"|h| ∈ [{st[2]:.1e}, {st[3]:.2f}] s, {res.wall_time * 1e3:.2f} ms/sim, numba {'ON' if HAVE_NUMBA else 'OFF'}"
          + "".join(f"; {k}: {v}" for k, v in extra.items()) + "</sub>"]
    return _join(M)
