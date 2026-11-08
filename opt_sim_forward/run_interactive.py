# %% [markdown]
# # Booster landing - Opt_sim_forward (planar, LQR grid fins, RCS flip, forward shooting from separation)
# Run cell by cell (VS Code: "Run Cell" above each `# %%`, or Shift+Enter).
# 1 **Settings** (edit only these) - 2 **Run** - 3 **Plots** - 4 **Animation** - 5 **Sweep** over separation angle.
# Results: `Opt_sim_forward/runs/run_NNNN/` (review.txt, summary.json, trajectory.npz, figures, gif).

# %% [1] SETTINGS ==============================================================
import math
import os

MODE = "optimize"            # "optimize": entry + landing burn timing optimised (fins = LQR feedback)
                             # "shoot":    fix T_DELAY, T_ENTRY; (t_coast, t_burn) solved so it lands
                             # "single":   fix all four timings below
T_DELAY, T_ENTRY, T_COAST, T_BURN = 4.0, 2.5, 105.0, 6.3   # [s] entry ignition AFTER falling back through H_sep,
                                                            #     entry burn, coast, landing burn
THETA_SEP_DEG = "prograde"   # attitude AT separation: "prograde" (nose along the ascent velocity), "tangent"
                             # (already at the target) or a number [deg from vertical]
# ---- RCS (above H_sep): hold while climbing, then flip after apex so that the booster is EXACTLY tangent to the
#      flow, engine first, when it falls back through H_sep (target iterated on the no-burn reference flight)
RCS_ALPHA_DEGPS2 = 3.0       # [deg/s^2] angular-acceleration limit
RCS_WMAX_DEGPS = 10.0        # [deg/s]   slew-rate limit

# ---- attitude control (fins only, no engine gimbal): LQR on  e'' = a e + b_f delta ----------------------
E_MAX_DEG = 2.0              # [deg]   Bryson weights: acceptable attitude error ...
EDOT_MAX_DEGPS = 2.0         # [deg/s] ... and rate
FIN_BOUND = 20.0             # [deg]   fin deflection limit (saturation and LQR weight)
CTRL_WMAX = 1.5              # [rad/s] closed-loop bandwidth cap (fin actuator speed; higher = slower simulation)
V_BLEND, V_VERT = 100.0, 20.0   # [m/s] attitude reference: retrograde above V_BLEND, vertical below V_VERT

# ---- separation state (initial condition) and touchdown precisions -----------
SEP_ELEVATION_DEG = 20.0     # angle of the separation velocity ABOVE the horizontal (0 level, 90 vertical)
PSI = {"y": 1.0, "ydot": 0.1, "m_fuel": 1e3,   # touchdown m, m/s; fuel kg (1 t of fuel = 1 unit of E: fuel dominates)
       "tilt_td": math.radians(5.0), "xdot_td": 1.0, "thdot_td": math.radians(2.0)}  # rad, m/s, rad/s
# touchdown tolerance: tilt [deg], xdot [m/s], thetadot [deg/s] - enforced (penalty w_td), pass/fail in review.
# The required propellant depends strongly on these (70 deg case: 15 deg tilt -> ~18 t, ~0 deg -> ~160 t).
TD_BOUNDS = (15.0, 3.0, 6.0)

# ---- limits (None = off) --------------------------------------------------------
QALPHA_KPA_DEG = 78.0        # [kPa deg] bending-load limit q*alpha (78 kPa deg: launcher ascent value,
                             # arXiv:1910.03268) -> q|sin alpha| <= 78e3 * pi/180 = 1.36 kPa
QALPHA_LIMIT = QALPHA_KPA_DEG * 1e3 * math.pi / 180.0   # [Pa]  q |sin alpha|, alpha = body axis vs velocity
Q_LIMIT = None               # [Pa]  dynamic pressure q = rho v^2 / 2
G_LIMIT = None               # [g0]  sensed acceleration
MELT_MARGIN = 0.0            # [K]   T_wall <= T_melt - margin at every site

# ---- heating sites: name: (material, skin thickness [m], nose/edge radius [m]) ---
#  "Al-Li 2195" (850 K)  "Ti-6Al-4V" (1877 K)  "Inconel 718" (1533 K)  "Stainless 304" (1673 K)
SITES = {
    "base":   ("Al-Li 2195",    4e-3, 7.0),      # bottom of the booster as a whole
    "fin":    ("Ti-6Al-4V",     3e-3, 5e-3),     # grid-fin leading edge
    "nozzle": ("Stainless 304", 1e-3, 2e-3),     # thin nozzle lip / exposed base hardware
}

# ---- vehicle ----------------------------------------------------------------------
F_MAX, M_DRY, V_EXHAUST = 6.6e6, 130e3, 3000.0   # [N], [kg], [m/s]
C_NTHETA = 2.0               # body normal-force slope (2 slender body, 0 Newtonian)

# ---- optimiser / output -----------------------------------------------------------
# multi-start: the optimiser (Nelder-Mead, a local method) runs once from each point and keeps the best,
# because the landscape has several local minima. Point = (entry-ignition altitude on the way DOWN [km], t_entry [s])
STARTS = [(40.0, 2.0), (30.0, 5.0), (50.0, 1.0), (25.0, 10.0)]
ANIMATION = "gif"            # "gif" (saved + shown), "window" (interactive slider), or None

# ---- sweep (cell 5): propellant vs separation-velocity angle -----------------------------------
SWEEP_ANGLES = [70.0 * k / 9 for k in range(10)]   # [deg FROM VERTICAL] 0, 7.8, ..., 70 (= elevation 90 ... 20)
SWEEP_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # parallel processes (1 = serial, warm-started)
RUN_SWEEP = True             # ~0.5-1 min per point and core

# %% [2] RUN =======================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if not os.path.isfile(os.path.join(HERE, "params.py")):
    HERE = os.path.join(HERE, "Opt_sim_forward")
sys.path.insert(0, HERE)
_local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
for _name in list(sys.modules):                   # drop cached modules with these names (an earlier run
    if _name in _local:                           # of this folder OR of Opt_sim), so the files here
        del sys.modules[_name]                    # are always the ones imported

import numpy as np  # noqa: E402

from params import Params  # noqa: E402
from review import build_review, new_run_dir, review_markdown, save_run  # noqa: E402
from simulate import Simulator  # noqa: E402

np.seterr(all="ignore")
prm = Params()
prm.Fmax, prm.mdry, prm.v_exhaust, prm.C_Ntheta = F_MAX, M_DRY, V_EXHAUST, C_NTHETA
prm.fin_bound_deg, prm.ctrl_wmax = FIN_BOUND, CTRL_WMAX
prm.e_max_deg, prm.edot_max_degps, prm.v_blend, prm.v_vert = E_MAX_DEG, EDOT_MAX_DEGPS, V_BLEND, V_VERT
prm.sep_angle, prm.psi, prm.td_bounds = math.radians(90.0 - SEP_ELEVATION_DEG), dict(PSI), TD_BOUNDS
prm.qa_limit, prm.q_limit, prm.g_limit = QALPHA_LIMIT or 0.0, Q_LIMIT or 0.0, G_LIMIT or 0.0
prm.melt_margin = MELT_MARGIN
prm.sites = {k: [m, d, r, True] for k, (m, d, r) in SITES.items()}
prm.starts = STARTS
prm.theta_sep = THETA_SEP_DEG if isinstance(THETA_SEP_DEG, str) else math.radians(THETA_SEP_DEG)
prm.rcs_alpha_degps2, prm.rcs_wmax_degps = RCS_ALPHA_DEGPS2, RCS_WMAX_DEGPS
sim = Simulator(prm)
print(f"theta_sep = {math.degrees(sim.theta_sep):+.2f} deg ({THETA_SEP_DEG}), RCS target "
      f"{math.degrees(sim.theta_target):+.2f} deg; falls back through H_sep at "
      f"t = {sim.t_cross:.1f} s; entry ignition allowed from then until t = {sim.t_ground:.1f} s")

extra, history = {}, None
if MODE == "single":
    res = sim.run(sim.t_cross + T_DELAY, T_ENTRY, T_COAST, T_BURN, thermal=True)
    sim.cost_terms(res)
elif MODE == "shoot":
    from optimizer import solve_landing
    tc, tb, res = solve_landing(sim, sim.t_cross + T_DELAY, T_ENTRY)
    res.sites = None
    sim.cost_terms(res)
elif MODE == "optimize":
    from optimizer import optimize
    out = optimize(sim)
    res, history = out["res"], out["history"]
    extra = {"optimiser wall time [s]": f"{out['wall_time']:.1f}", "simulations": out["n_sims"],
             "best E": f"{out['best'][2]:.6g}"}
else:
    raise ValueError(f"unknown MODE {MODE!r}")

RUN_NO, RUN_DIR = new_run_dir()
text, summary = build_review(sim, res, RUN_NO, MODE, extra)
save_run(RUN_DIR, text, summary, res, prm)
md = review_markdown(sim, res, summary)
with open(os.path.join(RUN_DIR, "review.md"), "w", encoding="utf-8") as fh:
    fh.write(md + "\n")
try:                                    # Jupyter / VS Code interactive window: rendered LaTeX
    get_ipython  # noqa: B018, F821
    from IPython.display import Markdown, display
    display(Markdown(md))
except NameError:                       # plain terminal
    print(text)
print("saved to", RUN_DIR)

# %% [3] PLOTS =====================================================================
import matplotlib.pyplot as plt  # noqa: E402

from plotting import fig_convergence, fig_flight, figs_descent  # noqa: E402

files = []
if res.ok:
    files += figs_descent(res, prm, os.path.join(RUN_DIR, "fig"))
    files += fig_flight(res, prm, os.path.join(RUN_DIR, "fig_flight"))
if history:
    files += fig_convergence(history, os.path.join(RUN_DIR, "fig_convergence"))
try:
    from IPython.display import Image, display
    for f in files:
        if f.endswith(".png"):
            display(Image(filename=f, width=900))
except ImportError:
    pass
print("\n".join(f for f in files if f.endswith(".pdf")))

# %% [4] ANIMATION =================================================================
from animation import save_gif, show_animation  # noqa: E402

if res.ok and ANIMATION == "gif":
    gif = save_gif(res, prm, sim.P, os.path.join(RUN_DIR, "animation.gif"))
    try:
        from IPython.display import Image, display
        display(Image(filename=gif))
    except ImportError:
        pass
    print(gif)
elif res.ok and ANIMATION == "window":
    try:                                              # Jupyter / VS Code interactive window
        get_ipython().run_line_magic("matplotlib", "tk")  # noqa: F821
    except NameError:
        plt.switch_backend("TkAgg")
    show_animation(res, prm, sim.P)

# %% [5] SWEEP: propellant vs separation angle ======================================
from plotting import fig_sweep  # noqa: E402
from review import RUNS_DIR  # noqa: E402
from sweep import new_sweep_dir, save_sweep, sweep_propellant  # noqa: E402

if RUN_SWEEP:
    prm_sweep = Params()                      # same settings as cell 2, angle and theta set per point
    prm_sweep.__dict__.update({k: v for k, v in prm.__dict__.items() if k != "sep_angle"})
    print(f"sweep: {len(SWEEP_ANGLES)} angles on {SWEEP_WORKERS} process(es) ...")
    sweep_rows = sweep_propellant(prm_sweep, SWEEP_ANGLES, workers=SWEEP_WORKERS)
    SWEEP_DIR = new_sweep_dir(RUNS_DIR)
    save_sweep(sweep_rows, SWEEP_DIR)
    sweep_files = fig_sweep(sweep_rows, os.path.join(SWEEP_DIR, "fig_sweep_propellant"))
    try:
        get_ipython  # noqa: B018, F821
        from IPython.display import Image, Markdown, display
        nan = float("nan")
        hdr = ("| angle from vertical | $m_{\\mathrm{f}}$ [t] | $\\theta_{\\mathrm{sep}}$ | entry: ignition alt., $t_{\\mathrm{entry}}$ [s] | "
               "tilt$_{\\mathrm{td}}$ | $\\dot x_{\\mathrm{td}}$ [m/s] | $\\dot\\theta_{\\mathrm{td}}$ [°/s] | "
               "$\\max q\\lvert\\sin\\alpha\\rvert$ [Pa] | feasible |")
        lines = [hdr, "|---:|---:|---:|---:|---:|---:|---:|---:|---|"] + [
            f"| ${r['angle_from_vertical']:.1f}^\\circ$ | ${r['m_fuel'] / 1e3:.2f}$ | "
            f"${r.get('theta_sep_deg', nan):+.2f}^\\circ$ | ${r.get('h_ignition_km', nan):.1f}$ km, ${r.get('t_entry', nan):.2f}$ | "
            f"${r.get('tilt_td_deg', nan):+.2f}^\\circ$ | ${r.get('xdot_td', nan):+.2f}$ | "
            f"${r.get('thdot_td_degps', nan):+.2f}$ | ${r.get('qa_max', nan):.0f}$ | "
            f"{'✅' if r['feasible'] else '❌'} |" for r in sweep_rows]
        display(Markdown("\n".join(lines)))
        display(Image(filename=sweep_files[1], width=600))
    except NameError:
        pass
    print("sweep saved to", SWEEP_DIR)

# %%
