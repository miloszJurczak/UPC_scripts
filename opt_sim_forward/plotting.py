"""Publication (PRL) figures.

Physical Review sizes: single column 3.375 in, double column 7.0 in; Times text with
STIX math; inward ticks on all sides; panel letters instead of titles; PDF + 600 dpi PNG.
Axis labels:  "Quantity ($symbol$) [unit]".
Conventions used in every figure
    data            one dark-blue line per panel (one quantity per panel, no colour coding)
    limits          red dashed line, labelled directly with its value
    engine burns    orange bands (entry burn, landing burn), labelled in the top panels
"""
import matplotlib.pyplot as plt
import numpy as np

from simulate import deltas_to_times

COL1, COL2 = 3.375, 7.0
LINE = "#0B3C8C"          # data
LIMIT = "#C62828"         # thresholds
BURN = "#F28E2B"          # engine-on bands
MARK = ["o", "s", "^", "D"]
CAT = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]   # categorical (only where categories are compared)


def prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 3.5,
        "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _panel(ax, letter):
    ax.text(0.02, 0.96, f"({letter})", transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=6)


def _limit(ax, value, text, where="right"):
    ax.axhline(value, color=LIMIT, ls="--", lw=1.1, zorder=4)
    x = 0.98 if where == "right" else 0.02
    ax.text(x, value, text, transform=ax.get_yaxis_transform(), color=LIMIT, fontsize=8,
            ha=where, va="bottom", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.8))


def _save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    return [path + ".pdf", path + ".png"]


def _burn_bands(ax, res, label=False):
    T_burn, T1, T2 = deltas_to_times(*res.timings)
    for (a, b), name in (((-T1, -T2), "entry burn"), ((-T_burn, 0.0), "landing burn")):
        ax.axvspan(a, b, color=BURN, alpha=0.35, lw=0, zorder=0)
        if label:
            ax.text(0.5 * (a + b), 1.02, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", color="#B35C00", fontsize=8)


def descent_window(res, prm, pad=5.0):
    """From the descending crossing of y = H_sep (or the entry burn, if earlier) to touchdown."""
    t, y, yd = res.traj[:, 0], res.traj[:, 2], res.traj[:, 5]
    down = (yd < 0) & (y >= prm.H_sep)                 # nodes still above H_sep while falling
    t_cross = t[down].max() if down.any() else t[-1]   # last moment above H_sep on the way down
    _, T1, _ = deltas_to_times(*res.timings)
    return min(t_cross, -T1) - pad, 0.0


def _partition_lines(ax, res):
    T_burn, T1, T2 = deltas_to_times(*res.timings)
    for a, b, n in ((-T1, -T2, 3), (-T2, -T_burn, 10), (-T_burn, 0.0, 3)):
        for k in range(1, n):
            ax.axvline(a + k * (b - a) / n, color="0.75", lw=0.5, zorder=0)


def _time_derivative(t, v):
    """Safe numerical derivative dv/dt (handles duplicate time knots at phase boundaries)."""
    _, idx = np.unique(t, return_index=True)
    return np.interp(t, t[idx], np.gradient(v[idx], t[idx]))


# one colour per quantity, the same in every figure; the y label wears the curve colour.
# Orange is reserved for the burn bands and red for limits, so no curve uses them.
QCOL = {"y": "#0072B2", "x": "#009E73", "ydot": "#5E3C99", "a": "#CC79A7",
        "xdot": "#00796B", "xddot": "#2E7D32", "yddot": "#6A1B9A",
        "theta": "#1F3A93", "thflow": "#3FA7D6", "fin": "#7B1FA2",
        "q": "#0072B2", "qa": "#5E3C99", "base": "#455A64", "finT": "#009E73",
        "nozzle": "#AD1457", "mf": "#0072B2"}


def _grid(n, single=False):
    if single:
        return plt.subplots(1, 1, figsize=(COL1, 2.4), constrained_layout=True, squeeze=False)
    if n in (4, 6):
        cols = 2
        rows = n // 2
    else:
        cols = n
        rows = 1
    return plt.subplots(rows, cols, figsize=(COL2, 2.5 * rows), sharex=True,
                        constrained_layout=True, squeeze=False)


def _panels(res, prm, panels, path, single=False):
    """panels: list of (y-values, label, colour key, limit or None, limit text, extra(ax))."""
    prl_style()
    tr = res.traj
    t = tr[:, 0]
    t0, t1 = descent_window(res, prm)
    fig, ax = _grid(len(panels), single)
    axs = ax.ravel()
    is_two_col = len(panels) in (4, 6)
    for k, (a, (tt, yv, lab, ck, lim, ltxt, extra)) in enumerate(zip(axs, panels)):
        tt = t if tt is None else tt
        mm = tt >= t0
        col = QCOL[ck]
        a.plot(tt[mm], yv[mm], color=col, lw=1.8)
        a.set_ylabel(lab, color=col)
        a.tick_params(axis="y", colors=col, which="both")
        if lim is not None and lim > 0:
            _limit(a, lim, ltxt)
        if extra:
            extra(a)
        _burn_bands(a, res, label=k < (2 if is_two_col else len(panels)))
        _panel(a, "abcdefgh"[k])
        a.set_xlim(t0, t1)
        if not is_two_col or k >= len(panels) - 2:
            a.set_xlabel(r"Time to touchdown ($t$) [s]")
    return _save(fig, path)


def figs_descent(res, prm, prefix):
    """Five figures of the descent (H_sep -> touchdown), each with its own panel letters:
    trajectory 3x2 (col 1: X, col 2: Y; rows: position, velocity, acceleration),
    attitude (theta_lab, theta, fin), loads (q, q sin a, a),
    heating (one panel per site), propellant."""
    tr = res.traj
    t = tr[:, 0]
    deg = np.degrees
    thf = np.where(np.abs(tr[:, 13]) > np.pi / 2, tr[:, 13] - np.sign(tr[:, 13]) * np.pi, tr[:, 13])
    fin = tr[:, 12] - tr[:, 3] if prm.fin_frame == "body" else tr[:, 12]
    B = prm.fin_bound_deg

    xddot = _time_derivative(t, tr[:, 4])
    yddot = _time_derivative(t, tr[:, 5])

    def fin_extra(a):
        for v in (-B, B):
            a.axhline(v, color=LIMIT, ls=":", lw=0.9)
        t0, _ = descent_window(res, prm)
        m = tr[:, 0] >= t0
        if np.any(tr[:, 14] != 0):
            a.plot(tr[m, 0], np.degrees(tr[m, 14]), color="#AD1457", lw=1.0, ls="--", label=r"gimbal $\tau$")
            a.legend(fontsize=7, loc="lower left")

    files = []
    # Macierz 3x2: kolumna 1 -> X, kolumna 2 -> Y | wiersze: położenie, prędkość, przyspieszenie
    files += _panels(res, prm, [
        (None, tr[:, 1] / 1e3, r"Horizontal position ($x$) [km]", "x", None, "", None),
        (None, tr[:, 2] / 1e3, r"Altitude ($y$) [km]", "y", None, "", None),
        (None, tr[:, 4], r"Horizontal velocity ($\dot x$) [m/s]", "xdot", None, "", None),
        (None, tr[:, 5], r"Vertical velocity ($\dot y$) [m/s]", "ydot", None, "", None),
        (None, xddot, r"Horizontal acceleration ($\ddot x$) [m/s$^2$]", "xddot", None, "", None),
        (None, yddot, r"Vertical acceleration ($\ddot y$) [m/s$^2$]", "yddot", None, "", None),
    ], prefix + "_trajectory")

    files += _panels(res, prm, [
        (None, deg(tr[:, 3]), r"Pitch angle ($\theta_{\mathrm{lab}}$) [deg]", "theta", None, "", None),
        (None, deg(thf), r"Body vs flow ($\theta$) [deg]", "thflow", None, "", None),
        (None, deg(fin), r"Fin deflection ($\delta$) [deg]", "fin", None, "", fin_extra),
    ], prefix + "_attitude")

    files += _panels(res, prm, [
        (None, tr[:, 8] / 1e3, r"Dynamic pressure ($q$) [kPa]", "q", prm.q_limit / 1e3,
         rf"$q_{{\max}} = {prm.q_limit / 1e3:g}$ kPa", None),
        (None, tr[:, 11] / 1e3, r"Bending load ($q\,|\sin\alpha|$) [kPa]", "qa", prm.qa_limit / 1e3,
         rf"$q_{{\alpha,\max}} = {prm.qa_limit / 1e3:g}$ kPa", None),
        (None, tr[:, 10], r"Sensed acceleration ($a$) [$g_0$]", "a", prm.g_limit,
         rf"$a_{{\max}} = {prm.g_limit:g}\,g_0$", None),
    ], prefix + "_loads")

    heat = []
    for name, s in (res.sites or {}).items():
        rn = f"{s['R_n']:g} m" if s["R_n"] >= 1 else f"{s['R_n'] * 1e3:g} mm"
        info = f"{s['material']}, $\\delta$ = {s['delta'] * 1e3:g} mm, $R_n$ = {rn}"

        def extra(a, info=info):
            a.set_title(info, fontsize=7.5, pad=14)
        ck = {"base": "base", "fin": "finT", "nozzle": "nozzle"}.get(name, "base")
        heat.append((s["t"], s["T"], rf"Wall temp., {name} ($T_{{\mathrm{{w}}}}$) [K]", ck,
                     s["T_melt"] - prm.melt_margin, rf"$T_{{\mathrm{{melt}}}} = {s['T_melt']:.0f}$ K", extra))
    if heat:
        files += _panels(res, prm, heat, prefix + "_heating")
    files += _panels(res, prm, [
        (None, tr[:, 7] / 1e3, r"Propellant ($m_{\mathrm{f}}$) [t]", "mf", None, "", None)],
        prefix + "_propellant", single=True)
    return files


def fig_flight(res, prm, path):
    """Whole flight, single column: altitude vs time; grey = window of the descent figure."""
    prl_style()
    fig, ax = plt.subplots(figsize=(COL1, 2.2), constrained_layout=True)
    t = res.traj[:, 0]
    ax.plot(t, res.traj[:, 2] / 1e3, color=LINE)
    t0, _ = descent_window(res, prm)
    ax.axvspan(t0, 0, color="0.85", lw=0, zorder=0)
    ax.text(0.5 * t0, 1.02, "descent figure", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    _burn_bands(ax, res)
    _limit(ax, prm.H_sep / 1e3, rf"$H_{{\mathrm{{sep}}}} = {prm.H_sep / 1e3:g}$ km", where="left")
    ax.set_xlim(t.min(), 0)
    ax.set_xlabel(r"Time to touchdown ($t$) [s]")
    ax.set_ylabel(r"Altitude ($y$) [km]")
    return _save(fig, path)


def fig_convergence(history, path):
    """Best error-function value so far vs evaluation; one curve per stage-1 start, then stage 2."""
    prl_style()
    fig, ax = plt.subplots(figsize=(COL1, 2.3), constrained_layout=True)
    H = [r for r in history if r[2] < 1e29]
    keys = list(dict.fromkeys((r[0], r[1]) for r in H))
    for k, (stage, sid) in enumerate(keys):
        c = np.array([r[2] for r in H if (r[0], r[1]) == (stage, sid)])
        lab = f"start {sid + 1}"
        ax.plot(np.arange(1, len(c) + 1), np.minimum.accumulate(c), color=CAT[k % 4],
                lw=1.0, marker=MARK[k % 4], ms=3, markevery=max(1, len(c) // 10), label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"Objective evaluation ($n$) [-]")
    ax.set_ylabel(r"Best error so far ($E_{\min}$) [-]")
    ax.legend(fontsize=7)
    return _save(fig, path)


def fig_thermal_sweep(table, path):
    """Melting margin T_melt - T_w,max vs skin thickness; below the red line = melts."""
    prl_style()
    sites = list(dict.fromkeys(r["site"] for r in table))
    mats = list(dict.fromkeys(r["material"] for r in table))
    fig, axs = plt.subplots(1, len(sites), figsize=(COL2, 2.6), constrained_layout=True, squeeze=False)
    for j, site in enumerate(sites):
        ax = axs[0, j]
        for i, mat in enumerate(mats):
            rows = sorted((r for r in table if r["site"] == site and r["material"] == mat), key=lambda r: r["delta"])
            d = np.array([r["delta"] for r in rows]) * 1e3
            mg = np.array([r["T_melt"] - r["T_max"] for r in rows])
            ax.plot(d, mg, color=CAT[i % 4], lw=1.2, marker=MARK[i % 4], ms=4, label=mat)
        ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -1, 0, color=LIMIT, alpha=0.08, lw=0)
        _limit(ax, 0.0, "melts below")
        ax.set_xlabel(r"Skin thickness ($\delta$) [mm]")
        ax.set_ylabel(r"Melting margin ($T_{\mathrm{melt}} - T_{\mathrm{w,max}}$) [K]")
        ax.set_title(site, fontsize=9)
        _panel(ax, "abcdef"[j])
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=len(mats))
    return _save(fig, path)


def fig_sweep(rows, path):
    """Propellant loaded at separation vs separation-velocity angle; one curve per theta mode.
    Filled markers: all constraints met; open markers: infeasible (landing or a limit violated)."""
    prl_style()
    fig, ax = plt.subplots(figsize=(COL1, 2.5), constrained_layout=True)
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    lab = {"tangent": r"$\theta_{\mathrm{sep}}$ tangent at descending $H_{\mathrm{sep}}$, LQR fins"}
    for k, mode in enumerate(modes):
        R = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["angle_from_vertical"])
        a = np.array([r["angle_from_vertical"] for r in R])
        m = np.array([r["m_fuel"] for r in R]) / 1e3
        ok = np.array([r["feasible"] for r in R])
        ax.plot(a, m, color=CAT[k % 4], lw=1.2, label=lab.get(mode, mode))
        ax.plot(a[ok], m[ok], ls="none", marker=MARK[k % 4], ms=4, color=CAT[k % 4])
        ax.plot(a[~ok], m[~ok], ls="none", marker=MARK[k % 4], ms=4, mfc="white", color=CAT[k % 4])
    ax.set_xlabel(r"Separation velocity from vertical ($a$) [deg]")
    ax.set_ylabel(r"Propellant at separation ($m_{\mathrm{f}}$) [t]")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 90 - x, lambda e: 90 - e))
    sec.set_xlabel(r"Elevation above horizontal [deg]")
    ax.legend(fontsize=7)
    return _save(fig, path)