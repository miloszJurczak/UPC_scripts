"""Animation of a simulated descent (after UPC booster_animation.py), with the engine flame.

Panels
  left  : trajectory (x, altitude), booster glyph enlarged, drawn at the current time
  right : true-scale close-up about the CM: body, grid fins (one drawn on each side, angle gamma),
          flame while the engines burn, velocity (red) and body-axis (blue dashed) arrows
  bottom: theta_lab, body-vs-flow theta and fin angle vs time, with a cursor
Two ways to use it
  show_animation(res, prm)              interactive window: time slider + Play/Pause
  save_gif(res, prm, path, fps=20)      GIF of the descent (from H_sep down to touchdown)
Conventions as booster_forces_lab.py: body axis a = (sin th, cos th), engine face at -r_a,
fins at +r_f = h_fins - r_a, fin line e = (cos gamma, -sin gamma).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from plotting import descent_window
from vehicle import r_cm

BODY, DARK, FIN, FLAME_OUT, FLAME_IN = "#d9d9d9", "#555555", "#1f5fbf", "#F28E2B", "#FFD34D"


def _outline(L, d, r_a):
    """Booster polygons in body coords (s along axis from CM, w across)."""
    s_bot, s_top, h = -r_a, L - r_a, d / 2
    body = [(s_bot, -h), (s_top, -h), (s_top, h), (s_bot, h)]
    bell = [(s_bot, -0.35 * d), (s_bot - 3.5, -0.45 * d), (s_bot - 3.5, 0.45 * d), (s_bot, 0.35 * d)]
    legs = [(s_bot + 1, h), (s_bot + 12, h), (s_bot + 1, h + 1.0)]
    return [np.array(p, float) for p in (body, bell, legs, [(s, -w) for s, w in legs])]


def _flame(d, r_a, length, k):
    s0, w = -r_a - 3.5, k * 0.45 * d
    return np.array([(s0, -w), (s0 - 0.35 * length, -0.7 * w), (s0 - length, 0.0),
                     (s0 - 0.35 * length, 0.7 * w), (s0, w)], float)


def _to_world(poly, X, Y, th, scale):
    a = np.array([np.sin(th), np.cos(th)])
    p = np.array([np.cos(th), -np.sin(th)])
    pts = poly[:, :1] * a + poly[:, 1:] * p
    return np.column_stack([X + scale * pts[:, 0], Y + scale * pts[:, 1]])


class _Scene:
    def __init__(self, res, prm, sim_P, interactive):
        f = res.traj[::-1]                                    # forward time
        self.t = f[:, 0]
        self.X, self.Y, self.th = f[:, 1], f[:, 2], f[:, 3]
        self.vx, self.vy, self.mf = f[:, 4], f[:, 5], f[:, 7]
        self.beta, self.a, self.gam, self.thf = f[:, 9], f[:, 10], f[:, 12], f[:, 13]
        self.q = f[:, 8]
        self.vs = np.sqrt(prm.gamma_gas * prm.R_gas * prm.T_atm())
        self.L, self.d, self.hf, self.P = prm.H_0 + prm.H_s, prm.d, prm.h_fins, sim_P
        self.fin_w = np.sqrt(prm.A_f)
        self.fig = plt.figure(figsize=(13, 8.5))
        top = 0.40 if interactive else 0.36
        self.ax_tr = self.fig.add_axes([0.05, top, 0.42, 0.55])
        self.ax_zm = self.fig.add_axes([0.56, top, 0.40, 0.55])
        self.ax_an = self.fig.add_axes([0.06, 0.08 if not interactive else 0.13, 0.91, 0.20])
        self._static(res)

    def _static(self, res):
        ax = self.ax_tr
        Xk, Yk = self.X / 1e3, self.Y / 1e3
        ax.plot(Xk, Yk, color="0.75", lw=1, label="full trajectory")
        self.done, = ax.plot([], [], color="C3", lw=2, label="flown so far")
        span = max(np.ptp(Xk), np.ptp(Yk)) * 1.1 + 1.0
        xc, yc = 0.5 * (Xk.min() + Xk.max()), 0.5 * (Yk.min() + Yk.max())
        ax.set_xlim(xc - span / 2, xc + span / 2)
        y0 = max(yc - span / 2, -0.5)
        ax.set_ylim(y0, y0 + span)
        ax.set_aspect("equal", adjustable="box")
        self.scale = 0.12 * span / self.L
        ax.set_xlabel(r"Horizontal position ($x$) [km]")
        ax.set_ylabel(r"Altitude ($y$) [km]")
        ax.set_title(f"booster glyph enlarged x{self.scale * 1e3:.0f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        mk = lambda a, c, lw: a.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=c, ec="k", lw=lw))
        cols = [BODY, DARK, DARK, DARK]
        self.g_tr = [mk(ax, c, 0.5) for c in cols]
        self.fl_tr = [mk(ax, FLAME_OUT, 0), mk(ax, FLAME_IN, 0)]
        self.fin_tr, = ax.plot([], [], color=FIN, lw=3, solid_capstyle="butt", zorder=5)
        z = self.ax_zm
        R = 0.95 * self.L
        z.set_xlim(-R, R)
        z.set_ylim(-R, R)
        z.set_aspect("equal")
        z.axvline(0, color="0.6", lw=0.8, ls=":")
        z.plot(0, 0, "k+", ms=10)
        z.set_xlabel(r"$x - x_{\mathrm{CM}}$ [m]")
        z.set_ylabel(r"$y - y_{\mathrm{CM}}$ [m]")
        z.set_title("close-up about the CM (true scale)", fontsize=9)
        self.fl_zm = [mk(z, FLAME_OUT, 0), mk(z, FLAME_IN, 0)]
        self.g_zm = [mk(z, c, 0.8) for c in cols]
        self.fin_zm, = z.plot([], [], color=FIN, lw=6, solid_capstyle="butt", zorder=5)
        self.arr_v = z.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C3", lw=2))
        self.arr_b = z.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.5, ls="--"))
        z.plot([], [], color="C3", lw=2, label="velocity")
        z.plot([], [], color="C0", lw=1.5, ls="--", label="body axis")
        z.legend(loc="lower right", fontsize=8)
        self.info = z.text(0.02, 0.98, "", transform=z.transAxes, va="top", family="monospace", fontsize=8.5,
                           bbox=dict(fc="white", ec="0.7", alpha=0.9))
        a = self.ax_an
        thf = np.where(np.abs(self.thf) > np.pi / 2, self.thf - np.sign(self.thf) * np.pi, self.thf)
        a.plot(self.t, np.degrees(self.th), color="C0", label=r"$\theta_{\mathrm{lab}}$")
        a.plot(self.t, np.degrees(thf), color="C2", label=r"$\theta$ (body vs flow)")
        a.plot(self.t, np.degrees(self.gam - self.th), color=FIN, ls="--", label=r"fin $\delta$")
        on = self.beta > 0.5
        a.fill_between(self.t, 0, 1, where=on, transform=a.get_xaxis_transform(), color=FLAME_OUT,
                       alpha=0.3, lw=0, label="engine on")
        self.cursor = a.axvline(self.t[0], color="C3", lw=1.2)
        a.set_xlabel(r"Time to touchdown ($t$) [s]")
        a.set_ylabel("Angle [deg]")
        a.set_xlim(self.t[0], self.t[-1])
        a.legend(loc="upper left", fontsize=8, ncol=4)

    def focus(self, t0):
        """Limit the angle panel's y range to the data after t0."""
        k = self.t >= t0
        thf = np.where(np.abs(self.thf) > np.pi / 2, self.thf - np.sign(self.thf) * np.pi, self.thf)
        v = np.degrees(np.r_[self.th[k], thf[k], self.gam[k] - self.th[k]])
        pad = 0.1 * (np.ptp(v) + 2.0)
        self.ax_an.set_ylim(v.min() - pad, v.max() + pad)

    def draw(self, t):
        f = lambda v: np.interp(t, self.t, v)
        X, Y, th, vx, vy, mf, gam = f(self.X), f(self.Y), f(self.th), f(self.vx), f(self.vy), f(self.mf), f(self.gam)
        k = min(np.searchsorted(self.t, t), len(self.t) - 1)
        burning = self.beta[k] > 0.5
        r_a = r_cm(mf, self.P)
        polys = _outline(self.L, self.d, r_a)
        self.done.set_data(np.r_[self.X[:k], X] / 1e3, np.r_[self.Y[:k], Y] / 1e3)
        for (glist, x0, y0, sc) in ((self.g_tr, X / 1e3, Y / 1e3, self.scale), (self.g_zm, 0.0, 0.0, 1.0)):
            for patch, poly in zip(glist, polys):
                patch.set_xy(_to_world(poly, x0, y0, th, sc))
        flick = 1.0 + 0.15 * np.sin(37.0 * t) + 0.1 * np.sin(91.0 * t)
        for (flist, x0, y0, sc) in ((self.fl_tr, X / 1e3, Y / 1e3, self.scale), (self.fl_zm, 0.0, 0.0, 1.0)):
            for patch, (Lf, kw) in zip(flist, ((28.0 * flick, 1.0), (16.0 * flick, 0.55))):
                patch.set_visible(burning)
                patch.set_xy(_to_world(_flame(self.d, r_a, Lf, kw), x0, y0, th, sc))
        a = np.array([np.sin(th), np.cos(th)])
        p = np.array([np.cos(th), -np.sin(th)])
        e = np.array([np.cos(gam), -np.sin(gam)])          # fin line (perpendicular to its normal)
        c = (self.hf - r_a) * a
        for ln, x0, y0, sc in ((self.fin_tr, X / 1e3, Y / 1e3, self.scale), (self.fin_zm, 0, 0, 1.0)):
            xs, ys = [], []
            for side in (-1.0, 1.0):                         # one fin on each side of the body
                ctr = c + side * (0.5 * self.d + 0.5 * self.fin_w) * p
                p0, p1 = ctr - 0.5 * self.fin_w * e, ctr + 0.5 * self.fin_w * e
                xs += [x0 + sc * p0[0], x0 + sc * p1[0], np.nan]
                ys += [y0 + sc * p0[1], y0 + sc * p1[1], np.nan]
            ln.set_data(xs, ys)
        V = np.hypot(vx, vy)
        lv = 0.5 * self.L
        if V > 1e-6:
            self.arr_v.xy = (lv * vx / V, lv * vy / V)
        self.arr_b.xy = (lv * a[0], lv * a[1])
        self.cursor.set_xdata([t, t])
        self.info.set_text(f"t   = {t:8.2f} s\nh   = {Y / 1e3:8.2f} km\nx   = {X / 1e3:8.3f} km\n"
                           f"M   = {V / self.vs:8.2f}\n|a| = {f(self.a):8.2f} g0\nq   = {f(self.q) / 1e3:8.2f} kPa\n"
                           f"m_f = {mf / 1e3:8.2f} t\nth  = {np.degrees(th):+8.2f} deg\n"
                           f"fin = {np.degrees(gam - th):+8.2f} deg\nengine {'ON ' if burning else 'off'}")


def show_animation(res, prm, sim_P, play_seconds=20.0):
    """Interactive window (slider + Play). Needs a GUI backend (e.g. TkAgg)."""
    from matplotlib.widgets import Button, Slider
    sc = _Scene(res, prm, sim_P, interactive=True)
    ax_sl = sc.fig.add_axes([0.20, 0.035, 0.6, 0.03])
    ax_bt = sc.fig.add_axes([0.06, 0.025, 0.07, 0.05])
    slider = Slider(ax_sl, "t [s]", sc.t[0], sc.t[-1], valinit=sc.t[0])
    button = Button(ax_bt, "Play")
    state = {"play": False}
    slider.on_changed(lambda v: (sc.draw(v), sc.fig.canvas.draw_idle()))
    timer = sc.fig.canvas.new_timer(interval=40)

    def step():
        if not state["play"]:
            return
        T = sc.t[-1] - sc.t[0]
        v = slider.val + T * 0.04 / play_seconds
        if v >= sc.t[-1]:
            v, state["play"] = sc.t[-1], False
            button.label.set_text("Play")
            timer.stop()
        slider.set_val(v)

    def on_play(_):
        state["play"] = not state["play"]
        if state["play"]:
            if slider.val >= sc.t[-1]:
                slider.set_val(sc.t[0])
            button.label.set_text("Pause")
            timer.start()
        else:
            button.label.set_text("Play")
            timer.stop()

    timer.add_callback(step)
    button.on_clicked(on_play)
    sc.draw(sc.t[0])
    sc._keep = (slider, button, timer)                 # keep widgets alive
    plt.show()
    return sc


def save_gif(res, prm, sim_P, path, fps=15, seconds=10.0, full=False):
    """GIF of the descent window (or the full flight with full=True)."""
    sc = _Scene(res, prm, sim_P, interactive=False)
    t0 = sc.t[0] if full else descent_window(res, prm)[0]
    sc.ax_an.set_xlim(t0, 0.0)
    sc.focus(t0)
    times = np.linspace(t0, sc.t[-1], int(fps * seconds))
    anim = FuncAnimation(sc.fig, lambda i: sc.draw(times[i]), frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(sc.fig)
    return path
