"""All parameters of the model, in one place.

`Params` is a plain mutable Python object (set from the settings cell of run_interactive.py).
`Params.pack()` flattens it into float64 arrays that the njit kernels read:
    P    - physics / numerics vector, indexed by the integer constants below
    prec - per-component error scale of the 7-state RK45 error norm
"""
import math

import numpy as np

# ---- indices into the packed vector P (njit kernels use only these) ----------
(V0, H_SEP, MDRY, VEX, K_OF, RHO_OX, RHO_F, D, M_S, M_TANK, H_0, H_S,
 EPS, COAST_STEP, BURN_STEP, T_LIMIT, FMAX, TVC, H_FIN,
 RHO0, H_ATM, T_ATM, R_GAS, GAMMA, R_EARTH, G0,
 A_FIN, N_FIN, SIGMA, C_OVER_S, C_NTH, M_CLAMP, FIN_FRAME, TUMBLE, Q_TUMBLE,
 CTRL_ON, CTRL_Q1, CTRL_Q2, CTRL_R, DELTA_MAX, V_BLEND, V_VERT, CTRL_RT, TVC_MAX,
 RCS_ALPHA, RCS_WN, RCS_TARGET, RCS_WMAX, CTRL_WMAX, CTRL_ZMAX, NP) = range(51)

# ---- status codes returned by the integrator --------------------------------
OK, ERR_STATE, ERR_TANK, ERR_STEP, ERR_UNSTABLE, ERR_SEP_EARLY, ERR_NO_SEP, ERR_BUF, ERR_TUMBLE = range(9)
STATUS_TEXT = {
    OK: "ok",
    ERR_STATE: "state outside physical range (y < -100 km, y > 1e7 m or NaN)",
    ERR_TANK: "fuel mass exceeds tank capacity M_w0",
    ERR_STEP: "step size underflow (|h| < 1e-9 s)",
    ERR_UNSTABLE: "unstable integration (NaN/inf truncation error)",
    ERR_SEP_EARLY: "separation event reached before end of entry burn",
    ERR_NO_SEP: "flight time t_free + t_entry + t_coast + t_burn exceeds t_limit",
    ERR_BUF: "trajectory buffer overflow (increase Params.n_buffer)",
    ERR_TUMBLE: "booster tumbles (body > 45 deg off the flow with q > q_tumble)",
}

# name: (rho_w [kg/m^3], c_w [J/(kg K)], T_melt [K])   (as wall_temperature.py)
MATERIALS = {
    "Al-Li 2195": (2700.0, 900.0, 850.0),
    "Ti-6Al-4V": (4430.0, 526.0, 1877.0),
    "Inconel 718": (8190.0, 435.0, 1533.0),
    "Stainless 304": (8000.0, 500.0, 1673.0),
}


class Params:
    def __init__(self):
        # ---------------- mission ----------------------------------------------
        self.v0 = 6000 / 3.6          # separation speed [m/s]
        self.sep_angle = 0.0          # flight-path angle of the separation velocity from vertical [rad]
        self.H_sep = 70000.0          # separation altitude [m]
        # ---------------- vehicle (sim1.py) -------------------------------------
        self.mdry = 130000.0          # dry mass [kg]
        self.v_exhaust = 3000.0       # [m/s]
        self.k = 3.6                  # oxidizer / fuel mass ratio
        self.rho_ox, self.rho_f = 1141.0, 422.0
        self.d = 7.0                  # diameter [m]
        self.m_s, self.m_tank = 60000.0, 70000.0
        self.H_0, self.H_s = 61.0, 4.0
        self.Fmax = 6.6e6             # [N]
        # ---------------- aerodynamics (booster_forces_lab.py) ------------------
        self.A_f = 12.5               # frontal area of one fin [m^2]
        self.n_f = 4                  # number of fins
        self.sigma = 0.05             # fin solidity
        self.c_over_s = 1.0           # fin chord / cell pitch
        self.C_Ntheta = 2.0           # body normal-force slope (2 slender body, 0 Newtonian)
        self.h_fins = 63.0            # fin hinge above the engine face [m] (= r_a + r_f of forces_lab)
        self.fin_frame = "body"       # fins are deflections delta relative to the body (LQR output)
        # ---------------- attitude control (LQR, dynamics.control): fins + engine gimbal ---------------
        self.ctrl_on = True
        self.e_max_deg = 2.0          # Bryson: acceptable attitude error  -> q1 = 1/e_max^2
        self.edot_max_degps = 2.0     #         acceptable attitude rate   -> q2 = 1/edot_max^2
        self.fin_bound_deg = 20.0     #         fin deflection limit (saturation) -> r = 1/delta_max^2
        self.tvc_max_deg = 0.0        #         engine gimbal limit; 0 = NO GIMBAL (this vehicle: fins only)
        self.ctrl_wmax = 1.5          # [rad/s] closed-loop bandwidth cap (fin actuator speed; also sets RK stiffness)
        self.ctrl_zmax = 1.0          #         damping-ratio cap
        self.v_blend = 100.0          # [m/s] attitude reference: retrograde (along the flow) above v_blend,
        self.v_vert = 20.0            # [m/s]   vertical below v_vert, linear blend in between
        self.tumble_deg = 45.0        # abort if |body - flow| exceeds this ...
        self.q_tumble = 1000.0        # ... while q > this [Pa] (~63 km at 1.7 km/s; above it aero is negligible)
        # ---------------- atmosphere ---------------------------------------------
        self.rho0 = 1.225
        self.H_atm = 8430.0           # scale height (dynamics + heating) [m]
        self.R_gas, self.gamma_gas = 287.05, 1.4
        self.g0, self.R_earth = 9.81, 6.4e6
        # ---------------- numerics ----------------------------------------------
        self.epsilon_err = 1.0
        self.coast_max_step = 1.0
        self.burn_max_step = 0.5
        self.t_limit = 5000.0
        self.precisions = np.array([0.1, 0.1, 1e-5, 0.1, 0.1, 1e-5, 100.0])
        self.n_buffer = 60000
        # ---------------- error function ----------------------------------------
        # Psi precisions at TOUCHDOWN: y [m], ydot [m/s] (solved inside every evaluation),
        # tilt [rad], xdot [m/s], thetadot [rad/s] (soft), m_fuel [kg] (loaded at separation)
        self.psi = {"y": 1.0, "ydot": 0.1, "m_fuel": 1e3,
                    "tilt_td": math.radians(5.0), "xdot_td": 1.0, "thdot_td": math.radians(2.0)}
        # touchdown tolerance (pass/fail in the verdict): tilt [deg], xdot [m/s], thetadot [deg/s]
        self.td_bounds = (15.0, 3.0, 3.0)
        self.q_limit = 0.0            # dynamic pressure limit [Pa]; 0 = off
        self.qa_limit = 4000.0        # q |sin(alpha)| limit [Pa]; 0 = off
        self.g_limit = 0.0            # sensed acceleration limit [g0]; 0 = off
        self.w_q, self.w_qa = 1e4, 1e4          # weights of the relative limit excesses
        self.w_accel, self.w_thermal = 1e4, 1e4
        self.w_td = 1e4               # weight of the relative excess over td_bounds
        self.limit_margin = 0.01      # cost aims 1 % inside every limit (soft penalty would sit just over it)
        # ---------------- heating (wall_temperature.py) -------------------------
        self.T_init = 250.0
        self.viscosity = "sutherland"
        self.melt_margin = 0.0
        self.sites = {"base": ["Al-Li 2195", 0.004, 7.0, True],
                      "fin": ["Ti-6Al-4V", 0.003, 0.005, True],
                      "nozzle": ["Stainless 304", 0.001, 0.002, True]}
        # ---------------- optimiser (forward shooting) ---------------------------
        self.theta_sep = "prograde"   # attitude AT separation: "prograde" (nose along the ascent velocity),
                                      # "tangent" (already at the RCS target), or a number [rad]
        # ---------------- RCS flip (dynamics.rcs) -------------------------------
        self.rcs_alpha_degps2 = 3.0   # cold-gas angular-acceleration limit [deg/s^2]
        self.rcs_wn = 0.5             # [rad/s] bandwidth of the final approach to the target attitude
        self.rcs_wmax_degps = 10.0    # slew-rate limit [deg/s]
        # The RCS holds the separation attitude while climbing, then (apex -> descending H_sep) slews to the
        # target theta_lab = -Omega(descending H_sep): tangent to the flow, engine first (simulate.rcs_target)
        self.thetadot_sep = 0.0       # pitch rate at separation [rad/s]
        self.bounds = [(0.0, None), (0.0, 100.0)]        # outer: t_delay (after descending H_sep; upper = no-burn
                                                         # ground impact, set per run), t_entry [s]
        self.inner_bounds = [(0.0, 300.0), (1.0, 100.0)]  # inner: t_coast, t_burn [s]
        self.starts = [(40.0, 2.0), (30.0, 5.0), (50.0, 1.0), (25.0, 10.0)]  # (entry-ignition altitude ON THE WAY DOWN [km], t_entry [s])
        self.maxiter = 40
        self.nm_xatol = 0.01         # Nelder-Mead stop: simplex size [s] ...
        self.nm_fatol = 0.01         # ... and spread of E (E ~ fuel in t, squared: 0.01 ~ 0.2 kg at 25 t)

    def T_atm(self):
        return self.g0 * 8000.0 / self.R_gas          # isothermal, as booster_forces_lab.py

    def pack(self):
        P = np.zeros(NP)
        vals = {V0: self.v0, H_SEP: self.H_sep, MDRY: self.mdry, VEX: self.v_exhaust, K_OF: self.k,
                RHO_OX: self.rho_ox, RHO_F: self.rho_f, D: self.d, M_S: self.m_s, M_TANK: self.m_tank,
                H_0: self.H_0, H_S: self.H_s, EPS: self.epsilon_err, COAST_STEP: self.coast_max_step,
                BURN_STEP: self.burn_max_step, T_LIMIT: self.t_limit, FMAX: self.Fmax,
                H_FIN: self.h_fins, RHO0: self.rho0, H_ATM: self.H_atm, T_ATM: self.T_atm(),
                R_GAS: self.R_gas, GAMMA: self.gamma_gas, R_EARTH: self.R_earth, G0: self.g0,
                A_FIN: self.A_f, N_FIN: self.n_f, SIGMA: self.sigma, C_OVER_S: self.c_over_s,
                C_NTH: self.C_Ntheta, M_CLAMP: math.sqrt(1.0 + self.c_over_s ** 2),
                FIN_FRAME: 1.0 if self.fin_frame == "lab" else 0.0,
                TUMBLE: math.radians(self.tumble_deg), Q_TUMBLE: self.q_tumble,
                CTRL_ON: 1.0 if self.ctrl_on else 0.0, CTRL_Q1: math.radians(self.e_max_deg) ** -2,
                CTRL_Q2: math.radians(self.edot_max_degps) ** -2, CTRL_R: math.radians(self.fin_bound_deg) ** -2,
                DELTA_MAX: math.radians(self.fin_bound_deg), V_BLEND: self.v_blend, V_VERT: self.v_vert,
                CTRL_RT: math.radians(max(self.tvc_max_deg, 1e-6)) ** -2, TVC_MAX: math.radians(self.tvc_max_deg),
                RCS_ALPHA: math.radians(self.rcs_alpha_degps2), RCS_WN: self.rcs_wn, RCS_TARGET: -self.sep_angle,
                RCS_WMAX: math.radians(self.rcs_wmax_degps), CTRL_WMAX: self.ctrl_wmax, CTRL_ZMAX: self.ctrl_zmax}
        for i, v in vals.items():
            P[i] = v
        return P, np.asarray(self.precisions, dtype=np.float64)

    def tank_capacity(self):
        return 0.25 * np.pi * self.d**2 * self.H_0 * (self.k + 1) / (1 / self.rho_f + self.k / self.rho_ox)

    def editable(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, (int, float, bool, str))}
