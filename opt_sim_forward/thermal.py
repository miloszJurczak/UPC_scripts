"""Stagnation-point skin temperature (hot wall, no radiation) - njit port of
UPC/wall_temperature.py, used as the melting cost of the optimisation.

    C_w dT_w/dt = G c_p (T_stag - T_w),           C_w = rho_w c_w delta     [J/(m^2 K)]
    T_stag = T_atm + v^2 / (2 c_p)                (total enthalpy conserved)
    G = 0.763 Pr^-0.6 sqrt(rho_e mu_e / R_n) (2 (p_e0 - p_inf) / rho_e)^(1/4)
    p_inf = rho R T_atm,  p_e0 - p_inf = cp_max(M) rho v^2 / 2,
    rho_e = p_e0 / (R T_stag),  mu_e = mu(T_stag)
Exact update for coefficients frozen at the step midpoint:
    T_{n+1} = T_stag + (T_n - T_stag) exp(-dt / tau),   tau = C_w / (G c_p)

Atmosphere: same scale height as the dynamics (Params.H_atm).
The trajectory is integrated backward, so heating is evaluated afterwards on the
time-reversed (forward) trajectory, starting from T_init at separation: the heat
equation is dissipative and must not be integrated backward.
"""
import math

import numpy as np

from atmosphere import cp_max
from jit import njit
from params import MATERIALS

PR = 0.71                    # Prandtl number of air
SIM = 0.763                  # Sibulkin / Fay-Riddell stagnation-point constant
MU_REF, T_REF, S_SUTH, OMEGA = 1.716e-5, 273.15, 110.4, 0.7

# packed thermal vector TP
TP_RHO0, TP_H, TP_R, TP_TATM, TP_GAMMA, TP_VS, TP_VISC, TP_N = range(8)


def pack_thermal(prm):
    TP = np.zeros(TP_N)
    TP[TP_RHO0], TP[TP_H], TP[TP_R] = prm.rho0, prm.H_atm, prm.R_gas
    TP[TP_TATM], TP[TP_GAMMA] = prm.T_atm(), prm.gamma_gas
    TP[TP_VS] = math.sqrt(prm.gamma_gas * prm.R_gas * prm.T_atm())
    TP[TP_VISC] = 1.0 if prm.viscosity == "power" else 0.0
    return TP


@njit(cache=True)
def viscosity(T, mode):
    if mode == 1.0:
        return MU_REF * (T / T_REF) ** OMEGA
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


@njit(cache=True)
def stagnation_heating(z, v, R_n, TP):
    """(T_stag [K], G [kg/(m^2 s)]) at height z [m], speed v [m/s]."""
    k, R, Ta = TP[TP_GAMMA], TP[TP_R], TP[TP_TATM]
    c_p = k * R / (k - 1.0)
    rho_inf = TP[TP_RHO0] * math.exp(-z / TP[TP_H])
    p_inf = rho_inf * R * Ta
    T_stag = Ta + v * v / (2.0 * c_p)
    dp = cp_max(v / TP[TP_VS], k) * 0.5 * rho_inf * v * v
    rho_e = (p_inf + dp) / (R * T_stag)
    mu_e = viscosity(T_stag, TP[TP_VISC])
    G = SIM * PR ** -0.6 * math.sqrt(rho_e * mu_e / R_n) * (2.0 * dp / rho_e) ** 0.25
    return T_stag, G


@njit(cache=True)
def wall_temperature(t, z, v, rho_w, c_w, delta, R_n, T_init, TP):
    """Skin temperature T_w(t_i) [K] for forward-ordered samples t_i, z_i, v_i."""
    k, R = TP[TP_GAMMA], TP[TP_R]
    c_p = k * R / (k - 1.0)
    C_w = rho_w * c_w * delta
    n = t.shape[0]
    T = np.empty(n)
    T[0] = T_init
    for i in range(n - 1):
        Ts, G = stagnation_heating(0.5 * (z[i] + z[i + 1]), 0.5 * (v[i] + v[i + 1]), R_n, TP)
        tau = C_w / max(G * c_p, 1e-300)
        T[i + 1] = Ts + (T[i] - Ts) * math.exp(-(t[i + 1] - t[i]) / tau)
    return T


def forward_samples(traj):
    """(t, z, v) in forward time from the backward trajectory array traj[n, NCOL]."""
    f = traj[::-1]
    v = np.hypot(f[:, 4], f[:, 5])
    return np.ascontiguousarray(f[:, 0]), np.ascontiguousarray(f[:, 2]), v


def site_temperatures(traj, prm, TP=None):
    """For every heating site in prm.sites: dict with T_w history (forward time),
    T_max, T_melt, margin = T_melt - margin_req - T_max (negative -> melts)."""
    TP = pack_thermal(prm) if TP is None else TP
    t, z, v = forward_samples(traj)
    out = {}
    for name, (mat, delta, R_n, active) in prm.sites.items():
        rho_w, c_w, T_melt = MATERIALS[mat]
        T = wall_temperature(t, z, v, rho_w, c_w, delta, R_n, prm.T_init, TP)
        T_max = float(T.max())
        out[name] = dict(material=mat, delta=delta, R_n=R_n, active=active, T=T, t=t,
                         T_max=T_max, T_melt=T_melt,
                         margin=T_melt - prm.melt_margin - T_max)
    return out


def melt_excess(sites):
    """sum over active sites of max(0, T_max - (T_melt - margin))^2 [K^2]."""
    return sum(max(0.0, -s["margin"]) ** 2 for s in sites.values() if s["active"])
