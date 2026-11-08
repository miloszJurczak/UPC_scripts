"""Mass properties of the booster (njit). Port of M_w0, R_cm, I_moment from sim1.py.

Propellant column (fuel mass fraction 1/(k+1), oxidizer k/(k+1)) of area A = pi d^2/4:
    h_f  = Mw / (A rho_f (k+1))          height of the fuel column
    h_ox = Mw k / (A rho_ox (k+1))       height of the LOX column
    H_ox = h_ox(M_w0)                    full LOX tank height (fuel sits above it)
    M_w0 = A H_0 (k+1) / (1/rho_f + k/rho_ox)    capacity of the tank volume A H_0
"""
import math

from jit import njit
from params import D, H_0, H_S, K_OF, M_S, M_TANK, RHO_F, RHO_OX


@njit(cache=True)
def tank_capacity(P):
    return 0.25 * math.pi * P[D] ** 2 * P[H_0] * (P[K_OF] + 1.0) / (1.0 / P[RHO_F] + P[K_OF] / P[RHO_OX])


@njit(cache=True)
def _columns(Mw, P):
    A = 0.25 * math.pi * P[D] ** 2
    k = P[K_OF]
    h_f = Mw / (A * P[RHO_F] * (k + 1.0))
    h_ox = Mw * k / (A * P[RHO_OX] * (k + 1.0))
    H_ox = tank_capacity(P) * k / (A * P[RHO_OX] * (k + 1.0))
    m_ox = k / (k + 1.0) * Mw
    m_f = Mw / (k + 1.0)
    return h_f, h_ox, H_ox, m_ox, m_f


@njit(cache=True)
def r_cm(Mw, P):
    """Height of the centre of mass above the engine plane [m]."""
    h_f, h_ox, H_ox, m_ox, m_f = _columns(Mw, P)
    num = (P[M_S] * P[H_S] / 2.0 + P[M_TANK] * (P[H_0] + P[H_S]) / 2.0
           + m_ox * (P[H_S] + h_ox / 2.0) + m_f * (P[H_S] + H_ox + h_f / 2.0))
    return num / (P[M_S] + m_ox + m_f + P[M_TANK])


@njit(cache=True)
def _I_cyl(m, h, r, d):
    # solid cylinder about a transverse axis through the CM, shifted by r (Steiner)
    return m * h * h / 12.0 + m * d * d / 16.0 + m * r * r


@njit(cache=True)
def inertia(Mw, P):
    """Pitch moment of inertia [kg m^2]; ok = False if Mw exceeds tank capacity."""
    if Mw > tank_capacity(P):
        return 0.0, False
    h_f, h_ox, H_ox, m_ox, m_f = _columns(Mw, P)
    rc = r_cm(Mw, P)
    d, Hs, H0, mt = P[D], P[H_S], P[H_0], P[M_TANK]
    I_tube = mt * (Hs + H0) ** 2 / 12.0 + 0.5 * mt * (d / 2.0) ** 2 + mt * ((Hs + H0) / 2.0 - rc) ** 2
    I = (_I_cyl(P[M_S], Hs, rc - Hs / 2.0, d)
         + _I_cyl(m_ox, h_ox, rc - Hs - h_ox / 2.0, d)
         + _I_cyl(m_f, h_f, rc - Hs - H_ox - h_f / 2.0, d) + I_tube)
    return I, True
