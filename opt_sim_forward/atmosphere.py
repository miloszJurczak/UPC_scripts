"""Atmosphere and gas-dynamics kernels (njit, scalar).

    g(h)      = g0 (R_E / (R_E + h))^2                     inverse-square gravity
    rho(z)    = rho0 exp(-z / H)                            isothermal atmosphere
    p_inf(z)  = rho(z) R T_atm                              ideal gas
    v_s       = sqrt(gamma R T_atm)                         speed of sound
    cp_max(M) = stagnation-point pressure coefficient
                M <= 1 : isentropic   2/(k M^2) [(1 + (k-1)/2 M^2)^(k/(k-1)) - 1]
                M  > 1 : Rayleigh pitot (normal shock + isentropic recompression)
"""
import math

from jit import njit
from params import G0, R_EARTH, RHO0, H_ATM, T_ATM, R_GAS, GAMMA


@njit(cache=True)
def gravity(h, P):
    r = P[R_EARTH] / (P[R_EARTH] + h)
    return P[G0] * r * r


@njit(cache=True)
def density(z, rho0, H):
    return rho0 * math.exp(-z / H)


@njit(cache=True)
def density_dyn(z, P):
    return density(z, P[RHO0], P[H_ATM])


@njit(cache=True)
def sound_speed(P):
    return math.sqrt(P[GAMMA] * P[R_GAS] * P[T_ATM])


@njit(cache=True)
def cp_max(M, k):
    """Stagnation Cp. The M -> 0 limit is 1 (incompressible Bernoulli);
    the original vectorised version returned 0/0 = NaN there."""
    if M < 1e-6:
        return 1.0
    if M <= 1.0:
        return 2.0 / (k * M * M) * ((1.0 + 0.5 * (k - 1.0) * M * M) ** (k / (k - 1.0)) - 1.0)
    M2 = M * M
    pitot = ((k + 1.0) ** 2 * M2 / (4.0 * k * M2 - 2.0 * (k - 1.0))) ** (k / (k - 1.0)) \
        * (1.0 - k + 2.0 * k * M2) / (k + 1.0)
    return 2.0 / (k * M2) * (pitot - 1.0)
