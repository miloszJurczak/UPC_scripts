import numpy as np
import matplotlib.pyplot as plt

class RocketParameters:
    def __init__(self):
        self.v_sep = 6000 / 3.6
        self.h_sep = 70000.0
        self.dry_mass = 130000.0
        self.exhaust_velocity = 3000.0
        self.mixture_ratio = 3.6
        self.density_lox = 1295.0
        self.density_ch4 = 449.0
        self.tank_diameter = 7.0
        self.tank_height = 61.0
        self.max_thrust = 66 * 10**6

params = RocketParameters()

def gravity(h):
    return 9.81 * (6400000 / (6400000 + h)) ** 2

def calc_fuel_mass():
    volume = (1 / 4) * (np.pi * params.tank_diameter ** 2) * params.tank_height
    return volume * (params.mixture_ratio + 1) / (
            (1 / params.density_ch4) + (params.mixture_ratio / params.density_lox))

initial_fuel_mass = calc_fuel_mass()
initial_total_mass = params.dry_mass + initial_fuel_mass
mass_flow_rate = params.max_thrust / params.exhaust_velocity

pitch_v = 20.0
pitch_angle = np.radians(1.5)
dt = 0.01
min_throttle = 0.20

def simulate_flight(mass_ratio, full_thrust_limit):
    current_mass = initial_total_mass * (1.0 + mass_ratio)
    v, beta, h = 0.0, 0.0, 0.0
    pitch_done = False
    burned_fuel = 0.0

    while True:
        g_current = gravity(h)

        if burned_fuel >= initial_fuel_mass:
            return None

        if burned_fuel < full_thrust_limit:
            current_thrust = params.max_thrust
        else:
            current_thrust = min_throttle * params.max_thrust

        mass_drop = (current_thrust / params.exhaust_velocity) * dt
        if burned_fuel + mass_drop > initial_fuel_mass:
            return None

        thrust_to_weight = current_thrust / (current_mass * g_current)

        if not pitch_done:
            v += g_current * (thrust_to_weight - 1.0) * dt
            if v >= pitch_v:
                beta = pitch_angle
                pitch_done = True
        else:
            dv = g_current * (thrust_to_weight - np.cos(beta)) * dt
            dbeta = (g_current / v) * np.sin(beta) * dt
            v += dv
            beta += dbeta

        h += v * np.cos(beta) * dt
        current_mass -= mass_drop
        burned_fuel += mass_drop

        if v <= 0 or beta >= np.pi / 2:
            return None
        if h >= params.h_sep:
            return v, np.degrees(beta), burned_fuel

def find_optimal_throttle(mass_ratio):
    if params.max_thrust <= initial_total_mass * (1 + mass_ratio) * gravity(0.0) * 1.05:
        return np.nan, np.nan, np.nan

    high = initial_fuel_mass
    test_flight = simulate_flight(mass_ratio, high)
    
    if test_flight is None:
        for test_limit in np.linspace(initial_fuel_mass * 0.98, initial_fuel_mass * 0.6, 25):
            result = simulate_flight(mass_ratio, test_limit)
            if result is not None and result[0] >= params.v_sep:
                high = test_limit
                test_flight = result
                break

    if test_flight is None or test_flight[0] < params.v_sep:
        return np.nan, np.nan, np.nan

    low = 0.0
    for _ in range(35):
        mid = 0.5 * (low + high)
        result = simulate_flight(mass_ratio, mid)
        
        if result is None or result[0] < params.v_sep:     
            low = mid
        else:
            high = mid

    final_result = simulate_flight(mass_ratio, high)
    if final_result is None:
        return np.nan, np.nan, np.nan
        
    v, beta_deg, total_burned = final_result
    return beta_deg, 90.0 - beta_deg, total_burned / 1000.0

if __name__ == "__main__":
    print(f"Initial fuel mass: {initial_fuel_mass / 1000:.2f} t")
    print(f"First stage wet mass: {initial_total_mass / 1000:.2f} t")
    print(f"Mass flow rate (max): {mass_flow_rate:.1f} kg/s\n")

    ratios = np.linspace(0.05, 1.0, 60)
    angles = []

    for r in ratios:
        _, alpha, _ = find_optimal_throttle(r)
        angles.append(alpha)

    plt.figure(figsize=(6, 4))
    plt.plot(ratios, angles, label=r'Angles in', color='tab:blue', lw=2)
    plt.xlabel('Mass ratio ($m_2 / m_1$)')
    plt.ylabel(r'Flight Path Angle $\alpha$ [°]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('gravity_turn.png')
    plt.show()