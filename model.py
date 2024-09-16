from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Six parameters initial values
# Using typical values calibrated for the CMIP5 models mentioned in the article​
# https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

# C = 8           # Upper layer heat capacity
# C0 = 100        # Lower layer heat capacity
# gamma = 0.7     # Heat exchange coefficient
# epsilon = 1     # Efficacy factor
# lambda_ = 1.3   # Climate feedback parameter
# F = 3.9         # External forcing for a 4x CO2 increase

# I make some assumptions to simplify the model:
# 1) efficiency = 1 fixed (so one less parameter)
# 2) Forcing is a STEP FORCING
# 3) T(0)=0 as initial conditions
# under these assumptions an analytical solution exists 


def define_solution_parameters(C, C0, λ, γ):
    # General parameters
    b = (λ + γ) / C + γ / C0
    b_star = (λ + γ) / C - γ / C0
    delta = b*b - 4 * λ * γ / (C * C0)

    # Mode parameters (Fast and Slow)
    # Fast mode
    τ_f = C * C0 * (b - np.sqrt(delta)) / (2 * γ * λ)
    φ_f = C / (2 * γ) * (b_star - np.sqrt(delta))
    a_f = φ_s * τ_f * λ / (C * (φ_s - φ_f))
    l_f = a_f * τ_f * λ / (C + C0)

    # Slow mode
    τ_s = C * C0 * (b + np.sqrt(delta)) / (2 * γ * λ)
    φ_s = C / (2 * γ) * (b_star + np.sqrt(delta))
    a_s = -φ_f * τ_s * λ / (C * (φ_s - φ_f))
    l_s = a_s * τ_s * λ / (C + C0)

    # Return all parameters in a dictionary
    return {
        'b': b,
        'b_star': b_star,
        'delta': delta,
        'τ_f': τ_f,
        'φ_f': φ_f,
        'a_f': a_f,
        'l_f': l_f,
        'τ_s': τ_s,
        'φ_s': φ_s,
        'a_s': a_s,
        'l_s': l_s
    }


def T_a(t, F, C, C0, λ, γ, dict):
    return F/λ(dict['a_f']*(1-np.exp(-t/dict['τ_f'])) + dict['a_s']*(1-np.exp(-t/dict['τ_s'])))


def T_o(t, F, C, C0, λ, γ, dict):
    return F/λ(dict['a_f']*dict['φ_f']*(1-np.exp(-t/dict['τ_f'])) + dict['φ_s']*dict['a_s']*(1-np.exp(-t/dict['τ_s'])))
