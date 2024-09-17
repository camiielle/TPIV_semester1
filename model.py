from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Six parameters initial values
# Using typical values calibrated for the CMIP5 models mentioned in the article​
# https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

# C = 8           # Upper layer heat capacity
# C0 = 100        # Lower layer heat capacity
# lambda_ = 1.3   # Climate feedback parameter
# gamma = 0.7     # Heat exchange coefficient
# epsilon = 1     # Efficacy factor
# F = 3.9         # External forcing for a 4x CO2 increase

# I make some assumptions to simplify the model:
# 1) efficiency = 1 fixed (so one less parameter)
# 2) Forcing is a STEP FORCING
# 3) T(0)=0 as initial conditions
# under these assumptions an analytical solution exists

C, C0, λ, γ, F, t = sp.symbols('C C0 λ γ F t')

# General parameters
b = (λ + γ) / C + γ / C0
b_star = (λ + γ) / C - γ / C0
delta = b*b - 4 * λ * γ / (C * C0)

# Mode parameters (Fast and Slow)
τ_f = C * C0 * (b - sp.sqrt(delta)) / (2 * γ * λ)
τ_s = C * C0 * (b + sp.sqrt(delta)) / (2 * γ * λ)

φ_s = C / (2 * γ) * (b_star + sp.sqrt(delta))
φ_f = C / (2 * γ) * (b_star - sp.sqrt(delta))

a_f = φ_s * τ_f * λ / (C * (φ_s - φ_f))
l_f = a_f * τ_f * λ / (C + C0)

a_s = -φ_f * τ_s * λ / (C * (φ_s - φ_f))
l_s = a_s * τ_s * λ / (C + C0)

# times at which I evaluate
# t = np.array([0.5, 1.0, 2.0])

# standard deviations
sigma_a = 0.1
sigma_o = 0.1

# analytically calculating derivatives
Ta = F/λ*(a_f*(1-sp.exp(-t/τ_f)) + a_s*(1-sp.exp(-t/τ_s)))
dTa_dC = sp.diff(Ta, C)
dTa_dC_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTa_dC, 'numpy')
dTa_dC0 = 0
dTa_dλ = sp.diff(Ta, λ)
dTa_dλ_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTa_dλ, 'numpy')
dTa_dγ = sp.diff(Ta, γ)
dTa_dγ_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTa_dγ, 'numpy')
dTa_dF = 1/λ*(a_f*(1-sp.exp(-t/τ_f)) + a_s*(1-sp.exp(-t/τ_s)))
dTa_dF_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTa_dF, 'numpy')

To = F/λ*(a_f*φ_f*(1-sp.exp(-t/τ_f)) + φ_s*a_s*(1-sp.exp(-t/τ_s)))
dTo_dC = 0
dTo_dC0 = sp.diff(To, C0)
dTo_dC0_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTo_dC0, 'numpy')
dTo_dλ = sp.diff(To, λ)
dTo_dλ_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTo_dλ, 'numpy')
dTo_dγ = sp.diff(To, γ)
dTo_dγ_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTo_dγ, 'numpy')
dTo_dF = 1/λ*(a_f*φ_f*(1-sp.exp(-t/τ_f)) + φ_s*a_s*(1-sp.exp(-t/τ_s)))
dTo_dF_numpy = sp.lambdify([C, C0, λ, γ, F, t], dTo_dF, 'numpy')

# calculating the FIM
time = np.array([0.5, 1.0, 2.0])
g11 = np.sum(1/(sigma_a**2)*(dTa_dC_numpy(8, 100, 1.3, 0.7, 3.9, time)**2))

print('all good')

# def T_a(F, λ, dict):
# return F/λ*(dict['a_f']*(1-np.exp(-t/dict['τ_f'])) + dict['a_s']*(1-np.exp(-t/dict['τ_s'])))
# def T_o(F, λ, dict):
# return F/λ*(dict['a_f']*dict['φ_f']*(1-np.exp(-t/dict['τ_f'])) + dict['φ_s']*dict['a_s']*(1-np.exp(-t/dict['τ_s'])))

# def dTa_dC(F, C, C0, λ, γ, dict):
# Extract the values from the dictionary
# a_f = dict['a_f']
# a_s = dict['a_s']
# t_f = dict['t_f']
# t_s = dict['t_s']

# Placeholder derivatives (Replace these with actual derivatives)

# da_f_dC = dict['φ_s'] * ((λ / (dict['φ_s'] - dict['φ_f'])) *
# dt_f_dC - (dict['τ_f'] * λ) / (C**2 * (dict['φ_s'] - dict['φ_f'])))
# da_s_dC = dict['φ_f'] * ((λ / (dict['φ_s'] - dict['φ_f'])) *
#       dt_s_dC - (dict['τ_s'] * λ) / (C**2 * (dict['φ_s'] - dict['φ_f'])))
# dt_f_dC = (C0 * (dict['b'] - sp.sqrt(dict['delta']))) / (2 * γ * λ) + \
#  (C * C0 * ((-λ - γ) / C0 / C**2 - (-2 * γ * λ / C0 / C**2) /
# (2 * sp.sqrt(dict['delta'])))) / (2 * γ * λ)
# dt_s_dC = (C0 * (dict['b'] + sp.sqrt(dict['delta']))) / (2 * γ * λ) + \
# (C * C0 * ((- λ - γ) / C0 / C**2 + (-2 * γ * λ / C0 / C**2) /
#  (2 * sp.sqrt(dict['delta'])))) / (2 * γ * λ)

# First term: a_f * (1 - exp(-t / t_f))
# term1 = da_f_dC * (1 - np.exp(-t / t_f)) + a_f * \
# np.exp(-t / t_f) * (t / (t_f ** 2)) * dt_f_dC

# Second term: a_s * (1 - exp(-t / t_s))
# term2 = da_s_dC * (1 - np.exp(-t / t_s)) + a_s * \
# np.exp(-t / t_s) * (t / (t_s ** 2)) * dt_s_dC

# Full derivative expression
# dT_a_dC = (F / λ) * (term1 + term2)

# return dT_a_dC
# def dTo_dC(F, C, C0, λ, γ, dict):

# Derivatives of τ_f and τ_s with respect to C0
# dτ_f_dC0 = (C * (dict['b'] - sp.sqrt(dict['delta'])) / (2 * λ)) + \
#  (C0 * (-(C * (dict['b'] - sp.sqrt(dict['delta']))) / (2 * λ * C0)))
# dτ_s_dC0 = (C * (dict['b'] + sp.sqrt(dict['delta'])) / (2 * λ)) + \
#   (C0 * (-(C * (dict['b'] + sp.sqrt(dict['delta']))) / (2 * λ * C0)))

# Derivatives of Φ_f and Φ_s with respect to C0
# dΦ_f_dC0 = (C / (2 * λ)) * ((-C / C0**2) * (dict['b_star'] + sp.sqrt(dict['delta'])) - (
# C * dict['b_star'] + sp.sqrt(dict['delta'])) * (-(dict['b_star'] + sp.sqrt(dict['delta']))))
# dΦ_s_dC0 = (C / (2 * λ)) * (-(dict['b_star'] - sp.sqrt(dict['delta'])))

# Derivative of a_f with respect to C0
# da_f_dC0 = (1 / (C + C0)) * (dΦ_f_dC0 - dΦ_s_dC0) - \
# (dict['φ_f'] - dict['φ_s']) / (C + C0)**2

# Derivative of a_s with respect to C0
# da_s_dC0 = -(1 / (C + C0)) * (dΦ_f_dC0 - dΦ_s_dC0) + \
# (dict['φ_f'] - dict['φ_s']) / (C + C0)**2

# Derivative of the exponential terms
# exp_f = np.exp(-t / dict['τ_f'])
# exp_s = np.exp(-t / dict['τ_s'])

# d_exp_f_dC0 = (t / dict['τ_f']**2) * exp_f * dτ_f_dC0
# d_exp_s_dC0 = (t / dict['τ_f']**2) * exp_s * dτ_s_dC0

# Chain rule application for derivative of T_o with respect to C0
# dT_o_f_dC0 = (da_f_dC0 * dict['φ_f'] * (1 - exp_f)
# + dict['a_f'] * dΦ_f_dC0 * (1 - exp_f)
# + dict['a_f'] * dict['φ_f'] * (-d_exp_f_dC0))

# dT_o_s_dC0 = (da_s_dC0 * dict['φ_s'] * (1 - exp_s)
# + dict['a_s'] * dΦ_s_dC0 * (1 - exp_s)
# + dict['a_s'] * dict['φ_s'] * (-d_exp_s_dC0))

# Final derivative expression
# dT_o_dC0 = (F / λ) * (dT_o_f_dC0 + dT_o_s_dC0)

# return dT_o_dC0
