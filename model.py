import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mbam.geodesic import Geodesic
from mbam.finite_difference import Avv_func, AvvCD4

# Six parameters initial values
# Using typical values calibrated for the CMIP5 models mentioned in the article​
# https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

# C = 8           # Upper layer heat capacity
# C0 = 100        # Lower layer heat capacity
# lambda_ = 1.3/8
# gamma = 0.7/8
# gamma0 = 0.7/100
# epsilon = 1      # Efficacy factor
# F = 3.9/8        # External forcing for a 4x CO2 increase divided by C

# I make some assumptions to simplify the model:
# 1) efficacy = 1 fixed (so one less parameter)
# 2) Forcing is a STEP FORCING
# 3) Tx(0)=0 as initial conditions for x=a,o
# under these assumptions an analytical solution exists and takes a simple form

pars = sp.symbols(['λ', 'γ', 'γ0', 'F', 't'])

# Unpack the list into individual variables
λ, γ, γ0, F, t = pars

# General parameters
b = λ + γ + γ0
b_star = λ + γ - γ0
delta = b*b - 4 * λ * γ0

# Mode parameters (Fast and Slow)
τ_f = (b - sp.sqrt(delta)) / (2 * γ0 * λ)
τ_s = (b + sp.sqrt(delta)) / (2 * γ0 * λ)

φ_s = 1 / (2 * γ) * (b_star + sp.sqrt(delta))
φ_f = 1 / (2 * γ) * (b_star - sp.sqrt(delta))

a_f = φ_s * τ_f * λ / (φ_s - φ_f)
l_f = a_f * τ_f * λ / (1 + γ/γ0)

a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
l_s = a_s * τ_s * λ / (1 + γ/γ0)

# standard deviations
sigma_a = 0.1
sigma_o = 0.1

# defining ODEs solutions
Ta = F/λ*(a_f*(1-sp.exp(-t/τ_f)) + a_s*(1-sp.exp(-t/τ_s)))
To = F/λ*(a_f*φ_f*(1-sp.exp(-t/τ_f)) + φ_s*a_s*(1-sp.exp(-t/τ_s)))


# Calculate the partial derivatives of the two functions
der_a = []
der_o = []

for p in pars[:-1]:
    der_a.append(sp.lambdify(pars, sp.diff(Ta, p), 'numpy'))
    der_o.append(sp.lambdify(pars, sp.diff(To, p), 'numpy'))

# times at which I evaluate
time = np.array([0.5, 1.0, 2.0])

# evaluating the derivatives at specific parameter values
der_a_v = []
der_o_v = []

for der in der_a:
    der_a_v.append(der(1.3/8, 0.7/8, 0.007, 3.9/8, time))
    # each element in der_a_v is an array of three values

for der in der_o:
    der_o_v.append(der(1.3/8, 0.7/8, 0.007, 3.9/8, time))
    # each element in der_o_v is an array of three values

# calculating the FIM
g = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        g[i, j] = np.sum(1/(sigma_a**2)*der_a_v[i]*der_a_v[j] +
                         1/(sigma_o**2)*der_o_v[i]*der_o_v[j])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(g)

# Find the index of the minimum eigenvalue
min_index = np.argmin(eigenvalues)

# Get the minimum eigenvalue and its corresponding eigenvector
min_eigenvalue = eigenvalues[min_index]
min_eigenvector = eigenvectors[:, min_index]

# x is a list containing the chosen initial values of the parameters


def r(x):
    Ta_v = sp.lambdify(pars, Ta, 'numpy')
    To_v = sp.lambdify(pars, To, 'numpy')
    return np.array(np.concatenate(([1/sigma_a*Ta_v(*x, time), 
    1/sigma_o*To_v(*x, time)])))


def jacob(x):
    result = np.concatenate(((1/sigma_a)*np.array(der_a_v),
                            (1/sigma_o)*np.array(der_o_v)), axis=1)
    return np.array([result[i] for i in range(4)]).T


x = np.array([1.3/8, 0.7/8, 0.007, 3.9/8])
v = np.array(min_eigenvector)
# Callback function used to monitor the geodesic after each step


def callback(gi):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnostic along the way
    print(
        "Iteration: %i, tau: %f, |v| = %f"
        % (len(gi.vs), gi.ts[-1], np.linalg.norm(gi.vs[-1]))
    )
    return np.linalg.norm(gi.vs[-1]) < 100.0


Avv = Avv_func(r, AvvCD4, h=1e-2)
# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo_forward = Geodesic(r, jacob, Avv, x, v, atol=1e-2,
                       rtol=1e-2, callback=callback)

# Integrate
geo_forward.integrate(25.0)

# Plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
plt.figure()
plt.plot(geo_forward.ts, geo_forward.xs)
plt.xlabel("tau")
plt.ylabel("Parameter Values")
plt.show()
