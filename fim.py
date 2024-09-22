import model 
import numpy as np
import sympy as sp

# times at which I evaluate
t = np.array([0.5, 1.0, 2.0])

# standard deviations
sigma_a = 0.1
sigma_o = 0.1

# model predictions
T_a = model.T_a
T_o = model.T_o


x = sp.Symbol('x')
derivative_Ta = sp.diff(T_a, x)
# efficiency = 1 fixed (so i have 5 parameters)