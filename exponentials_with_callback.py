
from mbam import Geodesic, initial_velocity
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svdvals

exp = np.exp

t = np.array([0.5, 1.0, 2.0])


# Model predictions
def r(x):
    return exp(-exp(x[0]) * t) + exp(-exp(x[1]) * t)


# Jacobian
def j(x):
    return np.array(
        [
            -t * exp(x[0]) * exp(-exp(x[0]) * t),
            -t * exp(x[1]) * exp(-exp(x[1]) * t),
        ]
    ).T


# Directional second derivative
def Avv(x, v):
    h = 1e-4
    return (r(x + h * v) + r(x - h * v) - 2 * r(x)) / h / h


# Choose starting parameters
x = np.log([1.0, 2.0])
v = initial_velocity(x, j, Avv)


# Callback function used to monitor the geodesic after each step
def callback(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    fim = j(g.xs[-1]).T @ j(g.xs[-1])
    eigenvalues, eigenvectors = np.linalg.eig(fim)
    # Find the index of the minimum eigenvalue
    min_index = np.argmin(eigenvalues)
    # Get the minimum eigenvalue and its corresponding eigenvector
    min_eigenvalue = eigenvalues[min_index]
    min_eigenvector = eigenvectors[:, min_index]
    print(
        "Iteration: %i, tau: %f, eigenvalue: %.10e, min_eigenvector: %s, vs[-1]: %s"
        % (len(g.vs), g.ts[-1], min_eigenvalue, min_eigenvector, g.vs[-1])
    )
    #svdvals(fim)[-1]

    return min_eigenvalue > 1e-10


# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2, callback=callback)

# Integrate
geo.integrate(25.0)

# Plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
plt.figure()
plt.plot(geo.ts, geo.xs)
plt.xlabel("tau")
plt.ylabel("Parameter Values")
plt.show()
