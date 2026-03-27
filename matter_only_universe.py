import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# simulating a matter only universe 

omega_m = 1.0           # matter
omega_r = 0.0           # radiation
omega_lambda = 0.0      # dark matter
H0 = 1.0                # hubble parameter

def friedmann(t, a):
    a_dot = a * H0 * np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return a_dot

t_span = (0.01, 5) # t = 0 would give a singularity
a0 = [0.001]

solution = solve_ivp(friedmann, t_span, a0, dense_output=True)

t = np.linspace(0.01, 5, 500)
a = solution.sol(t)[0]

plt.plot(t,a)
plt.xlabel("Time")
plt.ylabel("Scale factor a(t)")
plt.title("Cosmic Expansion in Matter Dominated Universe")
plt.show()
