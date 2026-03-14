import numpy as np
import matplotlib. pyplot as plt
from scipy.integrate import solve_ivp

# comparing behavior of expansion in certain dominations

H0 = 1.0                # hubble parameter

def friedmann(t, a, omega_m, omega_r, omega_lambda):
    a_dot = a * H0 * np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return a_dot

cosmologies ={
    "Matter Dominated": {"omega_m": 1.0, "omega_r": 0.0, "omega_lambda": 0.0},
    "Radiation Dominated": {"omega_m": 0.0, "omega_r": 1.0, "omega_lambda": 0.0},
    "Dark Matter Dominated": {"omega_m": 0.0, "omega_r": 0.0, "omega_lambda": 1.0}
}

# simulation parameters
t_span = (0.01, 9) # t = 0 would give a singularity
a0 = [0.001]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# plotting all figures on the same figure
plt.figure(figsize=(8, 5))
for label, omegas in cosmologies.items():
    solution = solve_ivp(lambda t, a: friedmann(t, a, **omegas), t_span, a0, dense_output=True)
    a_vals = solution.sol(t_eval)[0]
    plt.plot(t_eval, a_vals, label=label)

plt.xlabel("Time")
plt.ylabel("Scale factor a(t)")
plt.title("Cosmic Expansion For Different Cosmologies")
plt.legend()
plt.show()