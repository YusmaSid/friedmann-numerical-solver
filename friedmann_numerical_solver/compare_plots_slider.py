import numpy as np
import matplotlib. pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider # for varying parameters

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

init_omegas = {"omega_m": 0.3, "omega_r": 0.0, "omega_lambda": 0.7}
solution = solve_ivp(lambda t, a: friedmann(t, a, **init_omegas), t_span, a0, dense_output=True)
a_vals = solution.sol(t_eval)[0]

# plotting all figures on the same figure
fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(bottom=0.35)

line = ax.plot(t_eval, a_vals)
ax.set_xlabel("Time")
ax.set_ylabel("Scale factor a(t)")
ax.set_title("Cosmic Expansion For Different Cosmologies")

# create slider axes 
ax_m = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_r = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_lambda = plt.axes([0.2, 0.1, 0.65, 0.03])

# create slider
slider_m = Slider(ax_m, "omega_m_", 0, 1, valinit = 0.3)
slider_r = Slider(ax_r, "omega_r", 0, 1, valinit = 0.0)
slider_lambda = Slider(ax_lambda, "omega_lambda", 0, 1, valinit = 0.7)

# update simulation
def update(val):
    omegas = {
        "omega_m": slider_m.val,
        "omega_r": slider_r.val,
        "omega_lambda": slider_lambda.val
    }

    sol = solve_ivp(lambda t, a: friedmann(t, a, **omegas), t_span, a0, dense_output=True)
    a_vals = sol.sol(t_eval)[0]

    line.set_ydata(a_vals)
    fig.canvas.draw_idle()

slider_m.on_changed(update)
slider_r.on_changed(update)
slider_lambda.on_changed(update)

plt.show()