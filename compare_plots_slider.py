import numpy as np
import matplotlib. pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider, Button # for varying parameters and button for swtiching between preset cosmologies

# comparing behavior of expansion in certain dominations and calculated age

H0 = 1.0    # hubble parameter

# the graph line
def friedmann(t, a, omega_m, omega_r, omega_lambda):
    a_dot = a * H0 * np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return a_dot

# different values for parameters based on the type of universe
cosmologies ={
    "Matter Dominated": {"omega_m": 1.0, "omega_r": 0.0, "omega_lambda": 0.0},
    "Radiation Dominated": {"omega_m": 0.0, "omega_r": 1.0, "omega_lambda": 0.0},
    "Dark Energy Dominated": {"omega_m": 0.0, "omega_r": 0.0, "omega_lambda": 1.0},
}
cosmologies["Lambda-CDM"] = {"omega_m": 0.3, "omega_r": 0.0, "omega_lambda": 0.7} # preset for realistic cosmology with planck values

# general simulation parameters
t_span = (0.01, 9) # t = 0 would give a singularity
a0 = [0.001]
t_eval = np.linspace(t_span[0], t_span[1], 500)

init_omegas = {"omega_m": 0.3, "omega_r": 0.0, "omega_lambda": 0.7} # planck values 
solution = solve_ivp(lambda t, a: friedmann(t, a, **init_omegas), t_span, a0, dense_output=True)
a_vals = solution.sol(t_eval)[0]

# plotting all figures on the same figure
fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(bottom=0.35)
age_text = ax.text(0.05, 0.9, "", transform=ax.transAxes) #creating the text for the age counter

# creating the actual line
line, = ax.plot(t_eval, a_vals)
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

# create button for preset cosomologies
ax_btn_matter = plt.axes([0.025, 0.025, 0.2, 0.05])
ax_btn_radiation = plt.axes([0.275, 0.025, 0.2, 0.05])
ax_btn_lambda = plt.axes([0.525, 0.025, 0.2, 0.05])
ax_btn_lCDM = plt.axes([0.775, 0.025, 0.2, 0.05])

btn_matter = Button(ax_btn_matter, 'Matter')
btn_radiation = Button(ax_btn_radiation, 'Radiation')
btn_lambda = Button(ax_btn_lambda, 'Dark Energy')
btn_lCDM = Button(ax_btn_lCDM, 'ΛCDM')

# applying presets
def set_cosmology(name): 
    params = cosmologies[name]

    slider_m.set_val(params['omega_m'])
    slider_r.set_val(params['omega_r'])
    slider_lambda.set_val(params['omega_lambda'])

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
    
    # compute age of universe
    # accepted val ~ 13.8 B years = 113.8 Gyr

    age_index = np.argmin(np.abs(a_vals-1))
    age = t_eval[age_index]
    age_gyr = age*14    # age in Gigayears

    age_text.set_text(f"Cosmic Expansion (Age ≈ {age_gyr:.2f} Gyr)") 
    fig.canvas.draw_idle()

# triggering the update
slider_m.on_changed(update)
slider_r.on_changed(update)
slider_lambda.on_changed(update)

# triggering the update for preset cosmologies
btn_matter.on_clicked(lambda events: set_cosmology("Matter Dominated"))
btn_radiation.on_clicked(lambda event: set_cosmology("Radiation Dominated"))
btn_lambda.on_clicked(lambda event: set_cosmology("Dark Energy Dominated"))
btn_lCDM.on_clicked(lambda event: set_cosmology("Lambda-CDM"))

plt.show()