import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define your model formula
def model_function(x, a, b, c):
    return a * x**2 + b * x + c

# Initial parameters
initial_params = {'a': 1, 'b': 0, 'c': 0}

# Generate initial data points
x_data = np.linspace(-10, 10, 100)
y_data = model_function(x_data, **initial_params)# keywords arguments

# Create the initial plot
fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data, label='Curve')

# Add sliders for parameter adjustment
axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
ax_b = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor=axcolor)
ax_c = plt.axes([0.1, 0.11, 0.65, 0.03], facecolor=axcolor)

slider_a = Slider(ax_a, 'A', -5, 5, valinit=initial_params['a'])
slider_b = Slider(ax_b, 'B', -5, 5, valinit=initial_params['b'])
slider_c = Slider(ax_c, 'C', -5, 5, valinit=initial_params['c'])

# Function to update the plot when sliders are changed
def update(val):
    a = slider_a.val
    b = slider_b.val
    c = slider_c.val
    y_data = model_function(x_data, a, b, c)
    line.set_ydata(y_data)
    fig.canvas.draw_idle()

# Connect the sliders to the update function
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)

# Add a reset button to reset parameters
resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
button_reset = plt.Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_a.reset()
    slider_b.reset()
    slider_c.reset()

button_reset.on_clicked(reset)

plt.show()