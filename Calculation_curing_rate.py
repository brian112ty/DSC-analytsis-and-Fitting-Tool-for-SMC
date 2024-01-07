import sys
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
from PIL import Image
from numpy.random import rand
import mplcursors
import math

class FileBrowserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Please select the DSC data files for the first heating rate")

        self.file_path_label = tk.Label(root, text="Selected Files:")
        self.file_path_label.pack(pady=10)

        self.groups = {}  # Dictionary to store file paths in groups
        self.file_count = 0

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=10)

        self.add_data_button = tk.Button(root, text="Add Data for Another Heating Rate", command=self.add_data)
        self.add_data_button.pack(pady=10)

    def browse_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_paths:
            group_key = f"Group_{self.file_count + 1}"
            if group_key not in self.groups:
                self.groups[group_key] = []
            for i, file_path in enumerate(file_paths):
                file_name = f"{group_key}_{i + 1}"
                self.groups[group_key].append((file_name, file_path))
            self.update_file_path_label()
            self.file_count += 1

    def update_file_path_label(self):
        text = "Selected Files:\n"
        for group_key, file_paths in self.groups.items():
            for file_name, file_path in file_paths:
                text += f"{file_name}: {file_path}\n"
        self.file_path_label.config(text=text)

    def add_data(self):
        self.root.title(f"Please select the DSC data files for the heating rate {len(self.groups) + 1}")
        self.browse_file()





def kamal_equation(alpha, A1, E1, A2, E2, m, n, T):
    dalpha_dt = (A1 * np.exp(-E1 / T) + A2 * np.exp(-E2 / T) * (alpha ** m)) * ((1 - alpha) ** n)
    return dalpha_dt

# ********************* Step1: Input the data *********************
# Load the data (pandas data frame, columns with float64 data)
# Sample data YPE
def input_file():
    return 1

data = pd.read_csv('/Users/yuanpeien/Desktop/Internship_SMC/Python_code/Data_test.csv', skiprows=1)
temperature = data['/∞C']
heat_flow = data['/mW']
time = data['/s']
temperature = np.array(temperature)
temperature = list(reversed(temperature))
heat_flow = np.array(heat_flow)
time = np.array(time)

# 3D Timon Fitting tool data
data2 = pd.read_csv('/Users/yuanpeien/Desktop/Internship_SMC/Testdata.csv', skiprows=1)
temperature2 = data2['/∞C']
curing_deg2 = data2['/mW']
time2 = data2['/s']
temperature2 = np.array(temperature2)
time2 =  np.array(time2)
curing_deg2 = np.array(curing_deg2)

data_ = np.genfromtxt('/Users/yuanpeien/Desktop/Internship_SMC/Python_code/8_120_5h.txt', encoding='utf-16')
time_ = data_[:,0]
time_ = time_ * 60
temperature_ = data_[:,1]
heat_flow_ = data_[:,2]



# Plot heat flow over temperature
plt.plot(temperature, heat_flow, color ='red', label='Heat Flow')
selected_points = plt.ginput(2, timeout=0, show_clicks=True)
plt.close()  # Close the plot window after selection

# Extract x and y coordinates of the selected points
selected_x = [point[0] for point in selected_points]
selected_y = [point[1] for point in selected_points]

data_point1 = {'temperature': selected_x[0], 'heat_flow': selected_y[0]}
data_point2 = {'temperature': selected_x[1], 'heat_flow': selected_y[1]}

plt.plot(temperature, heat_flow, label='Heat Flow vs Temperature')
plt.scatter(data_point1['temperature'], data_point1['heat_flow'], color='red', label='Data Point 1')
plt.scatter(data_point2['temperature'], data_point2['heat_flow'], color='blue', label='Data Point 2')
plt.xlabel('Temperature')
plt.ylabel('Heat Flow')
plt.plot([data_point1['temperature'], data_point2['temperature']], [data_point1['heat_flow'], data_point2['heat_flow']], color='green', linestyle='--', label='Connecting Line')
plt.title('DSC Measurement')
plt.legend()
plt.show()
##### type in time to select point



# Find the corresponding index on curve
start_index = np.argmin(np.abs(temperature - data_point1['temperature']))
end_index = np.argmin(np.abs(temperature - data_point2['temperature']))


plt.plot(temperature,heat_flow)
plt.scatter(data_point1['temperature'], data_point1['heat_flow'], color='red', label='Data Point 1') # 增加點
plt.scatter(data_point2['temperature'], data_point2['heat_flow'], color='blue', label='Data Point 2')
plt.scatter(temperature[start_index],heat_flow[start_index], color='green')
plt.scatter(temperature[end_index],heat_flow[end_index], color='black')
plt.legend()
plt.show()

# Characterise the baseline into the data point, for the algorithm to identify numerically
baseline_x = np.linspace(data_point2['temperature'], data_point1['temperature'], num=end_index-start_index)  # Adjust 'num' based on the desired number of points
baseline_y = np.linspace(data_point2['heat_flow'], data_point1['heat_flow'], num=end_index-start_index)
print('**********  start_index and end_index  *************')
print(start_index, end_index)

# Calculate the overall reaction heat
temperature_selected = temperature[start_index:end_index]
heat_flow_curve_selected = heat_flow[start_index:end_index]
time_selected = time[start_index:end_index]
heat_flow_line_selected = baseline_y[0:len(baseline_y)]
difference = np.abs(heat_flow_curve_selected - baseline_y)
reaction_heat = np.trapz(difference, x=temperature_selected)
print('*************************************')
print('reaction heat is :', reaction_heat)
print('*************************************')
# Calculate the reaction heat of every point and accumulate it
accumulate = 0
curing_deg = []
#print(temperature_selected)

# Calculate the reaction heat on each point, and the curing degree
for i in range(start_index, end_index):
    #print(i)
    di = temperature[i+1]-temperature[i]
    h = heat_flow[i]-baseline_y[i-start_index]
    area = h*di
    accumulate = accumulate + area
    curing_rate = accumulate / reaction_heat
    curing_deg.append(curing_rate)
    #print(accumulate)# suppose to be close to overall reaction_heat

# Put the whole DSC process into account
endindex_time_selected = len(time_selected) - 1
time_selected = np.linspace(0,time_selected[endindex_time_selected],len(time_selected))
plt.plot(time_selected, curing_deg, label='curing degree over time')
plt.show()


# ********************* Step3: Calculate the fitting model data *********************
def solve_alpha(parameters, T_values, time_values, alpha_initial):

    alpha_values = []  # set a numpy array for alpha
    A1 = parameters[0]
    E1 = parameters[1]
    A2 = parameters[2]
    E2 = parameters[3]
    m = parameters[4]
    n = parameters[5]
    for i in range(0, len(T_values)-1):
        T = T_values[i] + 273.3
        dt = (time_values[i+1]-time_values[i])
        # Euler method for solving the differential equation
        alpha_dot = (kamal_equation(alpha_initial, A1, E1, A2, E2, m, n, T))
        alpha_new = alpha_initial + alpha_dot * dt
        alpha_values.append(alpha_new)
        alpha_initial = alpha_new
    alpha_values.append(1)

    return alpha_values

# initial_guess  (from Anna's paper)
A1 = 100
E1 = 500000
A2 = 10000000
E2 = 8.17144653e+03
m = 0.5
n = 1.9
alpha0 = 0.001 # it has to be more than 0
initial_guess = [A1,E1,A2,E2,m,n]
print(initial_guess)

alpha_solution = solve_alpha(initial_guess, temperature_selected, time_selected, alpha0)
plt.plot(time_selected, alpha_solution,label='Model Prediction', color ='red')
plt.scatter(time_selected, curing_deg, label='Experimental Data', color='blue')
plt.legend()
plt.show()
time_selected = np.array(time_selected)
temperature_selected = np.array(temperature_selected)
curing_deg = np.array(curing_deg)
# ********************* Step4: Objective function and optimization *********************
# Objective function to optimize
def objective_function(parameters, temperature_values, time_values, alpha_initial, experimental_data):
    model_data = solve_alpha(parameters,temperature_values, time_values, alpha_initial)
    model_data = np.array(model_data)
    residual = (model_data - experimental_data)
    #model data should be a alpha array of the fitting model
    return residual

# Minimization convergence criteria
tolerance = 1e-6 # could be defined by the virtual inspection on curves
max_iterations = 10 # trial and error for test the time consumption

# bounds
def Set_bounds_Kamal(bnd_A1,bnd_E1,bnd_A2,bnd_E2,bnd_m,bnd_n):
    bounds = ([bnd_A1[0], bnd_E1[0], bnd_A2[0], bnd_E2[0], bnd_m[0], bnd_n[0]], [bnd_A1[1], bnd_E1[1], bnd_A2[1], bnd_E2[1], bnd_m[1], bnd_n[1]])
    return bounds

if __name__ == "__main__":
    # Set the bounds
    bnd_A1 = (0.1, 1000)
    bnd_E1 = (12000, 600000)
    bnd_A2 = (6000000, 3000000000)
    bnd_E2 = (5000, 85000)
    bnd_m = (0.1, 10)
    bnd_n = (1, 40)
    bounds = Set_bounds_Kamal(bnd_A1,bnd_E1,bnd_A2,bnd_E2,bnd_m,bnd_n)
    # Set initial parameters

    # Browse the files
    root = tk.Tk()
    app = FileBrowserApp(root)
    root.mainloop()


    residual = []  # for convergence study
    a = objective_function(initial_guess, temperature_selected, time_selected, alpha0, curing_deg)
    # minimize the objective function
    print('******* Fitting start, Minimize the difference ************')
    for i in range(0, max_iterations):
        result = scipy.optimize.least_squares(objective_function, initial_guess, bounds=bounds,
                                              args=(temperature_selected, time_selected, alpha0, curing_deg))
        optimal_parameters = result.x
        for aa in range(0, len(curing_deg)):
            optimal_model_predictions = solve_alpha(optimal_parameters, temperature_selected, time_selected, alpha0)
            residual_per_point = int(curing_deg[aa] - optimal_model_predictions[aa])
            residual.append(residual_per_point)
        residual = np.max(np.abs(residual))
        if residual < tolerance:
            break
        else:
            initial_guess = optimal_parameters
            continue
    print('******* Fitting end ************')

    # print the fitting model after optimization
    print('The fitting parameters are:')
    print(optimal_parameters)

    # plot together with the experimental data
    optimal_model_predictions = solve_alpha(optimal_parameters, temperature_selected, time_selected, alpha0)
    plt.scatter(time_selected, curing_deg, label='Experimental Data', color='blue')
    plt.plot(time_selected, optimal_model_predictions, label='Model Prediction', color='red')
    plt.xlabel('Time')
    plt.ylabel('Alpha')
    plt.legend()
    plt.show()

    # Pro-process for the isothermal curve
    T_0 = [120, 130, 140, 150]
    sheet_name2 = 'SMC - 120 - V10 - 1'
    excel_file_path2 = '/Users/yuanpeien/Desktop/Internship_SMC/AW_ 21.11 meeting protocol/1mm - Squeeze Flow 2023 Analysis - SMC - 120GradC.xlsx'
    df2 = pd.read_excel(excel_file_path2, sheet_name=sheet_name2, engine='openpyxl')
    # time_iso = df2.iloc[1:, 22].to_numpy()
    # time_iso = pd.Series(time_iso).dropna().to_numpy()

    temperature_isothermal = np.full(len(time_iso), T_0[0])
    temperature_isothermal1 = np.full(len(time_iso), T_0[1])
    temperature_isothermal2 = np.full(len(time_iso), T_0[2])
    temperature_isothermal3 = np.full(len(time_iso), T_0[3])

    iso_alpha = solve_alpha(optimal_parameters, temperature_isothermal, time_iso, alpha0)
    iso_alpha1 = solve_alpha(optimal_parameters, temperature_isothermal1, time_iso, alpha0)
    iso_alpha2 = solve_alpha(optimal_parameters, temperature_isothermal2, time_iso, alpha0)
    iso_alpha3 = solve_alpha(optimal_parameters, temperature_isothermal3, time_iso, alpha0)

    # save 120 deg for example
    save_filename = "alpha_isothermal.txt"
    np.savetxt(save_filename, iso_alpha, delimiter=',')

    plt.plot(time_iso, iso_alpha, label='60 deg', color='red')
    plt.plot(time_iso, iso_alpha1, label='80 deg', color='blue')
    plt.plot(time_iso, iso_alpha2, label='100 deg', color='green')
    plt.plot(time_iso, iso_alpha3, label='120 deg', color='yellow')
    plt.legend()
    plt.show()
