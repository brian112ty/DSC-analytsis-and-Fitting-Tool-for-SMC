import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
import math

# Cross-Andrade Model definition
def Cross_Andrade(n,tau,a,b,T_0,shear_rate):
    eta_0 = a * np.exp(b/T_0) * T_0
    eta = eta_0 / (1 + (eta_0 * shear_rate / tau )**(1-n) )
    return eta

# *************** Step 1: input the Squeeze flow test data *********************
# Low temperature data preprocess (input, calculate shear rate and viscosity)
data_iso = np.loadtxt('/Users/yuanpeien/Desktop/Internship_SMC/Python_code/alpha_isothermal.txt')
sheet_name_low = 'SMC - 80 - V10 - 1'
sheet_name_high = 'SMC - 120 - V10 - 1'
excel_file_path1 = '/Users/yuanpeien/Desktop/Internship_SMC/1mm - Squeeze Flow 2023 Analysis - SMC - 80GradC.xlsx'
excel_file_path2 = '/Users/yuanpeien/Desktop/Internship_SMC/AW_ 21.11 meeting protocol/1mm - Squeeze Flow 2023 Analysis - SMC - 120GradC.xlsx'
df = pd.read_excel(excel_file_path1,sheet_name=sheet_name_low, engine='openpyxl')
df2 = pd.read_excel(excel_file_path2,sheet_name=sheet_name_high, engine='openpyxl')
# input the specific columns
Temperature = 120+273
shear_rate_kia = df.iloc[1:, 20].to_numpy()
viscosity_kia =  df.iloc[1:, 21].to_numpy()

viscosity_kia_1 = df2.iloc[1:, 21].to_numpy()
time_1 = df2.iloc[1:, 22].to_numpy()

# filter out the nan data from Excel (empty cell)
shear_rate_kia = pd.Series(shear_rate_kia).dropna().to_numpy()
viscosity_kia = pd.Series(viscosity_kia).dropna().to_numpy()
viscosity_kia_1 = pd.Series(viscosity_kia_1).dropna().to_numpy()
time_1 = pd.Series(time_1).dropna().to_numpy()
print(data_iso)
print(viscosity_kia_1)
#plt.plot(shear_rate_kia, viscosity_kia , label='Cross-Andrade, Low temperature data')
#plt.show()

# *************** Step 2: Set the initial parameters and bound for Cross-Andrade model *********************
initial_guess = [0.5, 8, 1, 8000] # please type in n, tau, a and b
bnd_n = (0.001 , 100)
bnd_tau = (0.001, 50)
bnd_a = (0.0001,20)
bnd_b = (1, 150000)
bounds = ([bnd_n[0], bnd_tau[0], bnd_a[0], bnd_b[0]], [bnd_n[1], bnd_tau[1], bnd_a[1], bnd_b[1]])

def Calculate_viscosity(initial_guess, T_0, shear_rate):
    viscosity_value = []
    n = initial_guess[0]
    tau = initial_guess[1]
    a = initial_guess[2]
    b = initial_guess[3]
    for i in range(0, len(shear_rate)):
        shearrate = shear_rate[i]
        viscosity = Cross_Andrade(n, tau, a, b, T_0, shearrate)
        viscosity_value.append(viscosity)
    return viscosity_value


viscosity_fit = Calculate_viscosity(initial_guess, Temperature, shear_rate_kia)
#viscosity_fit = pd.Series(viscosity_fit).dropna().to_numpy()
plt.plot(shear_rate_kia, viscosity_fit, label = 'Fit data', color = 'green')
plt.scatter(shear_rate_kia, viscosity_kia, label='Experimental Data', color='blue')
plt.legend()
plt.show()

# *************** Step 3: Set the objective function and optimization *********************
def objective_function(initial_guess, shear_rate, T_0, experimental_data):
    viscosity_model = Calculate_viscosity(initial_guess, T_0, shear_rate)
    residual = (experimental_data - viscosity_model)**2
    residual = np.max(residual)
    return residual

iteration = 1
tolerance = 10
residual = []

for i in range(0, iteration):
    result = scipy.optimize.least_squares(objective_function, initial_guess, bounds=bounds, args=(shear_rate_kia, Temperature, viscosity_kia))
    optimal_parameters = result.x
    viscosity_fit = Calculate_viscosity(optimal_parameters,Temperature,shear_rate_kia)
    residual = np.abs(viscosity_fit - viscosity_kia)
    initial_guess = optimal_parameters
    if np.max(residual) < tolerance:
        break
    else:
        initial_guess = optimal_parameters
        print('next parameters set for iteration:', initial_guess)
        continue

viscosity_fit = Calculate_viscosity(optimal_parameters,Temperature,shear_rate_kia)
print(' **************** The best fitting parameters are : ****************')
print(optimal_parameters)
print('********************************************************************')
n = optimal_parameters[0]
tau = optimal_parameters[1]
a = optimal_parameters[2]
b = optimal_parameters[3]
eta_0 = a * np.exp(b / Temperature) * Temperature
print(eta_0)
plt.scatter(shear_rate_kia, viscosity_kia, label = 'Experimental data', color = 'green')
plt.plot(shear_rate_kia, viscosity_fit, label='Fit Data', color='blue')
plt.legend()
plt.show()

# *************** Step 4:  High temperature viscosity model fitting *********************

# Macosko model
def Macosko(D, E, alpha_gel, eta_0, alpha):
    #epsilon = 1e-6
    #if abs(alpha_gel - alpha) < epsilon:
        # Handle the case when alpha_gel is very close to alpha
        #eta_1 = float('inf')
    #else:
    eta_1 = eta_0 * ((alpha_gel/(alpha_gel-alpha)) ** (D + E*alpha))
    return eta_1

initial_guess_Macosko = [1.2, 0.1, 0.85, n, tau]
D = initial_guess_Macosko[0]
E = initial_guess_Macosko[1]
alpha_gel = initial_guess_Macosko[2]

def Calculate_viscosity_Macosko(initial_guess,shear_rate, eta_0, alpha_array):
    Viscosity_Macosko = []
    D = initial_guess[0]
    E = initial_guess[1]
    alpha_gel = initial_guess[2]
    n = initial_guess[3]
    tau = initial_guess[4]
    for i in range(0, len(alpha_array)):
        alpha = alpha_array[i]
        alpha = np.float64(alpha)
        alpha_gel = np.float64(alpha_gel)
        print(alpha)
        viscosity_M_new = Macosko(D, E, alpha_gel, eta_0, alpha)
        viscosity_M_new = viscosity_M_new / (viscosity_M_new * shear_rate[i] /tau)**(1-n)+1
        print(viscosity_M_new)
        if math.isnan(viscosity_M_new):
            break
        else:
            Viscosity_Macosko.append(viscosity_M_new)
        print(Viscosity_Macosko)
    return Viscosity_Macosko
print('********************************************************************')
aaa = Calculate_viscosity_Macosko(initial_guess_Macosko,shear_rate_kia,eta_0,data_iso)
shear_rate_kia = shear_rate_kia[:len(aaa)]
plt.plot(shear_rate_kia, aaa, color='pink')
plt.show()
print('*************')
def objective_function_macosko(initial_guess, eta_0, alpha_array, experimental_data):
    model = Calculate_viscosity_Macosko(initial_guess,eta_0,alpha_array)
    if len(model) > len(experimental_data):
        model = model[:len(experimental_data)]

    residual = np.max(model - experimental_data)
    return residual



plt.plot(time, viscosity_kia, label ='High temperature', color ='yellow')
plt.show()
bnd_D = (0.001, 20)
bnd_E = (0.0001, 30)
bnd_alpha_gel = (0.0001 , 0.999)
bounds_macosko = ([bnd_D[0], bnd_E[0], bnd_alpha_gel[0]], [bnd_D[1], bnd_E[1], bnd_alpha_gel[1]])

iteration = 100
tolerance = 1

for i in range(0, iteration):
    result = scipy.optimize.least_squares(objective_function_macosko, initial_guess_Macosko, bounds=bounds_macosko, args = (eta_0, data_iso, viscosity_kia))
    optimal_parameters_m = result.x
    b = Calculate_viscosity_Macosko(optimal_parameters_m, eta_0, data_iso)
    dif = b - viscosity_kia_1
    if np.max(dif) < tolerance:
        break
    else:
        initial_guess_Macosko = optimal_parameters_m
        continue

print(optimal_parameters_m)
b = Calculate_viscosity_Macosko(optimal_parameters_m, eta_0, data_iso)

plt.plot(time_test,viscosity_kia_test)
plt.plot(time_test,b,color='black')
plt.show()
print(b)