import os
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import solve
import warnings
import functions as func

warnings.simplefilter("error")

#Define global functions

#Global variables
in_vars= {
    "H": 0.01,          #height of the column           [m]
    "t": 5e-0,          #total time of the simulation   [s]
    "nu": 9.2e-7,       #kinematic viscosity of fluid   [m2/s]
    "rho_w": 1000,      #density of water               [kg/m3]
    "rho_s": 1082.5,    #density of sediment            [kg/m3]
    "RZ": 4.0,
    "Db": 5e-8,         #diffusion coefficient that 
                        #account for particle interaction
    
    #Initial values for variables
    "phi_in": 0.1,     #Initial sediment fraction      [-]
    "w_s_in": 0.0,      #Initial sediment velocity      [m/s]
    "w_w_in": 0.0,      #Initial water velocity         [m/s]

    #Numerical parameters
    "n_dz": 500,
    "n_dt": 5000,
    
    #Variables for non-dimensionalization
    "l_ref": 2.5e-4,    #reference lenght scale         [m]
    "u_ref": 2.6364e-3, #reference settling velocity    [m/s]
    
    #Output parameters
    "out_f": 100,       #Output frequency
    "out_fl": "output/"
}

#Create output folder
if not os.path.exists(in_vars["out_fl"]):
    os.makedirs(in_vars["out_fl"])

#Derive other variables from input
d_rho_s = in_vars["rho_s"]-in_vars["rho_w"]
dz = in_vars["H"]/in_vars["n_dz"]
dt = in_vars["t"]/in_vars["n_dt"]

#Convection coefficient
s = dt*in_vars["u_ref"]/(2*dz)

#Non-dimensional parameters
t_ref = in_vars["l_ref"]/in_vars["u_ref"]

phi = func.init_constant(in_vars["n_dz"],in_vars["phi_in"])
phi_prev = func.init_constant(in_vars["n_dz"],in_vars["phi_in"])
w_s = func.init_constant(in_vars["n_dz"],in_vars["w_s_in"])
z = func.init_linear(0,dz,in_vars["n_dz"])
w0_arr = func.init_constant(in_vars["n_dz"],0)

#Initialize arrays for varialbes
int_conc = []
time_arr = []
total_mass = []
out_idx = 0

func.log_init_output(in_vars,dt,dz)

for n_t_it in tqdm(range(in_vars["n_dt"])):
    time = n_t_it*dt
    time_non_dim = time/t_ref
    time_arr.append(time)
    #Calculation of the new phi field
    for idx in range(len(phi)):
        w0 = func.w0_return(phi,in_vars["u_ref"],in_vars["RZ"],idx)
        w0_arr[idx] = w0

        conv = func.conv_term_reduced(phi_prev,
                              in_vars["u_ref"],
                              in_vars["RZ"],
                              idx,
                              dz)
        
        diff = func.diff_term(phi_prev,dz,in_vars["Db"],idx)
        phi[idx] = phi_prev[idx] + dt*(conv+diff)
    for idx,phi_cell in enumerate(phi):
        phi_prev[idx] = phi_cell

    total_mass.append(np.sum(phi))
    
    #Write output data
    if n_t_it%in_vars["out_f"] == 0:
        output = {"time":time,
                  "t_non_dim":time_non_dim,
                  "z":z,
                  "phi":phi,
                  "w0":w0_arr}
        out_df = pd.DataFrame(output)
        out_df.to_csv(f"{in_vars['out_fl']}res_{out_idx}.csv")
        out_idx += 1
plt.plot(time_arr,total_mass)
plt.show()