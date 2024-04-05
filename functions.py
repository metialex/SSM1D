import numpy as np
import math
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import solve

def w0_RZ_calc(u_ref,phi,n):
    #w0 = u_ref*((1-phi/0.6)**1.0)*(1-phi)/(1-phi/0.65)**(-5*0.65/2)
    w0 = u_ref*((1-phi)**n)
    return w0*(1-(math.e**(phi/0.6-1))**20)

def w0_RZ_new_calc(u_ref,phi,n,n2):
    #w0 = u_ref*((1-phi/0.6)**1.0)*(1-phi)/(1-phi/0.65)**(-5*0.65/2)
    w0 = u_ref*((1-phi)**n+n2*phi)
    return w0*(1-(math.e**(phi/0.6-1))**20)

def w_s_calc(w0,phi,phi_max=0.642,b=10):
    return w0*(1-(math.e**(phi/phi_max-1))**b)

def kappa_calc(w0,rho_w,d_rho_s,phi):
    if phi <= 0: return 0
    else:
        return w0*rho_w/(d_rho_s*phi)

def w_w_calc(w_s,phi):
    return (-w_s*phi/(1-phi))

def init_constant(n,value):
    res=np.empty(n)
    res.fill(value)
    return res

def init_linear(min,dx,n_dx):
    res = []
    for i in range(n_dx):
        res.append(min+i*dx)
    return res

def init_constant_modified(n,value):
    res = [value]*n
    min_idx = int(len(res)*0.80)
    max_idx = int(len(res)*0.85)
    diff = max_idx-min_idx
    for i in range(min_idx,max_idx):
        res[i] = (1-(i-min_idx)/diff)*value
    for i in range(max_idx,len(res)):
        res[i] = 0.0
    return res

def w0_return(phi,u_ref,RZ_const,idx):
    phi_p1 = var_ZF(phi,idx)
    w0 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
    return w0

def w0_new_return(phi,u_ref,RZ_const_1, RZ_const_2,idx):
    phi_p1 = var_ZF(phi,idx)
    w0 = w0_RZ_new_calc(u_ref,phi_p1,RZ_const_1,RZ_const_2)
    return w0

def conv_term(phi,u_ref,RZ_const,rho_w,d_rho_s,idx,dz):
    
    #Boundary conditions
    #Top
    if idx >= len(phi)-1:
        X_p1 = 0
        
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        w_s_m1 = w_s_calc(w0_m1,phi_m1)
        w_w_m1 = w_w_calc(w_s_m1,phi_m1)
        kappa_m1 = kappa_calc(w0_m1,rho_w,d_rho_s,phi_m1)
        
        #Define i+1 term
        X_m1 = w0_m1*(1-phi_m1)*(w_w_m1-w_s_m1)/kappa_m1
    
    #Bottom
    elif idx == 0:
        X_m1 = 0
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        w_s_p1 = w_s_calc(w0_p1,phi_p1)
        w_w_p1 = w_w_calc(w_s_p1,phi_p1)
        kappa_p1 = kappa_calc(w0_p1,rho_w,d_rho_s,phi_p1)
        #Define i+1 term
        X_p1 = w0_p1*(1-phi_p1)*(w_w_p1-w_s_p1)/kappa_p1
    
    #Not BC
    else:
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        w_s_p1 = w_s_calc(w0_p1,phi_p1)
        w_w_p1 = w_w_calc(w_s_p1,phi_p1)
        kappa_p1 = kappa_calc(w0_p1,rho_w,d_rho_s,phi_p1)
        
        #Define i+1 term
        X_p1 = w0_p1*(1-phi_p1)*(w_w_p1-w_s_p1)/kappa_p1
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        w_s_m1 = w_s_calc(w0_m1,phi_m1)
        w_w_m1 = w_w_calc(w_s_m1,phi_m1)
        kappa_m1 = kappa_calc(w0_m1,rho_w,d_rho_s,phi_m1)
        
        #Define i+1 term
        X_m1 = w0_m1*(1-phi_m1)*(w_w_m1-w_s_m1)/kappa_m1
    #Return the term
    return (X_p1-X_m1)/(2*dz)

#Correspond to eqution 21 in Toorman (1996)
def conv_term_reduced(phi,u_ref,RZ_const,idx,dz):
    
    #Boundary conditions
    #Top
    if idx == len(phi)-1:
        X_p2 = 0
        X_p1 = 0
        
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        X_m1 = w0_m1*phi_m1
        #Define i-2 varialbes
        phi_m2 = var_ZF(phi,idx-2)
        w0_m2 = w0_RZ_calc(u_ref,phi_m2,RZ_const)
        X_m2 = w0_m2*phi_m2
    
    elif idx == len(phi)-2:
        X_p2 = 0
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        X_p1 = w0_p1*phi_p1
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        X_m1 = w0_m1*phi_m1
        #Define i-2 varialbes
        phi_m2 = var_ZF(phi,idx-2)
        w0_m2 = w0_RZ_calc(u_ref,phi_m2,RZ_const)
        X_m2 = w0_m2*phi_m2
    
    #Bottom
    elif idx == 1:
        X_m2 = 0
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        X_m1 = w0_m1*phi_m1
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        X_p1 = w0_p1*phi_p1
        #Define i+2 varialbes
        phi_p2 = var_ZF(phi,idx+2)
        w0_p2 = w0_RZ_calc(u_ref,phi_p2,RZ_const)
        X_p2 = w0_p2*phi_p2
        
    elif idx == 0:
        X_m1 = 0
        X_m2 = 0
        
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        X_p1 = w0_p1*phi_p1
        #Define i+2 varialbes
        phi_p2 = var_ZF(phi,idx+2)
        w0_p2 = w0_RZ_calc(u_ref,phi_p2,RZ_const)
        X_p2 = w0_p2*phi_p2
    
    #Not BC
    else:
        #Define i+2 varialbes
        phi_p2 = var_ZF(phi,idx+2)
        w0_p2 = w0_RZ_calc(u_ref,phi_p2,RZ_const)
        X_p2 = w0_p2*phi_p2
        #Define i+1 varialbes
        phi_p1 = var_ZF(phi,idx+1)
        w0_p1 = w0_RZ_calc(u_ref,phi_p1,RZ_const)
        X_p1 = w0_p1*phi_p1
        #Define i-1 varialbes
        phi_m1 = var_ZF(phi,idx-1)
        w0_m1 = w0_RZ_calc(u_ref,phi_m1,RZ_const)
        X_m1 = w0_m1*phi_m1
        #Define i-2 varialbes
        phi_m2 = var_ZF(phi,idx-2)
        w0_m2 = w0_RZ_calc(u_ref,phi_m2,RZ_const)
        X_m2 = w0_m2*phi_m2
    #Return the term
    return (-X_p2+8*X_p1-8*X_m1+X_m2)/(12*dz)

def diff_term(phi,dz,Db,idx):
    if idx == len(phi)-1:
        return 0
    elif idx == 0:
        return 0
        #return Db*(var_ZF(phi,idx+1)-2*var_ZF(phi,idx)+0.6395)/dz**2
    else: 
        return Db*(var_ZF(phi,idx+1)-2*var_ZF(phi,idx)+var_ZF(phi,idx-1))/dz**2

#Returns the variable "var" with index "idx" 
#and applies zero gradient BC for both sides
def var_ZF(var,idx):
    if var[idx] < 0.0: return 0
    elif var[idx] > 1.0: return 1.0
    return var[idx] 

def log_init_output(in_vars,dt,dz):
    #CFL number
    u_ref = in_vars["u_ref"]
    
    print("Input varialbes for the present model")
    for key,value in in_vars.items():
        print(key,"\t = \t",value)
    print(f"CFL number is = {-u_ref*dt/dz}")