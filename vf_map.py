import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from functions import w0_RZ_calc

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = '8'

out_folder = "output/"

#Sim 3 parameters in Si units
sim3_file = "sim3.txt"
Re_st = 0.830031974
SSG_p = 0.0825 #0.0825 - for omega=0.99
D=2.5e-4
u_set_p = 0.0524

nu_p = np.sqrt(SSG_p/(18*Re_st))
u_st = SSG_p*9.81*(D**2)/(18*9.2e-7)
u_st_p = SSG_p/(18*nu_p)
t_ref_p = 1/u_st_p #Non-dimensional reference time
t_ref = D/u_st    #Dimensional reference time

u_set = u_set_p*(u_st/u_st_p)
u_set_p01 = w0_RZ_calc(u_set,0.1,4.0)
t_1000 = t_ref/t_ref_p*1000
print(u_st,u_st_p,t_ref_p,t_ref,t_1000)
x1 = np.linspace(0, t_1000, 200)
y1 = np.linspace(0, 40*D, 101)
X1, Y1 = np.meshgrid(x1, y1)
sim3 = np.genfromtxt(sim3_file,delimiter=',')


#Model parameters
df = pd.read_csv(out_folder+f"res_0.csv",index_col=0)
x2 = np.linspace(0, 5.0, 50)
y2 = np.linspace(0, 40*D, 500)
X2, Y2 = np.meshgrid(x2, y2)

res = []
for i in range(50):
    df = pd.read_csv(out_folder+f"res_{i}.csv",index_col=0)
    res.append(df["phi"])
res = list(map(list, zip(*res)))


fig,ax = plt.subplots(2,1, figsize=(3.0, 4.0),sharex=True)
ax[0].contourf(X1,Y1,sim3,cmap='binary',vmin=0.0,vmax=0.7,levels=20)
ax[1].contourf(X2,Y2,res,cmap='binary',vmin=0.0,vmax=0.7,levels=20)

ax[0].grid(True, which='major', axis='y')
ax[1].grid(True, which='major', axis='y')

ax[0].set_xlim([0,4.3])
ax[0].set_ylim([0,0.01])
ax[1].set_ylim([0,0.01])

ax[0].set_ylabel("H[m]")
ax[1].set_ylabel("H[m]")
ax[1].set_xlabel("t[s]")


#u_s_p01_n = w0_RZ_calc(u_set_p,0.1,5.5)
#u_s_p01_d = u_s_p01_n*

ax[0].plot([0,5],
           [0.01,0.01-(u_set_p01*5)],
           "--",
           color="black",
           lw=0.5,)
ax[0].text(1,0.0085,r"$u_s^{\phi=0.1}$")
ax[1].plot([0,5],
           [0.01,0.01-(u_set_p01*5)],
           "--",
           color="black",
           lw=0.5)
plt.subplots_adjust(left=0.2)
plt.savefig("output/VF_map.png",transparent=True,dpi=300)