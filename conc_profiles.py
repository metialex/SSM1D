import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = '8'

out_folder = "output/"

fig,ax = plt.subplots(1,1, figsize=(4.0, 3.0),sharex=True)
for i in range(0,49,5):
    df = pd.read_csv(out_folder+f"res_{i}.csv",index_col=0)
    df.plot(x="phi",y="z",ax=ax)

plt.xlim([0,0.7])
plt.ylim([0,0.01])
plt.show()