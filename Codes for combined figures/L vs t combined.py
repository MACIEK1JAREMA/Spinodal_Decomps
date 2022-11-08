# Code to combine plots from GLT and MC
# for L(t) vs t and getting gradients from each

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from scipy.stats import linregress as linreg
import timeit
import pandas as pd

# %%

# Read in MC data for Sk averaged over many initial considtions
# Need directory on Codes for combined figures
Sk1 = pd.read_excel("Data\MC\Sk_avg_over_reps.xlsx", index_col=0)
Sk1 = Sk1.to_numpy()
t_vals1 = pd.read_excel("Data\MC\Sk_avg_over_reps_t_vals.xlsx", index_col=0)
t_vals1 = t_vals1.to_numpy()[:, 0]
kvals1 = pd.read_excel("Data\MC\Sk_avg_over_reps_k_vals.xlsx", index_col=0)
kvals1 = kvals1.to_numpy()[:, 0]

# Read in GLT data
num_time_steps = 16
averaged_sf2 = np.zeros(num_time_steps, dtype=object)
for i in range(num_time_steps):
    name = "Data\GLT\model A average unnormalised sf #"+str(i)+" over 10 inits.txt)"
    averaged_sf2[i] = np.loadtxt(name)
kvals2 = np.loadtxt("Data\GLT\model A kvals.txt")
time_steps2 = np.loadtxt("Data\GLT\model A time steps.txt")

# normalise each
for sf in averaged_sf2[1:]:
    normalised_sf2 = sf/averaged_sf2[0]

# Normalise to intial
avgSk_norm1 = Sk1 / Sk1[:, 0][:, None]

# find average k
moment = 1
dk = 1   # from when data was run
k_vals_arr1 = np.tile(kvals1, (len(avgSk_norm1[0, :]), 1)).T
k1 = np.sum(avgSk_norm1*k_vals_arr1**(moment+1)*dk, axis=0)/np.sum(avgSk_norm1*k_vals_arr1*dk, axis=0)
L1 = (2*np.pi/k1)**(1/moment)  # get L

# plot resulting S(k) at each t step
figSn = plt.figure(figsize=(10, 7))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)
plotted_ts = [1, 5, 10, 15]   # Update once data is produced
for i in plotted_ts:   # Update once data is produced
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time = str(int(t_vals1[i])) + " MCS"
    axSn.plot(kvals1, avgSk_norm1[:, i], label=r"$t=$"+time)

# plot L vs t on log log:
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$t [MCS]$', fontsize=22)
ax.set_ylabel(r'$L(t)$', fontsize=22)
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(labelsize=22)
ax.plot(t_vals1, L1[1:], 'g^', ms=10)  # omit initial condition as at t=0

# check the gradient of the linear ones
m1, c1, rval1, _, std1 = linreg(np.log(t_vals1), np.log(L1[1:]))
print(f'Linear variation with gradient = {np.round(m1, 4)} and error +- {np.round(std1, 4)}')
print(f'with R-value of {np.round(rval1, 4)}')
print('\n')
ax.plot(t_vals1, np.exp(c1)*t_vals1**m1, '-.b', label=f'fit at gradient={np.round(m1, 4)}')

# plot S(k) rescaled for proposed universal scaling relation
figUni = plt.figure(figsize=(10, 7))
axUni = figUni.gca()
axUni.tick_params(labelsize=22)
axUni.set_xlabel(r"$kt^{\frac{1}{z}}$", fontsize=22)
axUni.set_ylabel(r"$\frac{S(k) t^{-2/z} }{S(k)|_{t=0}}$", fontsize=22)

for i in plotted_ts:
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time = str(int(t_vals1[i-1])) + " MCS"
    axUni.plot(kvals1*t_vals1[i-1]**m1, avgSk_norm1[:, i]/t_vals1[i-1]**(2*m1), label=r"$t=$"+time)


axSn.legend(fontsize=22)
ax.legend(fontsize=22)
axUni.legend(fontsize=22)


