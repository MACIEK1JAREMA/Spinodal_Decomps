# Code to combine plots from GLT and MC
# for L(t) vs t and getting gradients from each

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from scipy.stats import linregress as linreg
import timeit
import pandas as pd

# %%

# Read in both data sets for Sk averaged over many initial considtions
# Need directory on Codes for combined figures
Sk1 = pd.read_excel("Data\MC\Sk_avg_over_reps.xlsx", index_col=0)
Sk1 = Sk1.to_numpy()

t_vals = pd.read_excel("Data\MC\Sk_avg_over_reps_t_vals.xlsx", index_col=0)
t_vals = t_vals.to_numpy()[:, 0]

kvals = pd.read_excel("Data\MC\Sk_avg_over_reps_k_vals.xlsx", index_col=0)
kvals = kvals.to_numpy()[:, 0]

dk = 1

# Normalise to intial
avgSk_norm = Sk1 / Sk1[:, 0][:, None]

# find average k
moment = 1

k_vals_arr = np.tile(kvals, (len(avgSk_norm[0, :]), 1)).T
k = np.sum(avgSk_norm*k_vals_arr**(moment+1)*dk, axis=0)/np.sum(avgSk_norm*k_vals_arr*dk, axis=0)
L = (2*np.pi/k)**(1/moment)  # get L

# plot resulting S(k) at each t step
figSn = plt.figure(figsize=(10, 7))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)
for i in range(1, len(avgSk_norm[0, :])):
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time = str(int(t_vals[i-1])) + " MCS"
    axSn.plot(kvals, avgSk_norm[:, i], label=r"$t=$"+time)
axSn.legend(fontsize=22)

# plot L vs t on log log:
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$t [MCS]$', fontsize=22)
ax.set_ylabel(r'$L(t)$', fontsize=22)
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(labelsize=22)
ax.plot(t_vals, L[1:])  # omit initial condition as at t=0

# check the gradient of the linear ones
m1, c1, rval1, _, std1 = linreg(np.log(t_vals), np.log(L[1:]))
print(f'Linear variation with gradient = {np.round(m1, 4)} and error +- {np.round(std1, 4)}')
print(f'with R-value of {np.round(rval1, 4)}')
print('\n')
ax.plot(t_vals, np.exp(c1)*t_vals**m1, '-.b', label=f'fit at gradient={np.round(m1, 4)}')

# plot S(k) rescaled for proposed universal scaling relation
figUni = plt.figure(figsize=(10, 7))
axUni = figUni.gca()
axUni.tick_params(labelsize=22)
axUni.set_xlabel(r"$kt^{\frac{1}{z}}$", fontsize=22)
axUni.set_ylabel(r"$\frac{S(k) t^{-2/z} }{S(k)|_{t=0}}$", fontsize=22)

for i in range(1, len(avgSk_norm[0, :])):
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time = str(int(t_vals[i-1])) + " MCS"
    axUni.plot(kvals*t_vals[i-1]**m1, avgSk_norm[:, i]/t_vals[i-1]**(2*m1), label=r"$t=$"+time)
