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
kvals2 = np.loadtxt("Data\GLT\model A kvals.txt")
t_vals2 = np.loadtxt("Data\GLT\model A time steps.txt")
averaged_sf2 = np.zeros((num_time_steps, len(kvals2)))
for i in range(num_time_steps):
    name = "Data\GLT\model A average unnormalised sf #"+str(i)+" over 10 inits.txt"
    averaged_sf2[i, :] = np.loadtxt(name)

# averaged_sf2 now has different t in different rows
#     + different k in different columns
# to match, transpose it
averaged_sf2 = averaged_sf2.T


# Normalise to intial
avgSk_norm1 = Sk1 / Sk1[:, 0][:, None]
avgSk_norm2 = averaged_sf2/averaged_sf2[:, 0][:, None]

# find average k
moment = 1
dk = 1   # from when data was run
k_vals_arr1 = np.tile(kvals1, (len(avgSk_norm1[0, :]), 1)).T
k1 = np.sum(avgSk_norm1*k_vals_arr1**(moment+1)*dk, axis=0)/np.sum(avgSk_norm1*k_vals_arr1*dk, axis=0)
L1 = (2*np.pi/k1)**(1/moment)  # get L

k_vals_arr2 = np.tile(kvals2, (len(avgSk_norm2[0, :]), 1)).T
k2 = np.sum(avgSk_norm2*k_vals_arr2**(moment+1)*dk, axis=0)/np.sum(avgSk_norm2*k_vals_arr2*dk, axis=0)
L2 = (2*np.pi/k2)**(1/moment)  # get L

# plot resulting S(k) at each t step
# for MC
figSn1 = plt.figure(figsize=(10, 7))
axSn1 = figSn1.gca()
axSn1.tick_params(labelsize=22)
axSn1.set_xlabel(r"$k$", fontsize=22)
axSn1.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)
axSn1.set_xlim(0, np.pi/4)
plotted_ts1 = [1, 5, 10, 15]   # Update once data is produced
for i in plotted_ts1:   # Update once data is produced
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time1 = str(int(t_vals1[i])) + " MCS"
    axSn1.plot(kvals1, avgSk_norm1[:, i], label=r"$t=$"+time1)

# plot resulting S(k) at each t step
# for GLT
figSn2 = plt.figure(figsize=(10, 7))
axSn2 = figSn2.gca()
axSn2.tick_params(labelsize=22)
axSn2.set_xlabel(r"$k$", fontsize=22)
axSn2.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)
axSn2.set_xlim(0, np.pi/4)
plotted_ts2 = [1, 5, 10, 15]   # Update once data is produced
for i in plotted_ts2:   # Update once data is produced
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time2 = str(int(t_vals2[i]))
    axSn2.plot(kvals2, avgSk_norm2[:, i], label=r"$t=$"+time2)


# plot L vs t on log log TOGETHER
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$log(t)$', fontsize=22)
ax.set_ylabel(r'$log(L(t))$', fontsize=22)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.tick_params(labelsize=22)
ax.plot(np.log(t_vals1), np.log(L1[1:]), 'g^', ms=10)  # omit initial condition as at t=0
ax.plot(np.log(t_vals2[1:]), np.log(L2[1:]), 'bo', ms=10)  # omit initial condition as at t=0

# check the gradient of the linear ones
m1, c1, rval1, _, std1 = linreg(np.log(t_vals1), np.log(L1[1:]))
print(f'MC for 1/z = {np.round(m1, 4)} and error +- {np.round(std1, 4)}')
print(f'with R-value of {np.round(rval1, 4)}')
print('\n')
ax.plot(np.log(t_vals1), c1 + m1* np.log(t_vals1), '-.g', label=r'MC gradient={:.4f} $\pm$ {:.4f}'.format(np.round(m1, 4), np.round(std1, 4)))

m2, c2, rval2, _, std2 = linreg(np.log(t_vals2[1:]), np.log(L2[1:]))
print(f'GLT for 1/z = {np.round(m2, 4)} and error +- {np.round(std2, 4)}')
print(f'with R-value of {np.round(rval2, 4)}')
print('\n')
ax.plot(np.log(t_vals2[1:]), c2 + m2* np.log(t_vals2[1:]), '-.b', label=r'GLT gradient={:.4f} $\pm$ {:.4f}'.format(np.round(m2, 4), np.round(std2, 4)))

# plot S(k) rescaled for proposed universal scaling relation
# for MC
figUni1 = plt.figure(figsize=(10, 7))
axUni1 = figUni1.gca()
axUni1.tick_params(labelsize=22)
axUni1.set_xlabel(r"$kt^{\frac{1}{z}}$", fontsize=22)
axUni1.set_ylabel(r"$\frac{S(k) t^{-2/z} }{S(k)|_{t=0}}$", fontsize=22)
axUni1.set_xlim(0, (np.pi/4)*t_vals1[-1]**m1)

for i in plotted_ts1:
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time1 = str(int(t_vals1[i])) + " MCS"
    axUni1.plot(kvals1*t_vals1[i]**m1, avgSk_norm1[:, i]/t_vals1[i]**(2*m1), label=r"$t=$"+time1)


figUni2 = plt.figure(figsize=(10, 7))
axUni2 = figUni2.gca()
axUni2.tick_params(labelsize=22)
axUni2.set_xlabel(r"$kt^{\frac{1}{z}}$", fontsize=22)
axUni2.set_ylabel(r"$\frac{S(k) t^{-2/z} }{S(k)|_{t=0}}$", fontsize=22)
axUni2.set_xlim(0, (np.pi/4)*t_vals2[-1]**m2)
for i in plotted_ts2:
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time2 = str(int(t_vals2[i]))
    axUni2.plot(kvals2*t_vals2[i]**m2, avgSk_norm2[:, i]/t_vals2[i]**(2*m2), label=r"$t=$"+time2)


# Plot with gradient of 0.5:
m1, m2 = 0.5, 0.5
# plot S(k) rescaled for proposed universal scaling relation
# for MC
figUni05a = plt.figure(figsize=(10, 7))
axUni05a = figUni05a.gca()
axUni05a.tick_params(labelsize=22)
axUni05a.set_xlabel(r"$kt^{\frac{1}{2}}$", fontsize=22)
axUni05a.set_ylabel(r"$\frac{S(k) t^{-1} }{S(k)|_{t=0}}$", fontsize=22)
axUni05a.set_xlim(0, (np.pi/4)*t_vals1[-1]**m1)

for i in plotted_ts1:
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time1 = str(int(t_vals1[i])) + " MCS"
    axUni05a.plot(kvals1*t_vals1[i]**m1, avgSk_norm1[:, i]/t_vals1[i]**(2*m1), label=r"$t=$"+time1)


figUni05b = plt.figure(figsize=(10, 7))
axUni05b = figUni05b.gca()
axUni05b.tick_params(labelsize=22)
axUni05b.set_xlabel(r"$kt^{\frac{1}{2}}$", fontsize=22)
axUni05b.set_ylabel(r"$\frac{S(k) t^{-1} }{S(k)|_{t=0}}$", fontsize=22)
axUni05b.set_xlim(0, (np.pi/4)*t_vals2[-1]**m2)
for i in plotted_ts2:
    #time = str(int(nth*(i-1) + t0)) + " MCS"
    time2 = str(int(t_vals2[i]))
    axUni05b.plot(kvals2*t_vals2[i]**m2, avgSk_norm2[:, i]/t_vals2[i]**(2*m2), label=r"$t=$"+time2)


axSn1.legend(fontsize=22)
axUni1.legend(fontsize=22)
axSn2.legend(fontsize=22)
axUni2.legend(fontsize=22)
axUni05a.legend(fontsize=22)
axUni05b.legend(fontsize=22)
ax.legend(fontsize=22)
