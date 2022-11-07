# Average S(k, t) over initial realisations

import numpy as np
import MC_module as MC
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from scipy.stats import linregress as linreg
import timeit
import pandas as pd

# %%

# start the timer
start = timeit.default_timer()

# set up lattice and variables
N = 1024  # takes 30 mins with 10 reps, 1 hr with 20 reps
#N = 2048  # takes 2 hrs with 10 reps, 4 hrs with 20 reps
J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = int(N/10)
tm = int(N*0.64)
nth = int((tm-t0)/15)

reps = 60  # number of runs over different initial conditions
#reps = 20
dk = 1

# set up arrays and length values:
#kvals = 2*np.pi/np.arange(1, int(N/2), dk)
mcss = int(np.floor((tm-t0)/nth)) + 2
k_num = len(np.arange(1, int(N/2), dk))

# find circularly averages S(kx, ky) = S(k) for each MCS of each initial
# realisation and store in 3D array
average = np.zeros((k_num, mcss, reps))
for i in range(reps):
    average[:, :, i], kvals = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)
    print("Finished repetition " + str(i))

# return time to run
stop = timeit.default_timer()
print('Time: ', stop - start)


# ANALYSIS

# average over initial conditions and normalise w.r.t chosen t0
avgSk = np.sum(average, axis=2)/reps

# FROM here just for visual chekcs

avgSk_norm = avgSk / avgSk[:, 0][:, None]
#avgSk_norm = np.nan_to_num(avgSk_norm, 0)


moment = 1


kvals = (2*np.pi/N)*np.arange(1, int(N/2), dk)

# find average k from it and get L
k_vals_arr = np.tile(kvals, (len(avgSk_norm[0, :]), 1)).T
k = np.sum(avgSk_norm*k_vals_arr**(moment+1)*dk, axis=0)/np.sum(avgSk_norm*k_vals_arr*dk, axis=0)
L = (2*np.pi/k)**(1/moment)

t_vals = nth*(np.arange(1, len(avgSk_norm[0, :]), 1) - 1) + t0

# plot resulting S(k) at each t step
figSn = plt.figure(figsize=(8, 6))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)

for i in range(1, len(avgSk_norm[0, :])):
    time = str(int(nth*(i-1) + t0)) + " MCS"
    axSn.plot(kvals, avgSk_norm[:, i], label=r"$t=$"+time)
axSn.legend(fontsize=22)

# plot it as a function of t on log log:
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

# plot S(k) with circularly averaged k and averaged over initial conditions
# rescaled for proposed universal scaling relation
figUni = plt.figure(figsize=(10, 7))
axUni = figUni.gca()
axUni.tick_params(labelsize=22)
axUni.set_xlabel(r"$kt^{\frac{1}{z}}$", fontsize=22)
axUni.set_ylabel(r"$\frac{S(k) t^{-2/z} }{S(k)|_{t=0}}$", fontsize=22)

for i in range(1, len(avgSk_norm[0, :])):
    time = str(int(nth*(i-1) + t0)) + " MCS"
    axUni.plot(kvals*t_vals[i-1]**m1, avgSk_norm[:, i]/t_vals[i-1]**(2*m1), label=r"$t=$"+time)


# %%

# Saving data
Skdf = pd.DataFrame(avgSk)
Skdf.to_excel('Data Ising\Sk_avg_over_reps.xlsx', index=True)

kvalsdf = pd.DataFrame(kvals)
kvalsdf.to_excel('Data Ising\Sk_avg_over_reps_k_vals.xlsx', index=True)

t_valsdf = pd.DataFrame(t_vals)
t_valsdf.to_excel('Data Ising\Sk_avg_over_reps_t_vals.xlsx', index=True)

# Will need to copy over to combined data Data/MC
