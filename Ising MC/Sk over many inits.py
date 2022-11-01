# Average S(k, t) over initial realisations

import numpy as np
import MC_module as MC
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from scipy.stats import linregress as linreg
import timeit

# %%

# start the timer
start = timeit.default_timer()

# set up lattice and variables
N = 256
J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 1
tm = 50
nth = 2

reps = 10  # number of runs over different initial conditions
dk = 1

# set up arrays and length values:
kvals = np.arange(0, N, dk)
mcss = int(np.floor((tm-t0)/nth)) + 2
k_num = len(np.arange(0, N, dk))

# find circularly averages S(kx, ky) = S(k) for each MCS of each initial
# realisation and store in 3D array
average = np.zeros((k_num, mcss, reps))
for i in range(reps):
    average[:, :, i], kmax = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)


# average over initial conditions and normalise w.r.t chosen t0
avgSk = np.sum(average, axis=2)/reps
avgSk_norm = avgSk / avgSk[:, 0][:, None]
avgSk_norm = np.nan_to_num(avgSk_norm, 0)

## plot resulting S(k) at each t step
#figSn = plt.figure(figsize=(8, 6))
#axSn = figSn.gca()
#axSn.tick_params(labelsize=22)
#axSn.set_xlabel(r"$k$", fontsize=22)
#axSn.set_ylabel(r"S($k$)$/$S($k$)$|_{t=0}$", fontsize=22)
#
#for i in range(1, len(avgSk_norm[0, :])):
#    time = str(int(nth*(i-1) + t0)) + " MCS"
#    axSn.plot(kvals, avgSk_norm[:, i], label=r"$t=$"+time)
#
#
#axSn.legend(fontsize=22)

# return time to run
stop = timeit.default_timer()
print('Time: ', stop - start)

# find average k from it and get L
k_vals = np.tile(kvals, (len(avgSk_norm[0, :]), 1)).T
k = np.sum(avgSk_norm*k_vals**2, axis=0)/np.sum(avgSk_norm*k_vals, axis=0)
L = 2*np.pi/k

t_vals = nth*(np.arange(1, len(avgSk_norm[0, :]), 1) - 1) + t0

# plot it as a function of t on log log:
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
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

t_vals = np.append(np.array([0]), t_vals)

for i in range(0, len(avgSk_norm[0, :])):
    if i == 0:
        time = "0 MCS"
    else:
        time = str(int(nth*(i-1) + t0)) + " MCS"
        
    axUni.plot(kvals*t_vals[i]**m1, avgSk_norm[:, i]/t_vals[i]**(-2*m1), label=r"$t=$"+time)
