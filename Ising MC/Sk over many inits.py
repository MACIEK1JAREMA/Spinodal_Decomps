# Average S(k, t) over initial realisations

import numpy as np
import MC_module as MC
import matplotlib.pyplot as plt
import timeit

# %%

# start the timer
start = timeit.default_timer()

# set up lattice and variables
N = 128
J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 5
tm = 90
nth = 15

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
np.nan_to_num(avgSk_norm, 0)

# plot result
figSn = plt.figure(figsize=(8, 6))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"SF($k$)$/$S($k$)$|_{t=0}$", fontsize=22)

for i in range(1, len(avgSk_norm[0, :])):
    time = str(int(nth*(i-1) + t0)) + " MCS"
    axSn.plot(kvals, avgSk_norm[:, i], label=r"$t=$"+time)


axSn.legend(fontsize=22)

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
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$t [MCS]$', fontsize=22)
ax.set_ylabel(r'$L(t)$', fontsize=22)
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(t_vals, L[1:])  # omit initial condition as at t=0

