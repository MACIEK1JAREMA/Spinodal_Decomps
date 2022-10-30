# Average S(k, t) over initial realisations

import numpy as np
import MC_module as MC
import matplotlib.pyplot as plt
import timeit

# %%

# start the timer
start = timeit.default_timer()

# set up lattice and variables
N = 512
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
    average[:, :, i] = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)


# average over initial conditions and normalise w.r.t chosen t0
avgSk = np.sum(average, axis=2)/reps
avgSk_norm = avgSk / avgSk[:, 1][:, None]

# plot result
figSn = plt.figure(figsize=(8, 6))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"SF($k$)$/$S($k$)$|_{t_{0}}$", fontsize=22)

for i in range(2, len(avgSk_norm[0, :])):
    time = str(int(nth*(i-1) + t0)) + " MCS"
    axSn.plot(kvals, avgSk_norm[:, i], label=r"$t=$"+time)


axSn.legend(fontsize=22)

# return time to run
stop = timeit.default_timer()
print('Time: ', stop - start)

