# Average S(k, t) over initial realisations

import numpy as np
import MC_module as MC
import matplotlib.pyplot as plt


# %%

# set up lattice and variables
N = 512

J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 5
tm = 90

nth = 15

# number of intial condition runs
reps = 2
dk = 1
kvals = np.arange(0, N, dk)

mcss = int(np.floor((tm-t0)/nth)) + 2
k_num = len(np.arange(0, N, dk))
average = np.zeros((k_num, mcss, reps))

for i in range(reps):
    average[:, :, i] = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)


# average over initial conditions too:
avgSk = np.sum(average, axis=2)/reps

# and normalise:
avgSk_norm = np.divide(avgSk, avgSk[:, 1])

figSn = plt.figure(figsize=(8, 6))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"SF($k$)$/$S($k$)$|_{t_{0}}$", fontsize=22)

for i in range(len(avgSk_norm[0, :])):
    time = str(int(nth*(i-1) + t0)) + " MCS"
    axSn.plot(kvals, avgSk_norm, label=r"$t=$"+time)


















