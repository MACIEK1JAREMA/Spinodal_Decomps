# time scale testing of MC code
# MC module

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import MC_module as MC
import time

# %%


N = 20  # initial value for N
Nfinal = 500
N_step = 20  # make sure all used N are even!

# system parameters:
J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 1
tm = 20
nth = 2

tarray = np.array([])
Narray = np.array([])

while N <= Nfinal:
    st = time.time()  # start time
    
    # set up lattice and variables
    configs = MC.MC(N, J, T, t0, tm, nth=nth)
    et = time.time()  # end time
    
    tarray = np.append(tarray, et - st)
    Narray = np.append(Narray, N)
    
#    print('Execution time:', elapsed_time, 'seconds')
    N = N + N_step

#Narray[924] = 4
#sizearray[924] =4        

#Narray[916] = 4
#sizearray[916] =4


fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r"$N$", fontsize=22)
ax.set_ylabel(r"$t [s]$", fontsize=22)

ax.plot(Narray**2, tarray)  # if linear then scales quadratically as might be expected

# %%

# now for full code to S(k)

# system parameters:
J = 1
Tc = 2.2692*J
T = 0.1*Tc
dk = 1

t0 = 5
tm = 90
nth = 15
reps = 3

# set up values to test times for, both number of repetitions and system size
upto_power = 10
N_vals = 2**np.arange(2, upto_power, dtype=np.int64)  # keeping to powers of 2 for FT

tarray = np.zeros((upto_power-2))

n = 0
for N in N_vals:
    
    kvals = np.arange(0, N, dk)
    mcss = int(np.floor((tm-t0)/nth)) + 2
    k_num = len(np.arange(0, N, dk))
    
    st = time.time()  # start time
    
    average = np.zeros((k_num, mcss, reps))
    for i in range(reps):
        average[:, :, i] = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)
    
    # average over initial conditions and normalise w.r.t chosen t0
    avgSk = np.sum(average, axis=2)/reps
    avgSk_norm = avgSk / avgSk[:, 1][:, None]
    
    et = time.time()  # end time    
    
    tarray[n] = et - st
    n += 1

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r"$N$", fontsize=22)
ax.set_ylabel(r"$t [s]$", fontsize=22)

ax.plot(N_vals, tarray)


fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.gca()
ax1.tick_params(labelsize=22)
ax1.set_xlabel(r"$N^{2}$", fontsize=22)
ax1.set_ylabel(r"$t [s]$", fontsize=22)

ax1.plot(N_vals**2, tarray)

# %%

# and same but test reps too - 2D plot ----- NOT WORKING YET --------

# system parameters:
J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 1
tm = 20
nth = 2

dk = 1
kvals = np.arange(0, N, dk)
mcss = int(np.floor((tm-t0)/nth)) + 2
k_num = len(np.arange(0, N, dk))

# set up values to test times for, both number of repetitions and system size
upto_power = 8
N_vals = 2**np.arange(8, dtype=np.int32)

num_reps = 10
reps = np.arange(1, num_reps, 1, dtype=np.int32)

tarray = np.zeros((upto_power, num_reps))

i, j = 0, 0
for r in reps:
    for N in N_vals:
        st = time.time()  # start time
        
        average = np.zeros((k_num, mcss, r))
        for i in range(r):
            average[:, :, i] = MC.Sk_MCrun(N, J, T, dk, t0, tm, nth=nth)
        
        # average over initial conditions and normalise w.r.t chosen t0
        avgSk = np.sum(average, axis=2)/r
        avgSk_norm = avgSk / avgSk[:, 1][:, None]
        
        
        et = time.time() # end time
        elapsed_time = et - st
        
        tarray[j, i] = elapsed_time
    
        j += 1
    i += 1

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r"$Repetitions$", fontsize=22)
ax.set_ylabel(r"$N$", fontsize=22)
ax.set_zlabel(r"$t [s]$", fontsize=22)

ax.pcolormesh(reps, N_vals, tarray, cmap="Greys")
