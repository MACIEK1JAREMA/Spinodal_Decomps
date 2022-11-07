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
J = 1
Tc = 2.2692*J
T = 0.1*Tc

moment = 1

reps = 20  # number of runs over different initial conditions
dk = 1
n_nths = 12
upper_t_cut = 0.64   # or use 0.8 as in some papers

#Ns = 2**np.arange(5, 10, 1, dtype=np.int64)
Ns = np.array([64, 128, 256, 370, 512, 750, 1024], dtype=np.int64)
#Ns = np.append(Ns, np.arange(200, 700, 50, dtype=np.int64))

exponents = np.zeros(len(Ns))
exp_errs = np.zeros(len(Ns))

for j in range(len(Ns)):
    t0 = int(Ns[j]/10)
    tm = int(Ns[j]*upper_t_cut)
    nth = int((tm-t0)/n_nths)
    
    # set up arrays and length values:
    mcss = int(np.floor((tm-t0)/nth)) + 2
    k_num = len(np.arange(1, int(Ns[j]/2), dk))
    
    # find circularly averages S(kx, ky) = S(k) for each MCS of each initial
    # realisation and store in 3D array
    average = np.zeros((k_num, mcss, reps))
    for i in range(reps):
        average[:, :, i], kvals = MC.Sk_MCrun(Ns[j], J, T, dk, t0, tm, nth=nth)
        print("Finished repetition " + str(i) + " of N value = " + str(Ns[j]))
    
    avgSk = np.sum(average, axis=2)/reps
    avgSk_norm = avgSk / avgSk[:, 0][:, None]
    
    kvals = (2*np.pi/Ns[j])*np.arange(1, int(Ns[j]/2), dk)
    
    k_vals = np.tile(kvals, (len(avgSk_norm[0, :]), 1)).T
    k = np.sum(avgSk_norm*k_vals**(moment+1)*dk, axis=0)/np.sum(avgSk_norm*k_vals*dk, axis=0)
    L = (2*np.pi/k)**(1/moment)
    
    t_vals = nth*(np.arange(1, len(avgSk_norm[0, :]), 1) - 1) + t0
    
    # check the gradient of the linear ones
    exponents[j], _, _, _, exp_errs[j] = linreg(np.log(t_vals), np.log(L[1:]))

# return time to run
stop = timeit.default_timer()
print('Time: ', stop - start)


# plot it as a function of t
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$N$', fontsize=22)
ax.set_ylabel(r'$1/z$', fontsize=22)
ax.errorbar(Ns, exponents, yerr=exp_errs, capsize=2)

# %%

# Saving data
Nsdf = pd.DataFrame(Ns)
Nsdf.to_excel('Data Ising\convergence_N_vals.xlsx', index=True)

exponentsdf = pd.DataFrame(exponents)
exponentsdf.to_excel('Data Ising\convergence_exponents.xlsx', index=True)

exp_errsdf = pd.DataFrame(exp_errs)
exp_errsdf.to_excel('Data Ising\convergence_exp_errs.xlsx', index=True)

# Will need to copy over to combined data Data/MC
