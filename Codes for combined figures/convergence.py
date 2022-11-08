# Code to combine plots from GLT and MC
# for 1/z vs N

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from scipy.stats import linregress as linreg
import timeit
import pandas as pd

# %%


# Read in MC data sets for 1/z from each N
# Need directory on Codes for combined figures
Ns = pd.read_excel("Data\MC\convergence_N_vals.xlsx", index_col=0)
Ns = Ns.to_numpy()[:, 0]
exponents = pd.read_excel("Data\MC\convergence_exponents.xlsx", index_col=0)
exponents = exponents.to_numpy()[:, 0]
exp_errs = pd.read_excel("Data\MC\convergence_exp_errs.xlsx", index_col=0)
exp_errs = exp_errs.to_numpy()[:, 0]

# Read in GLT data set
zlist2 = np.loadtxt("Data\GLT\model A 1/z values.txt")
zerr2 = np.loadtxt("Data\GLT\model A 1/z error bars.txt")
Ns2 = np.loadtxt("Data\GLT\model A 1/z system sizes.txt")

# plot it as a function of t
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=22)
ax.set_xlabel(r'$N$', fontsize=22)
ax.set_ylabel(r'$1/z$', fontsize=22)
ax.errorbar(Ns, exponents, yerr=exp_errs, capsize=2, marker="^", color="g", ms=10, label="MC")
ax.errorbar(Ns2, zlist2, yerr=zerr2, capsize=2, marker="o", color="r", ms=10, label="GLT")
ax.legend(fontsize=22)
