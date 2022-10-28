# First Monte Carlo SImulation
# If time allows, later to be optimized and modularised

import numpy as np
import matplotlib.pyplot as plt


# %%

# set up lattice and variables
N = 64  # keep to powers of 2
lat = np.random.randint(2, size=(N, N))

J = 1  # isotropic
Tc = 2.2692*J
T = 0.5*Tc  # Using k_{B}=1
n0 = 10
nm = 100

for t in range(nm):
    # pick a random spin site:
    i, j = np.random.randint(N, size=2)
    
