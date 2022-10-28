# First Monte Carlo SImulation
# If time allows, later to be optimized and modularised

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def MC(N, J, T, t0, tm, nth=1):
    """
    
    """
    
    # generate lattice
    lat = np.random.rand(N, N)
    lat = np.sign(lat - 0.5)
    
    # generate random comparison numbers
    z = np.random.rand(1, tm*N**2)  # random number for attempt of each MCS
    
    # set up nn index arrays:
    inds = np.arange(0, N, 1)
    inds_c = np.ones((N,1), dtype=np.int64)*inds
    inds_r = inds_c.T
    
    nn_t = np.roll(inds_r, N)
    nn_b = np.roll(inds_r, -N)
    nn_r = nn_b.T
    nn_l = nn_t.T
    
    # save results:
    #configs = np.zeros((N, N, tm))
    configs = np.zeros((N, N, int(np.floor((tm-t0)/nth))+1))
    
    # plot the resulting image onto that plot
    
    for t in range(tm-t0):
        for n in range(N**2):
            # pick a random spin site:
            i = np.random.randint(N)
            j = np.random.randint(N)
            # calculate energy factor:
            dE = 2*J*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i,j], j] + lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
            r = np.exp(-dE/T)
            if r > z[0, t*n + n]:
                lat[i, j] *= -1
            # otherwise do nothing
            
        # save configuration from this MCS:
        if t >= t0 and t % nth == 0:
            configs[:, :, int(t/nth)] = lat
    
    return configs



# %%

# set up lattice and variables
N = 512  # keep to powers of 2

J = 1  # isotropic
Tc = 2.2692*J
T = 0.5*Tc  # Using k_{B}=1
t0 = 0
tm = 300

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xticks(np.arange(0, N+1, int(N/5)))
ax.set_yticks(np.arange(0, N+1, int(N/5)))

nth = 5
configs = MC(N, J, T, t0, tm, nth=nth)

# animate it:
for t in range(int(np.floor((tm-t0)/nth))):
    ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
    plt.pause(0.001)


# %%

# plotting snapshots as for report:















