# First Monte Carlo SImulation
# If time allows, later to be optimized and modularised

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def MC(N, J, T, t0, tm, nth=1):
    """
    Function to compute Monte Carlo simulation of 2D Ising Model
    
    Params:
      ---------------------------
     -- N - int64 - size of lattice side
     -- J - int64 - sisotropic coupling constant
     -- T - int64 - temperature, remembering that Tc = 2.2692*J
     -- t0 - int64 - MCS to start saving from to exclude init conditions
     -- tm - int64 - max MCS
     -- nth - int64 - optional - every nth MCS to save
    
    Returns:
        ----------------------
    -- configs - np.ndarray - size [N x N x ((t0-tm)/nth)] for lattice at each
                            desired MSC. Includes intiial random set up at
                            step 0, so array is 1 larger than expected from t0
                            tm and nth. The lattice as is after MCS number 4
                            is saved in column with index 4
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
    configs[:, :, 0] = lat
    
    # plot the resulting image onto that plot
    
    for t in range(1, tm-t0+1):
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
T = 0.1*Tc  # Using k_{B}=1
t0 = 0
tm = 50

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xticks(np.arange(0, N+1, int(N/5)))
ax.set_yticks(np.arange(0, N+1, int(N/5)))

nth = 2
configs = MC(N, J, T, t0, tm, nth=nth)

# animate it:
for t in range(int(np.floor((tm-t0)/nth))):
    ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
    plt.pause(0.001)


# %%

# plotting snapshots as for report:

# evolve system:

N = 512
J = 1
Tc = 2.2692*J
T = 0.5*Tc
t0 = 0
tm = 200

nth = 10
configs = MC(N, J, T, t0, tm, nth=nth)


# set up visuals

fig = plt.figure(figsize=(15, 4))
fig.subplots_adjust(right=0.942, left=0.092, top=0.885, bottom=0.117, hspace=0.301, wspace=0.171)
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)

ax1.tick_params(labelsize=22)
ax1.set_title(r"$t=0$", fontsize=22)
ax1.set_ylabel(r'$pixel$', fontsize=22)
#ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
#ax1.yaxis.set_major_locator(plt.MaxNLocator(4))

ax2.tick_params(labelsize=22)
ax2.set_title(r"$t=10$", fontsize=22)
#ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

ax3.tick_params(labelsize=22)
ax3.set_title(r"$t=50$", fontsize=22)
#ax3.xaxis.set_major_locator(plt.MaxNLocator(4))

ax4.tick_params(labelsize=22)
ax4.set_title(r"$t=200$", fontsize=22)
#ax4.xaxis.set_major_locator(plt.MaxNLocator(4))


# plot snapshots
i = 1
for t in [0, 1, 5, 20]:
    exec("ax" + str(i) + ".imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)")
    i += 1

