# MC module

import numpy as np
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
    
    # number of MCS to complete:
    num = int(np.floor((tm-t0)/nth)) + 1
    
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
    configs = np.zeros((N, N, num+1))
    configs[:, :, 0] = lat
    
    # plot the resulting image onto that plot
    
    for t in range(1, tm+1):
        for n in range(N**2):
            # pick a random spin site:
            i = np.random.randint(N)
            j = np.random.randint(N)
            # calculate energy factor:
            dE = 2*J*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i,j], j] + lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
            r = np.exp(-dE/T)
            if r > z[0, (t-t0)*n + n]:
                lat[i, j] *= -1
            # otherwise do nothing
        # save configuration from this MCS:
        if t >= t0 and (t-t0) % nth == 0:
            configs[:, :, int(t/nth)+1] = lat
    
    return configs


# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 8  
    
    J = 1
    Tc = 2.2692*J
    T = 0.1*Tc
    t0 = 1
    tm = 5
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax.tick_params(labelsize=16)
    ax.set_xticks(np.arange(0, N+1, int(N/5)))
    ax.set_yticks(np.arange(0, N+1, int(N/5)))
    
    nth = 2
    configs = MC(N, J, T, t0, tm, nth=nth)
    
    """
    Displays initial condition t=0, then when (t-t0) is a multiple of nth
    t=1
    t=3
    t=5
    """
    
    # animate it:
    for t in range(len(configs[0, 0, :])):
        print('showing '+ str(t))
        ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
        plt.pause(0.5)
