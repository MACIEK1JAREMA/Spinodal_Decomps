# Finding excess energy in Ising systems

import numpy as np
from numba import jit


@jit(nopython=True)
def MC_DE(N, J, T, t0, tm, nth=1):
    """
    Function to simulate Monte Carlo simulation of 2D Ising Model
    while tracking energy.
    
    Params:
      ---------------------------
     -- N - int64 - size of lattice side
     -- J - float64 - isotropic coupling constant
     -- T - float64 - temperature, remembering that Tc = 2.2692*J
     -- t0 - int64 - MCS to start saving from to exclude init conditions
     -- tm - int64 - max MCS
     -- nth - int64 - optional - every nth MCS to save
    
    Returns:
        ----------------------
    -- E_tot -- np.ndarray - array of total system energy at
                                saved MCS (nth in t0 to tm)
    -- times -- np.ndarray - times at which the system excess energy was saved.
    """
    
    # generate lattice
    lat = np.random.rand(N, N)
    lat = np.sign(lat - 0.5)
    
    
    num = int(np.floor((tm-t0)/nth)) + 1  # number of MCS to complete:
    z = np.random.rand(1, tm*N**2)  # random number for attempt of each MCS
    
    # set up nn index array and it's rolled over ones for nn finding
    inds = np.arange(0, N, 1)
    inds_c = np.ones((N,1), dtype=np.int64)*inds
    inds_r = inds_c.T
    
    nn_t = np.roll(inds_r, N)
    nn_b = np.roll(inds_r, -N)
    nn_r = nn_b.T
    nn_l = nn_t.T
    
    # find it's total energy to begin with:
    #E_init = J*(np.sum( lat * (lat[nn_t, :] + lat[nn_b, :] + lat[:, nn_r] + lat[:, nn_l] )))
    
    for i in range(N):
        for j in range(N):
            E_init = J*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i,j], j] + lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
    
    # perp arrays to save results:
    E_tot = np.zeros((num+1))
    E_tot[0] = E_init
    times = np.zeros((num+1))  # times that end up being saved
    
    # look over all MCS times with N^2 attempts of flip on each run
    for t in range(1, tm+1):
        E_change = 0
        for n in range(N**2):
            # pick a random spin site:
            i = np.random.randint(N)
            j = np.random.randint(N)
            # calculate energy factor:
            dE = 2*J*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i,j], j] + lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
            r = np.exp(-dE/T)
            if r > z[0, (t-t0)*n + n]:
                lat[i, j] *= -1
                E_change -=  dE # update total energy
            # otherwise do nothing
        
        # save energy from this MCS:
        if t >= t0 and (t-t0) % nth == 0:
            E_tot[int((t-t0)/nth)+1] = E_tot[int((t-t0)/nth)] + E_change
            times[int((t-t0)/nth)+1] = t
    
    
    return E_tot, times

# %%


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 256
    
    J = 1
    Tc = 2.2692*J
    T = 0.1*Tc
    n_nths = 12
    reps = 10
    
    t0 = int(N/10)
    tm = int(N)
    nth = int((tm-t0)/n_nths)
    
    Energies = np.zeros((int(np.floor((tm-t0)/nth))+2))
    
    for r in range(reps):
        Es, times = MC_DE(N, J, T, t0, tm, nth=nth)
        Energies += Es
        print("Done repeat " + str(r))
    
    Energies /= reps
    excess = 2*J - Energies/(N**2)
    
    # plot it:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax.tick_params(labelsize=22)
    ax.set_xlabel(r"$t \ [MCS]$", fontsize=22)
    ax.set_ylabel(r"$\frac{\Delta E}{J}$", fontsize=22)
    
    ax.plot(times, excess/J)
    
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.gca()
    ax1.tick_params(labelsize=22)
    ax1.set_xlabel(r"$t^{-1/2} \ [MCS]$", fontsize=22)
    ax1.set_ylabel(r"$\frac{\Delta E}{J}$", fontsize=22)
    
    ax1.plot(times**(-0.5), excess/J)
    
