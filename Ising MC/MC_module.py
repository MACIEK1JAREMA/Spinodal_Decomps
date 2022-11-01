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
     -- J - float64 - isotropic coupling constant
     -- T - float64 - temperature, remembering that Tc = 2.2692*J
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


@jit(nopython=True)
def MC_ani(N, Jx, Jy, T, t0, tm, nth=1, seed=None):
    """
    Function to compute Monte Carlo simulation of 2D Ising Model with anisotropy
    
    Params:
      ---------------------------
     -- N - int64 - size of lattice side
     -- Jx - float64 - x direction coupling strength
     -- Jy - float64 - y direction coupling strength
     -- T - float64 - temperature, remembering that Tc = 2.2692*J
     -- t0 - int64 - MCS to start saving from to exclude init conditions
     -- tm - int64 - max MCS
     -- nth - int64 - optional - every nth MCS to save
     -- seed - int64 - optional - seed for random initial config.
    
    Returns:
        ----------------------
    -- configs - np.ndarray - size [N x N x ((t0-tm)/nth)] for lattice at each
                            desired MSC. Includes intiial random set up at
                            step 0, so array is 1 larger than expected from t0
                            tm and nth. The lattice as is after MCS number 4
                            is saved in column with index 4
    """
    
    if seed is not None:
        np.random.seed(seed)
    
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
            dEx = 2*Jx*lat[i, j]*( lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
            dEy = 2*Jy*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i, j], j] )
            dE = dEx + dEy
            r = np.exp(-dE/T)
            if r > z[0, (t-t0)*n + n]:
                lat[i, j] *= -1
            # otherwise do nothing
        # save configuration from this MCS:
        if t >= t0 and (t-t0) % nth == 0:
            configs[:, :, int((t-t0)/nth)+1] = lat
    
    return configs


@jit(nopython=True)
def MC_frust(N, Jnn, Jnnn, T, t0, tm, nth=1, seed=None):
    """
    Function to compute Monte Carlo simulation of 2D Ising Model with frustration
    
    Params:
      ---------------------------
     -- N - int64 - size of lattice side
     -- Jnn - float64 - nn interaction J (>0)
     -- Jnnn - float64 - nnn interaction (<0 or opp to Jnn for frsutration)
     -- T - float64 - temperature, remembering that Tc = 2.2692*J
     -- t0 - int64 - MCS to start saving from to exclude init conditions
     -- tm - int64 - max MCS
     -- nth - int64 - optional - every nth MCS to save
     -- seed - int64 - optional - seed for random initial config.
    
    Returns:
        ----------------------
    -- configs - np.ndarray - size [N x N x ((t0-tm)/nth)] for lattice at each
                            desired MSC. Includes intiial random set up at
                            step 0, so array is 1 larger than expected from t0
                            tm and nth. The lattice as is after MCS number 4
                            is saved in column with index 4
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if np.sign(Jnn) == np.sign(Jnnn):
        print("Warning - Jnn not opp sign to Jnnn so model is not frustrated")
        # not actual warning raised as numba interferes
    
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
            dEnn = 2*Jnn*lat[i, j]*( lat[nn_t[i, j], j] + lat[nn_b[i,j], j] + lat[i, nn_r[i, j]] + lat[i, nn_l[i, j]] )
            dEnnn = 2*Jnnn*lat[i, j]*( lat[nn_t[i, j], nn_r[i, j]] + lat[nn_t[i,j], nn_l[i,j]] + lat[nn_b[i, j], nn_r[i, j]] + lat[nn_b[i, j], nn_l[i, j]] )
            dE = dEnn + dEnnn
            r = np.exp(-dE/T)
            if r > z[0, (t-t0)*n + n]:
                lat[i, j] *= -1
            # otherwise do nothing
        # save configuration from this MCS:
        if t >= t0 and (t-t0) % nth == 0:
            configs[:, :, int((t-t0)/nth)+1] = lat
    
    return configs


def annulus_avg(ft, N, dk):
    """
    Finds average of function from fourier transform in k space
    in circles
    
    parameters:
        ------------------
    -- ft - np.ndarray - 2D function in kspace
    -- N - int - number of real space points from it
    -- dk - int - k space step
    """
    
    kvals = np.arange(0, N, dk)
    average = np.zeros(len(kvals))
    
    for j, k in enumerate(kvals):
    
        # prepare axes
        N_half = int(N/2)
        axes = np.arange(-N_half, N_half, 1)
        kx, ky = np.meshgrid(axes, axes)
        
        # get square radius
        dist_sq = kx**2 + ky**2
        
        # Get all values in [k, k+dk]
        indices = np.argwhere((dist_sq >= k**2) & (dist_sq < (k+dk)**2))
        rows = indices[:, 0]
        columns = indices[:, 1]
        
        # find average of all of those up to cutoff frequency
        if len(indices) != 0:
            average[j] = np.mean(abs(ft[rows, columns]))
        else:
            k_max = k
            break
    
    # cut data upto k_max only, average stays same size with zeros after
    # cut off k
    kvals = kvals[:k_max]
    
    return average, kvals, k_max


def Sk_MCrun(N, J, T, dk, t0, tm, nth):
    """
    
    """
    
    configs = MC(N, J, T, t0, tm, nth=nth)
    
    # Calculating structure factor for various time steps
    k_num = len(np.arange(0, N, dk))
    kmax = np.zeros((len(configs[0, 0, :])))
    average = np.zeros((k_num, len(configs[0, 0, :])))
    for i in range(len(configs[0, 0, :])):
        # Lattice spins FT
        ft = np.fft.ifftshift(configs[:, :, i])
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        # Finding average for k over multiple radii
        av, kvals, km = annulus_avg(ft*np.conj(ft), N, dk)
        kmax[i] = km
        
        # add zeros to average so that it retains right shape:
        avg = np.append(av, np.zeros(( len(average[:, 0]) - len(av) )))
        average[:, i] = avg
    
    return average, kmax


def Sk_MCrun_ani(N, Jx, Jy, T, dk, t0, tm, nth):
    """
    """
    
    configs = MC_ani(N, Jx, Jy, T, t0, tm, nth=nth)
    
    # Calculating structure factor for various time steps
    k_num = len(np.arange(0, N, dk))
    kmax = np.zeros((len(configs[0, 0, :])))
    average = np.zeros((k_num, len(configs[0, 0, :])))
    for i in range(len(configs[0, 0, :])):
        # Lattice spins FT
        ft = np.fft.ifftshift(configs[:, :, i])
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        # Finding average for k over multiple radii
        av, kvals, km = annulus_avg(ft*np.conj(ft), N, dk)
        kmax[i] = km
        
        # add zeros to average so that it retains right shape:
        avg = np.append(av, np.zeros(( len(average[:, 0]) - len(av) )))
        average[:, i] = avg
    
    return average, kmax


def Sk_MCrun_frust(N, Jnn, Jnnn, T, dk, t0, tm, nth):
    """
    """
    
    configs = MC_frust(N, Jnn, Jnnn, T, t0, tm, nth=nth)
    
    # Calculating structure factor for various time steps
    k_num = len(np.arange(0, N, dk))
    kmax = np.zeros((len(configs[0, 0, :])))
    average = np.zeros((k_num, len(configs[0, 0, :])))
    for i in range(len(configs[0, 0, :])):
        # Lattice spins FT
        ft = np.fft.ifftshift(configs[:, :, i])
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        # Finding average for k over multiple radii
        av, kvals, km = annulus_avg(ft*np.conj(ft), N, dk)
        kmax[i] = km
        
        # add zeros to average so that it retains right shape:
        avg = np.append(av, np.zeros(( len(average[:, 0]) - len(av) )))
        average[:, i] = avg
    
    return average, kmax


# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 8  
    
    J = 1
    Tc = 2.2692*J
    T = 0.1*Tc
    t0 = 1  # at least 1 (0th saved automatically anyway!)
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

# %%
        
# anisotropic code:

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 256
    
    Jx = 10
    Jy = 0.1
#    Jx = 1
#    Jy = 1
    Tc = 2.2692*np.sqrt(Jx**2 + Jy**2)  # just as zeroth order approx here.
    T = 0.01*Tc
    t0 = 1
    tm = 150
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax.tick_params(labelsize=16)
    ax.set_xticks(np.arange(0, N+1, int(N/5)))
    ax.set_yticks(np.arange(0, N+1, int(N/5)))
    
    nth = 5
    configs = MC_ani(N, Jx, Jy, T, t0, tm, nth=nth, seed=1314)
    
    # animate it:
    for t in range(len(configs[0, 0, :])):
        ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
        plt.pause(0.01)

# %%

# fristrated code:

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 256
    
    Jnn = 1
    Jnnn = -1
    
    Tc = 2.2692*np.sqrt(Jnn**2 + Jnnn**2)  # just as zeroth order approx here.
    T = 0.01*Tc
    t0 = 1
    tm = 150
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax.tick_params(labelsize=16)
    ax.set_xticks(np.arange(0, N+1, int(N/5)))
    ax.set_yticks(np.arange(0, N+1, int(N/5)))
    
    nth = 5
    configs = MC_frust(N, Jnn, Jnnn, T, t0, tm, nth=nth, seed=1314)
    
    # animate it:
    for t in range(len(configs[0, 0, :])):
        ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
        plt.pause(0.01)

# %%

# TEST of anulkus average with kmax included


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # set up lattice and variables
    N = 128
    
    J = 1
    Tc = 2.2692*J
    T = 0.1*Tc
    t0 = 5  # at least 1 (0th saved automatically anyway!)
    tm = 90
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax.tick_params(labelsize=16)
    ax.set_xticks(np.arange(0, N+1, int(N/5)))
    ax.set_yticks(np.arange(0, N+1, int(N/5)))
    
    nth = 15
    configs = MC(N, J, T, t0, tm, nth=nth)
    
    # animate it:
    for t in range(len(configs[0, 0, :])):
        print('showing '+ str(t))
        ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
        plt.pause(0.5)
    
    # Lattice spins FT
    ft = np.fft.ifftshift(configs[:, :, -3])
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    dk = 1
    # Finding average for k over multiple radii
    av, kvals, kmax = annulus_avg(ft*np.conj(ft), N, dk)




