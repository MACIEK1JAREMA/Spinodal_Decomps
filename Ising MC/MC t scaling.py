# time scale testing of MC code
# MC module

import numpy as np
from numba import jit
import time #import time module

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


def annulus_avg(ft, N, dk):
    """
    Finds average of 
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
        rows = indices[:,0]
        columns = indices[:,1]
        
        # find average of all of those
        average[j] = np.mean(abs(ft[rows, columns]))
    
    return average, kvals


def Sk_MCrun(N, J, T, dk, t0, tm, nth):
    """
    """
    
    configs = MC(N, J, T, t0, tm, nth=nth)
    
    # Calculating structure factor for various time steps
    k_num = len(np.arange(0, N, dk))
    average = np.zeros((k_num, len(configs[0, 0, :])))
    for i in range(len(configs[0, 0, :])):
        # Lattice spins FT
        ft = np.fft.ifftshift(configs[:, :, i])
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        # Finding average for k over multiple radii
        av, _ = annulus_avg(ft, N, dk)
        average[:, i] = av
    
    return average

# %%
N = 9#initial value for N
Nfinal = 100
sizearray = np.array([])
Narray = np.array([])
while N <= Nfinal:
    if __name__ == "__main__":
        st = time.time() #start time
        import matplotlib.pyplot as plt
        # set up lattice and variables
        N = N  
    
        J = 1
        Tc = 2.2692*J
        T = 0.1*Tc
        t0 = 1
        tm = 20
     
        #fig = plt.figure(figsize=(10, 7))
        #ax = fig.gca()
        #ax.tick_params(labelsize=16)
        #ax.set_xticks(np.arange(0, N+1, int(N/5)))
        #ax.set_yticks(np.arange(0, N+1, int(N/5)))
    
        nth = 2
        configs = MC(N, J, T, t0, tm, nth=nth)
    
        """
        Displays initial condition t=0, then when (t-t0) is a multiple of nth
        t=1
        t=3
        t=5
        """
        et = time.time() #end time
        elapsed_time = et - st
        
        sizearray = np.append(sizearray,elapsed_time)
        Narray = np.append(Narray,N)
        print('Execution time:', elapsed_time, 'seconds')
        # animate it:
        #for t in range(len(configs[0, 0, :])):
        #    print('showing '+ str(t))
        #    ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
        #    plt.pause(0.5)
        N = N + 2

#Narray[924] = 4
#sizearray[924] =4        

#Narray[916] = 4
#sizearray[916] =4   

plt.plot(Narray,sizearray)
plt.title("How does run time scale with lattice size")
plt.xlabel("N")
plt.ylabel("Runtime(seconds")