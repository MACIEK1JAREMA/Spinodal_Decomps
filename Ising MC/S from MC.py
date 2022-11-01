# Extracting structure factor plots in 2D k space
# and circularly avergaed ones from MC of 2D ISING

import numpy as np
from MC_module import MC as solve_MC
import matplotlib.pyplot as plt


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


# %%


# set up lattice and variables
N = 512

J = 1
Tc = 2.2692*J
T = 0.1*Tc
t0 = 5
tm = 90

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.tick_params(labelsize=16)
ax.set_xticks(np.arange(0, N+1, int(N/5)))
ax.set_yticks(np.arange(0, N+1, int(N/5)))

nth = 15
configs = solve_MC(N, J, T, t0, tm, nth=nth)

# animate it:
for t in range(len(configs[0, 0, :])):
    ax.imshow(configs[:, :, t], cmap='Greys', vmin=-1, vmax=1, interpolation=None)
    plt.pause(0.5)

# Calculating and plotting structure factor for various time steps
figS = plt.figure(figsize=(8,6))
axS = figS.gca()
axS.tick_params(labelsize=22)
axS.set_xlabel(r"$k$", fontsize=22)
axS.set_ylabel(r"S($k$)", fontsize=22)


# Same but normalise it to S(k, 0)
figSn = plt.figure(figsize=(8,6))
axSn = figSn.gca()
axSn.tick_params(labelsize=22)
axSn.set_xlabel(r"$k$", fontsize=22)
axSn.set_ylabel(r"S($k$)$/$S($k$)$|_{t_{0}}$", fontsize=22)


for i in range(len(configs[0, 0, :])):
    # Lattice spins FT
    ft = np.fft.ifftshift(configs[:, :, i])
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    
    # Finding average for k over multiple radii
    dk = 1
    average, kvals, kmax = annulus_avg(ft*np.conj(ft), N, dk)
    average = average[:kmax]
    
    # plot for all different MCS
    if i == 0:
        axS.plot(kvals, average, label=r"$t=0$ MCS")
    elif i == 1:  # at t0
        avg0 = average
        axS.plot(kvals, average, label=r"$t=t_{0}=$"+ str(t0) + " MCS")
    else:
        time = str(int(nth*(i-1) + t0)) + " MCS"
        axS.plot(kvals, average, label=r"$t=$"+time)
        axSn.plot(kvals, average/avg0, label=r"$t=$"+time)

axS.legend(fontsize=22)
axSn.legend(fontsize=22)

