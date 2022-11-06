import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from continuum_solvers import solver
#%%

def annulus_average(ft, N, k1, dk):
    half_grid_size = int(N/2)
    grid_range = np.arange(-half_grid_size, half_grid_size, 1)
    kx, ky = np.meshgrid(grid_range, grid_range)
    
    sum_of_squares = kx**2 + ky**2
    
    # Grabbing all |Fourier transform|^2 values between two radii
    k2 = k1 + dk
    
    indices = np.argwhere((sum_of_squares >= k1**2) & (sum_of_squares < k2**2))
    rows = indices[:,0]
    columns = indices[:,1]
    
    sf = ft * np.conj(ft) # Square magnitude

    average = np.mean(sf[rows, columns]) # returns slope ~ 0.66
    # average = np.sum(k1*sf[rows, columns])/np.sum(sf[rows, columns]) # returns slope ~0
    return average

def sf_calculator(grid_size, grid_spacing, dk, t_array, num_repeats):

    # Variables and array for structure factor
    dk = 1
    # k ranges from 1 to N/2
    kvals = np.arange(1, int(grid_size/2), dk)
    interval = 171
    sf_times = t_array[::interval]
    sf = np.zeros((num_repeats, len(sf_times)), dtype=object)
    
    for repeat in range(num_repeats):
        print("Repeat number "+str(repeat+1))
        grid = np.random.rand(grid_size,grid_size)*2 - 1
        
        # Evolves the lattice over time. "driving" adds the driving term if True
        phi = solver(grid, t_array, grid_size, grid_spacing, driving=True)
        
        # States to analyse structure factor for
        sf_states = phi[::interval]
        
        for i, time_and_state in enumerate(zip(sf_times, sf_states)):
            time = time_and_state[0]
            state = time_and_state[1]
            
            # Fourier transform of the lattice
            ft = np.fft.ifftshift(state)
            ft = np.fft.fft2(ft)
            ft = np.fft.fftshift(ft)
            
            # Preparing structure factor array for each value of k
            sf[repeat][i] = np.zeros(len(kvals))
            
            # Calculating structure factor
            for j, k in enumerate(kvals):
                sf[repeat][i][j] = annulus_average(ft, grid_size, k, dk)
    
    # Averaging structure factor over each repeat and plotting
    averaged_sf = np.mean(sf, axis=0)
    return sf_times, averaged_sf, kvals

#%%
# Structure factor calculations

if __name__ == "__main__":
    # Set up lattice
    grid_size = 512
    grid_spacing = 1
    
    # Time array
    tmax = 200
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    num_repeats = 10
    dk = 1
    
    sf_times, averaged_sf, kvals = sf_calculator(grid_size, grid_spacing, dk, t_array, num_repeats)

    fig_sf = plt.figure(figsize=(8,6))
    ax_sf = fig_sf.gca()
    ax_sf.tick_params(labelsize=22)
    ax_sf.set_xlabel(r"$k$", fontsize=22)
    ax_sf.set_ylabel(r"S($k$)$/$S($k$)$|_{t_{0}}$", fontsize=22)
        
    L = []
    for time, structure_factor in zip(sf_times[1:], averaged_sf[1:]):
        
        structure_factor = structure_factor/averaged_sf[0]
        k = 2*np.pi*np.sum(structure_factor*kvals**2*dk)/np.sum(structure_factor*kvals*dk)
        L.append(2*np.pi/k)
        # ax_sf.plot(kvals*time**0.5, structure_factor/time, label="$t$="+str(np.round(time,0)))#
        ax_sf.plot(kvals, structure_factor, label="$t$="+str(np.round(time,0)))
    ax_sf.legend(fontsize=22)
    
    
#%%
# Finding gradient of SF(k=0) vs time

if __name__ == "__main__":
    # Wrong size broadcasting
    
    log_time = np.log(sf_times[1:])
    log_l = np.log(np.array(L))
    
    fig_log = plt.figure(figsize=(8,6))
    ax_log = fig_log.gca()
    ax_log.plot(log_time, log_l, "kx")
    ax_log.set_xlabel("$\\log(t)$", fontsize=16)
    ax_log.set_ylabel("$\\langle k(t) \\rangle$", fontsize=16)
    
    
#%%
# Displays various states

if __name__ == "__main__":
    # Set up lattice
    grid_size = 2
    grid_spacing = 1
    grid = np.random.rand(grid_size, grid_size)*2 -1
    
    # Time array
    tmax = 200
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    phi = solver(grid, t_array, grid_size, grid_spacing, driving=True)
    
    fig_phi = plt.figure(figsize=(8,6))
    ax_phi = fig_phi.gca()
    ax_phi.set_title("Landau-Ginzburg Equation Evolution\nN="+str(grid_size)+", $t$="+\
                     str(tmax)+" seconds", fontsize=16)
    
    img_phi = ax_phi.imshow(phi[-1], cmap="Greys", vmin=-1, vmax=1)
    
    cax = fig_phi.add_axes([0.2, 0.1, 0.8, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig_phi.colorbar(img_phi, orientation='vertical')
    plt.show()
    
    # Snapshots for report
    
    phi_0 = phi[np.min(np.argwhere(t_array >=0))]
    phi_10 = phi[np.min(np.argwhere(t_array >=10))]
    phi_50 = phi[np.min(np.argwhere(t_array >=50))]
    phi_200 = phi[np.min(np.argwhere(t_array >=200))]
    
    fig_snapshots, ax_snapshots = plt.subplots(1,4, figsize=(15,4))
    fig_snapshots.subplots_adjust(right=0.942, left=0.092, top=0.885, bottom=0.117, hspace=0.301, wspace=0.071)
    
    for ax in ax_snapshots:
        ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(labelsize=22)
    
    ax_snapshots[0].imshow(phi_0, cmap="Greys")
    ax_snapshots[0].set_title("$t=0$", fontsize=20)
    ax_snapshots[0].set_ylabel("$pixel$", fontsize=22)
    ax_snapshots[0].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_snapshots[0].yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax_snapshots[1].imshow(phi_10, cmap="Greys", vmin=-1, vmax=1)
    ax_snapshots[1].set_title("$t=10$", fontsize=20)
    ax_snapshots[1].set_yticks([])
    ax_snapshots[1].xaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax_snapshots[2].imshow(phi_50, cmap="Greys", vmin=-1, vmax=1)
    ax_snapshots[2].set_title("$t=50$", fontsize=20)
    ax_snapshots[2].set_yticks([])
    ax_snapshots[2].xaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax_snapshots[3].imshow(phi_200, cmap="Greys", vmin=-1, vmax=1)
    ax_snapshots[3].set_title("$t=200$", fontsize=20)
    ax_snapshots[3].set_yticks([])
    ax_snapshots[3].xaxis.set_major_locator(plt.MaxNLocator(4))
    
#%%
# Time taken to run simulations VS lattice size
import time
if __name__ == "__main__":
    time_to_run = np.zeros(9)
    size_array = 2**np.arange(1, 10, 1)
    
    for i, grid_size in enumerate(size_array):
        print("Running grid size = "+str(grid_size))
        start = time.time()
        # Set up lattice
        grid_spacing = 1
        grid = np.random.rand(grid_size, grid_size)*2 -1
        
        # Time array
        tmax = 20
        num_time_steps = 1024
        t_array = np.linspace(0, tmax, num_time_steps)
        
        phi = solver(grid, t_array, grid_size, grid_spacing, driving=True)
        time_to_run[i] = time.time() - start
    
    fig_time = plt.figure(figsize=(10,7))
    ax_time = fig_time.gca()
    ax_time.set_xlabel("$N$", fontsize=22)
    ax_time.set_ylabel("$t [s]$", fontsize=22)
    ax_time.plot(size_array, time_to_run)
    
    fig_squared = plt.figure(figsize=(10,7))
    ax_squared = fig_squared.gca()
    ax_squared.set_xlabel("$N^{2}$", fontsize=22)
    ax_squared.set_ylabel("$t [s]$", fontsize=22)
    ax_squared.plot(size_array**2, time_to_run)
        
#%%
# Time taken to calculate structure factor and run the simulation
import time
if __name__ == "__main__":
    # Set up lattice
    grid_size = 512
    grid_spacing = 1
    
    # Time array
    tmax = 100
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    num_repeats = 1
    dk = 1
    time_to_run = np.zeros(9)
    size_array = 2**np.arange(1, 10, 1)
    
    for i, grid_size in enumerate(size_array):
        print("Running grid size = "+str(grid_size))
        start = time.time()
        sf_times, averaged_sf, kvals = sf_calculator(grid_size, grid_spacing, dk, t_array, num_repeats)
        time_to_run[i] = time.time() - start
    
    fig_time = plt.figure(figsize=(10,7))
    ax_time = fig_time.gca()
    ax_time.tick_params(labelsize=22)
    ax_time.set_xlabel("$N$", fontsize=22)
    ax_time.set_ylabel("$t [s]$", fontsize=22)
    ax_time.plot(size_array, time_to_run)
    
    fig_squared = plt.figure(figsize=(10,7))
    ax_squared = fig_squared.gca()
    ax_squared.tick_params(labelsize=22)
    ax_squared.set_xlabel("$N^{2}$", fontsize=22)
    ax_squared.set_ylabel("$t [s]$", fontsize=22)
    ax_squared.plot(size_array**2, time_to_run)

#%%
# Animating and saving the time evolution

if __name__ == "__main__":
    grid_size = 512
    grid_spacing = 1
    grid = np.random.rand(grid_size, grid_size)*2 -1
    
    # Time array
    tmax = 20
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    phi = solver(grid, t_array, grid_size, grid_spacing, driving=True)
    
    fig_ani, ax_ani = plt.subplots()

    ims = []
    for i in range(num_time_steps):
        im = ax_ani.imshow(phi[i], cmap="Greys", vmin=-1, vmax=1, animated=True)
        if i == 0:
            ax_ani.imshow(phi[i], cmap="Greys", vmin=-1, vmax=1)  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig_ani, ims, interval=5, blit=True,
                                    repeat_delay=10)
    
    file_name = str(grid_size)+"_grid_diffusion.gif"
    writer = animation.PillowWriter(fps=15)
    ani.save(file_name, writer=writer)
