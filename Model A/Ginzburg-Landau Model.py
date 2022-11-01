import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from continuum_solvers import solver

def annulus_average(ft, N, k1, dk):
    half_grid_size = int(N/2)
    grid_range = np.arange(-half_grid_size, half_grid_size, 1)
    kx, ky = np.meshgrid(grid_range, grid_range)
    
    sum_of_squares = kx**2 + ky**2
    
    # Grabbing all Fourier transform values between two radii
    k2 = k1 + dk
    
    indices = np.argwhere((sum_of_squares >= k1**2) & (sum_of_squares < k2**2))
    rows = indices[:,0]
    columns = indices[:,1]
    
    average = np.mean(abs(ft[rows, columns]))
    return average


if __name__ == "__main__":
    # Set up lattice
    grid_size = 128
    grid_spacing = 1
    
    # Time array
    tmax = 50
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    num_repeats = 5
    
    # Variables and array for structure factor
    dk = 1
    kvals = np.arange(0, grid_size, dk)
    interval = 128
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
            
            # Preparing structure factor array
            sf[repeat][i] = np.zeros(len(ft))
            
            # Calculating structure factor
            for j, k in enumerate(kvals):
                sf[repeat][i][j] = annulus_average(ft, grid_size, k, dk)
            
            # Normalising structure factor
            sf[repeat][i] = sf[repeat][i]/sf[repeat][0]
    
    # Averaging structure factor over each repeat and plotting
    # Plot 1 - no time adjustment
    # Plot 2 - Multiplying k by sqrt(t)
    # Plot 3 - Dividing k by sqrt(t)
    averaged_sf = np.mean(sf, axis=0)
    fig_sf, ax_sf = plt.subplots(1, 3, figsize=(16,8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
    fig_sf.suptitle("Structure Factor (averaged over "+str(num_repeats)+" repeats)", fontsize=20)
    ax_sf[0].set_xlabel("$|k|$", fontsize=16)
    ax_sf[0].set_ylabel("S($|k|$)", fontsize=16)
    ax_sf[1].set_xlabel("$|k|t^\\frac{1}{2}$", fontsize=16)
    ax_sf[1].set_ylabel("S($|k|$)$/t^\\frac{3}{2}$", fontsize=16)
    ax_sf[2].set_xlabel("$|k|/t^\\frac{1}{2}$", fontsize=16)
    ax_sf[2].set_ylabel("S($|k|$)$t^\\frac{3}{2}$", fontsize=16)
    
    for time, structure_factor in zip(sf_times[1:], averaged_sf[1:]):
        ax_sf[0].plot(kvals, structure_factor, label="$t$="+str(np.round(time,0)))
        ax_sf[1].plot(kvals*(time**0.5), structure_factor/(time**1.5), label="$t$="+str(np.round(time,0)))
        ax_sf[2].plot(kvals/(time**0.5), structure_factor*(time**1.5), label="$t$="+str(np.round(time,0)))
    ax_sf[0].legend()
    ax_sf[1].legend()
    ax_sf[2].legend()
        
    # Displays the final state
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
    
#%%
    # Calculating and plotting structure factor for various time steps
    fig_sf = plt.figure(figsize=(8,6))
    ax_sf = fig_sf.gca()
    ax_sf.set_xlabel("$k$", fontsize=16)
    ax_sf.set_ylabel("SF($k$)", fontsize=16)
    ax_sf.set_title("Structure Factor", fontsize=18)
    
    # num_curves = 20
    # time_interval = int(np.round(num_time_steps/num_curves, 0))
    # state_interval = int(np.round(len(phi)/num_curves, 0))
    '''Can't quite work out how to do an integer number of curves, so picking 16'''
    time_interval, state_interval = 64, 64
    sf_times = t_array[::time_interval]
    sf_states = phi[::state_interval]
    dk = 1
    kvals = np.arange(0, grid_size, dk)
    sf = np.zeros(len(sf_times), dtype=object)
    
    for i, time_and_state in enumerate(zip(sf_times, sf_states)):
        time = time_and_state[0]
        state = time_and_state[1]
        
        # Fourier transform of the lattice
        ft = np.fft.ifftshift(state)
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        
        # Set up structure factor array
        sf[i] = np.zeros(len(ft))
        
        # Finding average for k over multiple radii
        for j, k in enumerate(kvals):
            sf[i][j] = annulus_average(ft, grid_size, k, dk)
    
    # Plotting structure factor for each time step
    for structure_factor, time in zip(sf[1:], sf_times):
        ax_sf.plot(kvals, structure_factor/sf[0], label="$t$="+str(np.round(time,1)))
    ax_sf.legend(fontsize=12)
        
    '''
    TO-DO
    Need to make this average over a bunch of states now, either wrap the
    above in a for loop or define it as a function
    '''
    
    # Displaying the end state
    end_state = phi[-1]
    fig_phi = plt.figure(figsize=(8,6))
    ax_phi = fig_phi.gca()
    ax_phi.set_title("Landau-Ginzburg Equation Evolution\nN="+str(grid_size)+", $t$="+\
                     str(tmax)+" seconds", fontsize=16)
    
    img_phi = ax_phi.imshow(end_state, cmap="Greys", vmin=-1, vmax=1)
    
    cax = fig_phi.add_axes([0.2, 0.1, 0.8, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig_phi.colorbar(img_phi, orientation='vertical')
    
    '''
    Remarks:
        1) We could do a fun animation of evolving the state in time and
          plotting the structure factor curve at the same time on another plot.
          Just need to do all the calculations and then make the animation after
    '''
    
#%%

if __name__ == "__main__":
    # Animating the time evolution
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
    
#%%

if __name__ == "__main__":
    file_name = str(grid_size)+"_grid_diffusion.gif"
    writer = animation.PillowWriter(fps=15)
    ani.save(file_name, writer=writer)