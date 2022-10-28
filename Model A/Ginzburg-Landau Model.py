import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from continuum_solvers import solver

if __name__ == "__main__":
    # Set up lattice
    grid_size = 128
    grid_spacing = 1
    grid = np.random.rand(grid_size,grid_size)*2 - 1
    
    # Time array
    tmax = 10
    num_time_steps = 1024
    t_array = np.linspace(0, tmax, num_time_steps)
    
    # Evolves the lattice over time. "driving" adds the driving term if True
    phi = solver(grid, t_array, grid_size, grid_spacing, driving=True)
    
    # Displaying lattice at t=0
    fig_phi = plt.figure(figsize=(8,6))
    ax_phi = fig_phi.gca()
    ax_phi.set_title("Landau-Ginzburg Equation Evolution\nN="+str(grid_size)+", $t$="+\
                     str(tmax)+" seconds", fontsize=16)
    
    # Selecting a state to look at
    comparison_state = phi[-1]
    
    img_phi = ax_phi.imshow(comparison_state, cmap="Greys", vmin=-1, vmax=1)
    
    cax = fig_phi.add_axes([0.2, 0.1, 0.8, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig_phi.colorbar(img_phi, orientation='vertical')
    
    # Fourier transform of the lattice. No shift to centralise peak
    ft = np.fft.ifftshift(comparison_state)
    ft = np.fft.fft2(ft)
    
    # Finding average for k over a radius
    # Beware that it's slightly asymmetric
    grid_range = np.arange(-32, 32, 1)
    kx, ky = np.meshgrid(grid_range, grid_range)
    
    sum_of_squares = kx**2 + ky**2
    
    average_k = sum_of_squares
    
    fig_ft = plt.figure(figsize=(8,6))
    ax_ft = fig_ft.gca()
    ax_ft.imshow(abs(ft), cmap="Greys", vmin=-1, vmax=1)
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