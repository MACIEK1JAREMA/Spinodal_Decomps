import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Here, we're solving the equation

dphi/dt = grad^2 phi

By discretizing the Laplacian operator, grad^2
'''

def field_function(lattice, t, N, delta_x): 
    # phi_dot is the change in order parameter at each point in the NxN lattice
    # with respect to time
    
    lattice = np.reshape(lattice, (N,N))
    
    phi_right = np.roll(lattice, 1, axis=1)
    phi_left = np.roll(lattice, -1, axis=1)
    phi_above = np.roll(lattice, N)
    phi_below = np.roll(lattice, N)
    
    phi_dot = (phi_right + phi_left + phi_above + phi_below - 4*lattice)/delta_x**2
    
    phi_dot = np.reshape(phi_dot, np.size(phi_dot))
    return phi_dot

if __name__ == "__main__":
    # Creating a lattice with a default spacing of 1 (distance between
    # neighbouring lattice points) with side length grid_size
    grid_size = 3
    grid_spacing = 1
    grid = np.zeros((grid_size,grid_size))
    
    # Setting up the initial conditions
    grid[1,1] = 1
    initial_conditions = np.reshape(grid, np.size(grid))
    
    # Establishing time array to run simulation over
    num_time_steps = 1024
    t_array = np.linspace(0, 1, num_time_steps)
    
    # Using odeint to get solution
    sol = odeint(field_function, initial_conditions, t_array, args=(grid_size, grid_spacing))
    
    # Each row needs to be reshaped back into a 3x3 grid for every time step
    # Needs to be made more efficient if possible
    phi = np.zeros(num_time_steps, dtype=object)
    for i in range(len(phi)):
        phi[i] = np.reshape(sol[i], (grid_size, grid_size))
        
    # Grabbing the centre value for plotting. Can't figure out how to vectorise
    centre_phi = np.zeros(num_time_steps)
    for i, grid in enumerate(phi):
        centre_phi[i] = grid[1,1]
        
    # Plotting centre value of phi
    fig_centre = plt.figure(figsize=(8,6))
    ax_centre = fig_centre.gca()
    ax_centre.plot(t_array, centre_phi)
    ax_centre.set_xlabel("Time $t$", fontsize=16)
    ax_centre.set_ylabel("$\phi_{1,1}(t)$", fontsize=16)
    
    # Displays an image which shows how the system is behaving at a given timestep
    fig = plt.figure()
    ax = fig.gca()
    img = ax.imshow(phi[500], cmap="Greys")
    ax.set_aspect("equal")
    
    cax = fig.add_axes([0.2, 0.1, 0.8, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig.colorbar(img, orientation='vertical')
    plt.show()
    
    '''
    #### Remarks #### 
    1) Maybe perform the maths with a row vector instead of a grid? This would
    vectorise the operations more efficiently by skipping out on the lattice
    reshaping steps in field_function
    
    2) Given the fact we have the grid form of phi, we can create an animation
    from this and t_array, but I'm not sure how to do that yet
        2.5) imshow seems to do a great job of visalising the lattice, so we could 
        use it to create an animation
    '''
    
#%%
# Animation
import matplotlib.animation as animation

if __name__ == "__main__":
    '''
    Code for animation retrieved and adjusted from:
        https://matplotlib.org/stable/gallery/animation/dynamic_image.html
    '''
    
    fig_ani, ax_ani = plt.subplots()
    
    def f(x, y):
        return np.sin(x) + np.cos(y)

    x = np.linspace(0, 2 * np.pi, 120)
    y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(500):
        val = phi[i]
        x += np.pi / 15.
        y += np.pi / 20.
        im = ax_ani.imshow(val, cmap="Greys", animated=True)
        if i == 0:
            print(np.shape(val))
            ax_ani.imshow(val, cmap="Greys")  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig_ani, ims, interval=5, blit=True,
                                    repeat_delay=1000)


