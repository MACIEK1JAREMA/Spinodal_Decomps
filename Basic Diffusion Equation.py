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
        centre_grid[i] = grid[1,1]
        

    fig_centre = plt.figure(figsize=(8,8))
    ax_centre = fig_centre.gca()
    # ax_centre.plot(t_array, phi)
    
    '''
    #### Remarks #### 
    1) Maybe perform the maths with a row vector instead of a grid? This would
    vectorise the operations more efficiently by skipping out on the lattice
    reshaping steps in field_function
    
    2) Given the fact we have the grid form of phi, we can create an animation
    from this and t_array, but I'm not sure how to do that yet
    '''
    
    


