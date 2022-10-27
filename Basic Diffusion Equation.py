import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Here, we're solving the equation

dphi/dt = grad^2 phi

By discretizing the Laplacian operator, grad^2
'''

def field_function(lattice, t, N, grid_spacing): 
    # phi_dot is the change in order parameter at each point in the NxN lattice
    # with respect to time
    
    phi_right = np.roll(lattice, 1, axis=1)
    phi_left = np.roll(lattice, -1, axis=1)
    phi_above = np.roll(lattice, N)
    phi_below = np.roll(lattice, N)
    
    phi_dot = (phi_right + phi_left + phi_above + phi_below - 4*lattice)/grid_spacing**2
    return phi_dot

if __name__ == "__main__":
    # Creating an NxN lattice with a default spacing of 1 (distance between
    # neighbouring lattice points)
    grid_size = 3 
    grid_spacing = 1
    grid = np.zeros((grid_size,grid_size))
    
    grid[1,1] = 1
    
    initial_conditions = grid
    t_array = np.linspace(0, 1, 1024)
    
    


