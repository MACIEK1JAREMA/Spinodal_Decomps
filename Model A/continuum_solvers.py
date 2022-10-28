import numpy as np
from scipy.integrate import odeint

def diffusion_eqn(lattice, t, N, delta_x): 
    # phi_dot is the change in order parameter at each point in the NxN lattice
    # with respect to time
    
    lattice = np.reshape(lattice, (N,N))
    
    phi_right = np.roll(lattice, 1, axis=1)
    phi_left = np.roll(lattice, -1, axis=1)
    phi_above = np.roll(lattice, N)
    phi_below = np.roll(lattice, -N)
    
    phi_dot = (phi_right + phi_left + phi_above + phi_below - 4*lattice)/delta_x**2
    
    phi_dot = np.reshape(phi_dot, np.size(phi_dot))
    return phi_dot

def ginzburg_landau_eqn(lattice, t, N, delta_x): 
    # phi_dot is the change in order parameter at each point in the NxN lattice
    # with respect to time
    
    lattice = np.reshape(lattice, (N,N))
    
    phi_right = np.roll(lattice, 1, axis=1)
    phi_left = np.roll(lattice, -1, axis=1)
    phi_above = np.roll(lattice, N)
    phi_below = np.roll(lattice, -N)
    
    phi_dot = (phi_right + phi_left + phi_above + phi_below - 4*lattice)/delta_x**2 +\
        lattice * (1-lattice * lattice)
    
    phi_dot = np.reshape(phi_dot, np.size(phi_dot))
    return phi_dot

def solver(lattice, t, N, delta_x, driving=False):
    initial_conditions = np.reshape(lattice, np.size(lattice))
    if driving == False:
        func = diffusion_eqn
    else:
        func = ginzburg_landau_eqn        
    sol = odeint(func, initial_conditions, t, args=(N, delta_x))
    
    phi = np.zeros(len(t), dtype=object)
    for i in range(len(phi)):
        phi[i] = np.reshape(sol[i], (N,N))
        
    return phi