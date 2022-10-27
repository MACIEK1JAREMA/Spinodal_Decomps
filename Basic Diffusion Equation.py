import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Here, we're solving the equation

dphi/dt = grad^2 phi

By discretizing the Laplacian operator, grad^2
'''

def field_function(initial_conditions, t):
    laplacian = ()
    pass    

if __name__ == "__main__":
    # Creating an NxN lattice with a default spacing of 1 (distance between
    # neighbouring lattice points)
    N = 3 
    spacing = 1
    grid = np.ones(N)


