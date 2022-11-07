import numpy as np
import matplotlib.pyplot as plt

# data was made with 512x512, 10 repeats, and 16 time steps (including t=0)
# simulation was run until tmax=200

# number of time steps (16 inclind t=0)
num_time_steps = 16

# preparing array of arrays, where each element is S(k) for all k values at
# a given time step
# so first element of averaged_sf is S(k,0)
# second element is S(k, t1) etc.
# so, to retrieve the 4th S(k,t), you use:
#   averaged_sf[3]

averaged_sf = np.zeros(num_time_steps, dtype=object)

# assigns each S(k,t) to an element of averaged_sf
for i in range(num_time_steps):
    name = "data//model A average unnormalised sf #"+str(i)+" over 10 inits.txt)"
    averaged_sf[i] = np.loadtxt(name)
    
# loading k values and time steps
kvals = np.loadtxt("data//model A kvals.txt")
time_steps = np.loadtxt("data//model A time steps.txt")

# normalise each S(k,t) for each time step by dividing by S(k,0)
# then plotting agains k values
for sf in averaged_sf[1:]:
    normalised_sf = sf/averaged_sf[0]
    plt.plot(kvals, normalised_sf)
