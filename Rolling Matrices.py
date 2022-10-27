import numpy as np

''' 
Here, we're using the np.roll function to move all the elements in the lattice
matrix.

np.roll(matrix, a)
    Moves each element in the matrix one element to the right by a indices.
    If the element is at the end of a row, it moves to the row below.
    a must be an integer
    
np.roll(matrix, N)
    Moves the bottom row to the top (moves each row down by 1 row)
    
np.roll(matrix, 2*N)
    Moves the bottom row to the second row (moves each row down by 2)

np.roll(matrix, -N)
    Moves the top row to the bottom (moves each row up by 1)
'''

# NxN lattice
N = 3
M = np.zeros((N,N))

# Set up each row to have a value equal to 1+row index
for i in range(3):
    M[i,:] = i+1
    
# Move each row down by 1
X = np.roll(M, N)

print(M)
print(X)