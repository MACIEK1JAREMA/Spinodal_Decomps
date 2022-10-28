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
A = np.zeros((N,N))

# Set up each row to have a value equal to 1+row index
for i in range(3):
    A[i,:] = i+1
    
# Move each row down by 1
B = np.roll(A, N)

print(A)
print(B)

#%%
# We can also move columns, but now we have to set a=1 and the axis=1.

N = 3
X = np.zeros((N,N))

for i in range(3):
   X[:,i] = i+1
   
Y = np.roll(X, 1, axis=1)

print(X)
print(Y)
    
