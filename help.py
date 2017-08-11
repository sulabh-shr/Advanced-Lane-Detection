import numpy as np

a = np.array([[0, 0, 4, 6],
              [5, 0, 0, 1],
              [8, 0, 2, 0],
             [0, 11, 0, 7]])

nonzero_indices = a.nonzero()
nonzero_a = a[a.nonzero()]

print('NON-ZERO')
print('Non-zero row and column indices respectively: ', nonzero_indices)
print('Non-zero row indices: ', nonzero_indices[0])
print('Non-zero col indices: ', nonzero_indices[1])
print('Non-zero a elements: ', nonzero_a)

b = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

c = np.array([['a', 'b', 'c', 'd'],
              ['e', 'f', 'g', 'h'],
              ['i', 'j', 'k', 'l']])
stacked = np.dstack((b, c))
print('\nDSTACK')
print('stacked: \n', stacked)
print('shape: ', stacked.shape)
print('(num_rows, num_cols, num_arrays)')

