import numpy as np

a = np.array([[0, 0, 14, 6],
              [-1, 0, 0, 19],
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

print('\nARGMAX')
print('Index of max value: ', np.argmax(a))
print('Row with max value per column')
print('Column:  1 2 3 4')
print('Index: ', np.argmax(a, axis=0))
print('Col with max value per row')
print('Row:    1 2 3 4')
print('Index:', np.argmax(a, axis=1))

print('\nCONCATENATE')
print('Before concatenation: \n', a)
print('Concatenate with single argument: ', np.concatenate(a))
print('Concatenation is equal to unfolding')

print('\nLINESPACE')
print('1-6 with num=5 (and endpoint=False by default): ')
print(np.linspace(start=1, stop=6, num=5, endpoint=False))
print('1-6 with num=5 and endpoint=True with 14 samples: ')
print(np.linspace(start=1, stop=6, num=5, endpoint=True))


