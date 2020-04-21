import numpy as np

def read_matrix(filename):
    matrix = []
    with open(filename, 'r') as fin:
        rows = fin.read().strip().split('\n')
        for row in rows:
            entries = row.split(',')
            matrix.append(list(map(float, entries)))
    return np.array(matrix)