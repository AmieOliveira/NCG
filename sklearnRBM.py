""" File designed to help validate my implemented RBM training """

import numpy as np
from sklearn.neural_network import BernoulliRBM

size = 6

# Create BAS 4x4 data
dataSize = 2**(size+1)
data = np.zeros( ( dataSize, size**2 ) )
labels = np.zeros( dataSize )
vec = np.zeros(size)
idx = 0

def fill_BAS(n, vector):
    global size, data, idx

    if n == 0:
        # print(f"vector: {vector}, idx: {idx}")

        if np.count_nonzero(vector) == 0:
            # Pass this case, it contains only zeros...
            idx = idx + 2
            return

        for i in range(size):
            for j in range(size):
                # Horizontal
                data[idx, size*i + j] = vector[i]
                # Vertical
                data[idx+1, size*j + i] = vector[i]

        labels[idx] = 0
        labels[idx+1] = 1

        idx = idx + 2
        return

    fill_BAS(n-1, vector)
    vector[n-1] = abs(1 - vector[n-1])
    fill_BAS(n - 1, vector)


fill_BAS(size, vec)

print("Data: ")
print(data)

model = BernoulliRBM(n_components=size**2, learning_rate=0.1, n_iter=100, batch_size=5, verbose=1)

model.fit(data)#, labels)

scores = model.score_samples(data)
#print(scores)
print(f"NLL: {-sum(scores)/dataSize}")