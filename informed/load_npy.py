import numpy as np

a = np.load('test-probs.npy')
L = a.shape[0]
print("machine", "human")
for i in range(L):
    print(a[i,:])