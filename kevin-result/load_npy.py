import numpy as np

a = np.load('kevin_test-probs.npy')
L = a.shape[0]
print("machine", "human")
for i in range(L):
    print(a[i,:])