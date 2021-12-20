import numpy as np

with open('embed+grad.npy','rb') as f:
    embedding = np.load(f)
    gradient = np.load(f)

print(embedding.shape)
print(gradient.shape)
V, K = embedding.shape
token_list_idx = []
for i in range(V):
    gradient_i = gradient[i,:]
    if np.any(gradient_i):
        token_list_idx.append(i)

print(token_list_idx)

epsilons = [0.001, 0.01, 0.1, 0.15, 0.5, 1]

embedding_normalized = np.zeros_like(embedding)
for k in range(V):
    embedding_normalized[k,:] = embedding[k,:] / np.linalg.norm(embedding[k,:])

replace_dict = {}

for j in token_list_idx:
    for i, eps in enumerate(epsilons):
        embedding_new = embedding + eps * gradient
        malicious_embedding = embedding_new[j,:]
        malicious_embedding_normalized = malicious_embedding / np.linalg.norm(malicious_embedding)
        # max_sim = -1
        # max_idx = -1
        dot_product = np.dot(embedding_normalized, malicious_embedding_normalized)
        ind = np.argpartition(dot_product, -2)[-2:]
        if (dot_product[ind][0] + dot_product[ind][1] > 1.18):
            print(eps, j, ind, dot_product[ind])
            replace_dict[ind[1]] = ind[0]

print(replace_dict)
    # assert 0

