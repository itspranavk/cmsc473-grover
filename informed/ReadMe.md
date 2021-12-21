This folder contains the result of informed perturbations.

adversarial.py: applies the gradient to the embedding table, then find the similar pair of embeddings. (the cosine similarity is greater than 0.18)

test-probs.npy: original classification logits.

adverse_test-probs.npy: classification logits after adversarial attacks.

load_npy.py: the script to inspect the npy file.

run_discrimination.py: the script to run perturbations and get the classification result.
