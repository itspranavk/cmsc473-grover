import numpy as np
import sys
from matplotlib import pyplot as plt

probs = np.load(sys.argv[1])

plt.title("Grover Confidence vs. Number of Synonyms (Human Article)")
plt.xlabel("Number of synonyms replaced")
plt.ylabel("Grover confidence")

for i in range(0, 373):
    plt.plot(i, probs[i][0], i, probs[i][1])

plt.show()

plt.title("Grover Confidence vs. Number of Synonyms (Machine Article)")
plt.xlabel("Number of synonyms replaced")
plt.ylabel("Grover confidence")

for i in range(373, 1041):
    plt.plot(i, probs[i][0], i, probs[i][1])

plt.show()