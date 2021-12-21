from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
import numpy as np

import sys

#probs result from article 2
res2 = np.load(sys.argv[1])
res2


res2 = np.transpose(res2)

machine = res2[0].copy()
human = res2[1].copy()

x = np.array([i for i in range(len(res2[0])) ])
#xn = np.linspace(1, len(res2[0]), len(res2[0])*10)

machine_data = plt.scatter(x,machine, c ="blue",
            linewidths = 1,
            marker ="^", 
            s = 50,
           label = "Machine")
human_data = plt.scatter(x, human, c ="orange", 
            linewidths = 2, 
            marker ="s", 
            s = 30)

plt.ylabel("Probability")
plt.xlabel("Number of synonyms")
plt.title("Number of synonyms vs. Grover confidence")

plt.legend((machine_data, human_data), ("Machine", "Human"))

plt.show()
