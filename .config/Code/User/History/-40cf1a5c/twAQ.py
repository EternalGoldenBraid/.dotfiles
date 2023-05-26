from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
y = np.exp(-x)

plt.plot(x, y, 'r-')
plt.show()