from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
y = np.exp(-x)
y1 = 2**(-x)

plt.plot(x, y, 'r-')
plt.plot(x, y1, 'b-')
plt.show()