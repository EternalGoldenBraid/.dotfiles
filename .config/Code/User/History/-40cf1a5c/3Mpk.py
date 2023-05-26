from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
y = np.exp(-x)
y1 = 2**(-x)
y2 = 2**(x)

plt.plot(x, y, 'r-', label='exponential')
plt.plot(x, y1, 'b-', label='power 2')
plt.plot(x, y2, 'g-', label='power 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and power 2')
plt.legend()
plt.show()