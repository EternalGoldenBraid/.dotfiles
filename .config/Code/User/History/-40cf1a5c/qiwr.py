from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)

# y = np.exp(-x)
# plt.plot(x, y, 'r-', label='exponential')

# y1 = 0.5**(-x)
# plt.plot(x, y1, 'b-', label='power 2')

# y2 = 0.5**(x)
# plt.plot(x, y2, 'g-', label='power 2')

y4 = 1/x
plt.plot(x,y4, 'k-')
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))



plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and power 2')
plt.legend()
plt.show()