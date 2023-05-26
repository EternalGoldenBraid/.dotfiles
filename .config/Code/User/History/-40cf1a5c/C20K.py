from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(-10, 10, 100)

# y = np.exp(-x)
# plt.plot(x, y, 'r-', label='exponential')

# y1 = 0.5**(-x)
# plt.plot(x, y1, 'b-', label='power 2')

mu = 0
y2 = np.exp**((-x-mu)*2)
plt.plot(x, y2, 'g-', label='power 2')

# y4 = 1/x
# plt.plot(x,y4, 'k-', label='1/x')

plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))



plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and power 2')
plt.legend()
plt.show()