from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(-10, 10, 100)

# y = np.exp(-x)
# plt.plot(x, y, 'r-', label='exponential')

# y1 = 0.5**(-x)
# plt.plot(x, y1, 'b-', label='power 2')

# mus = np.arange(-11, 11, 6)
mus = [-4, 0, 4]
sd = [0.5, 1, 8]
for mu in mus:
    for s in sd:
        y = np.exp(-(x-mu)**2/(2*s**2))
        plt.plot(x, y, label='mu='+str(mu)+', sd='+str(s))


plt.xticks(np.arange(-11, 11, 1))
# plt.yticks(np.arange(-11, 11, 1))



plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and power 2')
plt.legend()
plt.show()