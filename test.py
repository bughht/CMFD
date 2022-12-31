import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.0001)
y = -np.sign(x)*np.log(np.abs(x))

plt.plot(x, y)
plt.show()

for i in range(len(x)):
    print(x[i], y[i])
