import numpy as np
import matplotlib.pylab as plt

# 3.2.7  ReLU函数
def relu(x):
    return np.maximum(0, x)

#
x = np.arange(-6.0, 6.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()
