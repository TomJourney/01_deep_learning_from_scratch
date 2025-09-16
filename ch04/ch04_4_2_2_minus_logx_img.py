import numpy as np
import matplotlib.pylab as plt

# y = -log(x) 图像
def minus_log_func(x):
    return np.array(-np.log(x))

x = np.arange(0.0001, 1.1, 0.0001)
print(x)
y = minus_log_func(x)
print(y)
plt.plot(x, y)
plt.ylim(0.0, 5.0) # 指定y轴的范围
plt.show()
