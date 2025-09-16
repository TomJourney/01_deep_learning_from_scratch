import numpy as np
import matplotlib.pylab as plt

# 3.2.5 sigmoid函数和阶跃函数的比较
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 画图
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
print(y)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴范围

# 阶跃函数画图
def step_function(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y, linestyle='--')
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()

