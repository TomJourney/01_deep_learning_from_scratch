import numpy as np
import matplotlib.pylab as plt

# 3.2.4 sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1, 1, 2])
y = sigmoid(x)
print(y)  # [0.26894142 0.73105858 0.88079708]

# NumPy数组的广播功能
t = np.array([1, 2, 3])
print(t + 1)  # [2 3 4]
print(1 / t) # [1.         0.5        0.33333333]

# 画图
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴范围
plt.show()


