import numpy as np
import matplotlib.pyplot as plt


# 基于中心差分的数值梯度
def numerical_diff_v2(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试案例
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.show()

# 计算function_1函数在x=5， x=10处的导数
# 按照公式推到 导数=0.02x+0.1，所以在x=5的导数为0.2， 在x=10的导数为0.3
print(numerical_diff_v2(function_1, 5))  # 0.1999999999990898
print(numerical_diff_v2(function_1, 10))  # 0.2999999999986347
