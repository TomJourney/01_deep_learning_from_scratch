import numpy as np


# 阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(int)

x = np.array([-1.0, 1.0, 2.0])
print(x)  # [-1.  1.  2.]
y = x > 0
print(y)  # [False  True  True]

# 布尔类型转为int型
print(y.astype(int)) # [0 1 1]


