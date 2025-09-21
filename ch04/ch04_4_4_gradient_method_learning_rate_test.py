import numpy as np
import ch04_4_4_gradient_method_code as grad_method

# 梯度法（或梯度下降法）测试案例
def funcion_2(x):
    return x[0] ** 2 + x[1] ** 2

# 初始点(-3, 4)
init_x = np.array([-3.0, 4.0])

# 学习率过大的例子 lr=10.0
minimum_point = grad_method.gradient_descend(funcion_2, init_x, lr=10.0, step_num=100)
print(minimum_point) # [-2.58983747e+13 -1.29524862e+12]

# 学习率过小的例子 lr=1e-10
init_x = np.array([-3.0, 4.0])
minimum_point_2 = grad_method.gradient_descend(funcion_2, init_x, lr=1e-10, step_num=100)
print(minimum_point_2) # [-2.99999994  3.99999992]
