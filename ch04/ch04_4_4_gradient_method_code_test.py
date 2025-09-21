import numpy as np
import ch04_4_4_gradient_method_code as grad_method


# 梯度法（或梯度下降法）测试案例
def funcion_2(x):
    return x[0] ** 2 + x[1] ** 2

# 初始点(-3, 4)
init_x = np.array([-3.0, 4.0])
# 计算结果
minimum_point = grad_method.gradient_descend(funcion_2, init_x, lr=0.1, step_num=100)
print(minimum_point) # [-6.11110793e-10  8.14814391e-10]

