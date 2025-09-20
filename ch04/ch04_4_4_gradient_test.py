import numpy as np
import matplotlib.pylab as plt
import ch04_4_4_gradient as grad
import ch04_4_3_3_partial_derivative_src_func as src_func2

# 计算梯度
# 计算在 (3,4)  (0,2) (3,0) 处的梯度
print(grad.numerical_gradient(src_func2.func2, np.array([3.0, 4.0])))
print(grad.numerical_gradient(src_func2.func2, np.array([0, 2.0])))
print(grad.numerical_gradient(src_func2.func2, np.array([3.0, 0.0])))
# [6. 8.]
# [0. 4.]
# [6. 0.]

