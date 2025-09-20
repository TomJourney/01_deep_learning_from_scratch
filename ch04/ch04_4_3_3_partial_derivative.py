import numpy as np
import matplotlib.pyplot as plt

# 偏导数
# f(x0,x1) = x0^2 + x1 ^2
def partial(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)

