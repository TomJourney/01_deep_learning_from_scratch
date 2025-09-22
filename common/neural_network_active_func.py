import numpy as np

# 使用激活函数sigmoid转换神经元的加权和
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# identy_func : 是输出层的激活函数，也称恒等函数
def identity_func(x):
    return x

# 激活函数选择：
# 回归问题使用恒等函数， 分类问题使用softmax函数
# 定义softmax函数
# Softmax函数是一种将任意实数向量转化为概率分布的归一化指数函数，其输出向量的每个元素都在0到1之间，且所有元素的和为1
def softmax(x):
    exp_a = np.exp(x)  # 计算指数函数
    sum_exp_a = np.sum(exp_a)  # 指数函数值求和
    y = exp_a / sum_exp_a  # 每个元素的指数函数值 除以 求和值
    return y


# 定义解决溢出问题的softmax函数
def softmax_no_overflow(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 交叉熵损失函数 cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))