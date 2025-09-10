import numpy as np

# 3.5 输出层的设计
# 激活函数选择：
# 回归问题使用恒等函数， 分类问题使用softmax函数
a = np.array([0.3, 2.9, 4, 0])
exp_a = np.exp(a)  # 指数函数
print(exp_a)
# [ 1.34985881 18.17414537 54.59815003  1.        ]

print("\n=== 计算指数函数的和")
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)  # 75.1221542101633

print("\n=== 计算softmax函数值")
y = exp_a / sum_exp_a
print(y)  # [0.01796885 0.2419279  0.72679159 0.01331165]


# 定义softmax函数
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

# 验证解决溢出问题的softmax函数
print("\n=== 验证解决溢出问题的softmax函数")
x = np.array([0.3, 2.9, 4.0])
y = softmax_no_overflow(x)
print(y) # [0.01821127 0.24519181 0.73659691]
print(np.sum(y)) # 1.0

