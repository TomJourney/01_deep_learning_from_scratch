import ch04_4_4_gradient as grad

# 梯度法（或梯度下降法）代码实现
# f指目标函数或损失函数， init_x初始化点，lr是学习率或步长，step_num是梯度法的重复次数
def gradient_descend(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad_value = grad.numerical_gradient(f, x)
        x -= lr * grad_value

    return x
