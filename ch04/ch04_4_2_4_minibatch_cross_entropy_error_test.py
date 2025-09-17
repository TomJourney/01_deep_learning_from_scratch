import sys, os
import ch04_4_2_4_minibatch_cross_entropy_error as loss_func
import numpy as np

# 测试 交叉熵损失函数
# 例1： 索引为2的概率最高，0.6
print("\n=== 非one-hot标签的交叉熵损失函数")
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(loss_func.non_one_hot_mini_batch_cross_entropy_error(np.array(y1), np.array(t))) # 2.9957302735559908

# 例2： 索引为7的概率最高 0.6
print("\n=== one-hot标签的交叉熵损失函数")
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(loss_func.one_hot_mini_batch_cross_entropy_error(np.array(y1), np.array(t))) # 0.510825457099338