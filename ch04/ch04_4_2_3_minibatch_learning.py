import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 导入mnist 数据集
# x_train 表示训练特征
# t_train 表示训练标签
# x_test  表示测试特征
# t_test  表示测试标签
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 打印mnist数据集中训练数据，测试数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 抽取小批量-minibatch
print("\n=== 抽取小批量-minibatch ")
train_size = x_train.shape[0]
print("train_size:", train_size) # train_size: 60000
batch_size = 10
# np.random.choice(60000， 10) 从0到59999之间随机选择10个数字
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask) # [48518 20742 15521 28731 49193 47555 22867 15607 56529 53532]

print("\n=== 选择的小批量数据如下：")
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch)
print(t_batch)

