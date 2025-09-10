import numpy as np
import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而设定
from dataset.mnist import load_mnist

# 3.6 手写数字识别

# 读入MNIST数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
# 输出各个数据的形状
print("\n=== 训练数据与训练标签")
print(x_train.shape)
print(t_train.shape)
# (60000, 784)
# (60000, 10)
print("\n=== 测试数据与测试标签")
print(x_test.shape)
print(t_test.shape)
# (10000, 784)
# (10000, 10)

