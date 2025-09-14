import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


# mini-batch版本one-hot标签(0 1标签)的交叉熵损失函数 cross entropy error
def mini_batch_one_hot_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# mini-batch版本的非one-hot标签的交叉熵损失函数 cross entropy error
def mini_batch_non_one_hot_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
