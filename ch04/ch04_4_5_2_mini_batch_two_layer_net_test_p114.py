import numpy as np
import ch04_4_5_1_two_layer_net_p111 as two_layer_network
from dataset import mnist
from dataset.mnist import load_mnist
import common.time_utils as time_utils

# 加载minist数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
# 训练损失列表
train_loss_list = []

# 超参数
iters_num = 10 # 迭代次数应该设置为10000，设置为10仅本地演示
train_size = x_train.shape[0] # 训练集数据量
batch_size = 100  # mini-batch的大小（每批数据量）
learning_rate = 0.1 # 学习率

# 创建两层神经网络实例
network = two_layer_network.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print(f"{time_utils.get_now_year_month_day_hour_minite_second()} 第{i}次训练开始")
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size) # 获取mini-batch小批量
    x_batch = x_train[batch_mask] # mini-batch的训练集
    t_batch = t_train[batch_mask] # mini-batch的测试集

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(f"{time_utils.get_now_year_month_day_hour_minite_second()} 第{i}次训练结束")

# 打印每次训练的损失
print("\n=== 打印每次训练的损失")
print(train_loss_list)