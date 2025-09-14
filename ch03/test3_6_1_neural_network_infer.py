import numpy as np
import pickle
import sys, os
from common.neural_network_active_func import sigmoid, softmax_no_overflow
sys.path.append(os.pardir) # 为了导入父目录中的文件而设定
from dataset.mnist import load_mnist

# 3.6.2 神经网络的推理处理
def get_data():
    # 获取测试数据，包括测试图像，测试标签
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test

def init_network():
    # 读入保存在pickle文件sample_weight.pkl中学习到的权重参数
    with open("../dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

# 预测分类
def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_no_overflow(a3)

    return y

# ******************** 执行神经网络的推理处理
x, t = get_data() # 获取测试数据，包括测试图像x，测试标签t
network = init_network() # 初始化神经网络(读入保存在pickle文件sample_weight.pkl中学习到的权重参数)

accuracy_cnt = 0  # 识别准确的个数
for i in range(len(x)): # 遍历测试图像x
    # 预测分类
    y = predict(network, x[i]) # 预测得到预测值
    p = np.argmax(y) # 获取y的数组中最大值的索引
    if p == t[i]:
        accuracy_cnt += 1

print("accuracy: ", str(float(accuracy_cnt / len(x))))
# accuracy:  0.9207
