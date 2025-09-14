import numpy as np
import pickle
import sys, os
from common.neural_network_active_func import sigmoid, softmax_no_overflow

sys.path.append(os.pardir)  # 为了导入父目录中的文件而设定
from dataset.mnist import load_mnist
import test3_6_1_neural_network_infer_func as infer_func

# 3.6.2 神经网络的推理处理


# ******************** 执行神经网络的推理处理
x, t = infer_func.get_data()  # 获取测试数据，包括测试图像x，测试标签t
network = infer_func.init_network()  # 初始化神经网络(读入保存在pickle文件sample_weight.pkl中学习到的权重参数)

accuracy_cnt = 0  # 识别准确的个数
for i in range(len(x)):  # 遍历测试图像x
    # 预测分类
    y = infer_func.predict(network, x[i])  # 预测得到预测值
    p = np.argmax(y)  # 获取y的数组中最大值的索引
    if p == t[i]:
        accuracy_cnt += 1

print("accuracy: ", str(float(accuracy_cnt / len(x))))
# accuracy:  0.9207
