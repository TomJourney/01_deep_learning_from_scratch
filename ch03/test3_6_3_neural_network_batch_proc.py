import sys, os
import numpy as np

sys.path.append(os.pardir)  # 为了导入父目录中的文件而设定
import test3_6_1_neural_network_infer_func as infer_func

# 【基于批处理的神经网络推理代码实现】
x, t = infer_func.get_data()  # x=测试集 t=测试标签
network = infer_func.init_network()

# 批次数量
batch_size = 2000
accuracy_cnt = 0
print(x.shape) # (10000, 784)
print(t.shape) # (10000,)

# 基于批处理的推理
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size] # 取出分批数据，每批一个单位
    print(x_batch.shape) # (2000, 784)
    y_batch = infer_func.predict(network, x_batch) # 以批为单位执行预测分类，得到概率数组
    print(y_batch.shape) # (2000, 10)
    p = np.argmax(y_batch, axis=1) # 概率最大的索引
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("预测准确率 = " + str(float(accuracy_cnt / len(x))))
# 预测准确率 = 0.9207
