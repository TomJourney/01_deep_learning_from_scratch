[toc]               

# 【1】python基础

## 【1.3】python解释器

```python
# 1.3.2 数据类型
print("=== 1.3.2 数据类型")
print(type(10))
print(type(3.14))
print(type("hello"))
# <class 'int'>
# <class 'float'>
# <class 'str'>

# 1.3.4 列表
a = [1, 2, 3, 4, 5]
print(a)
```

<br>

---

## 【1.5】NumPy

1）numpy定义：数组和矩阵的计算库；

```python
import numpy as np

# 生成numpy数组
print("==== 生成numpy数组")
x = np.array([1, 2, 3])
print(x)
print(type(x))
# [1 2 3]
# <class 'numpy.ndarray'>

# numpy的算术运算
print("=== numpy的算术运算")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
# [5 7 9]
# [-3 -3 -3]
# [ 4 10 18]
# [0.25 0.4  0.5 ]

# numpy的n维数组
print("\n=== # 1.5.4 numpy的n维数组")
A = np.array([[1,2], [3,4]])
print(A)
print(A.shape)
print(A.dtype)
# [[1 2]
#  [3 4]]
# (2, 2)
# int64

print("\n=== 1.5.4 矩阵加法与乘法")
B = np.array([[3,0], [0,6]])
print(A+B)
print(A*B)
# [[ 4  2]
#  [ 3 10]]
# [[ 3  0]
#  [ 0 24]]


print("\n=== 矩阵广播")
print(A*10)
# [[10 20]
#  [30 40]]

print("\n=== 1.5.5 广播")
A= np.array([[1,2], [3,4]])
B= np.array([10,20])
print(A*B)
# [[10 40]
#  [30 80]]

print("\n=== 1.5.6 访问元素")
x = np.array([[51, 55],[14, 19],[0,4]])
print(x)
print(x[0])
print(x[0][1])
# [[51 55]
#  [14 19]
#  [ 0  4]]

# [51 55]
# 55

print("\n=== 使用数组访问各个元素")
x = x.flatten()
print(x)
# 获取索引0 2 4 的元素
print(x[np.array([0, 2, 4])])
# [51 55 14 19  0  4]
# [51 14  0]

# 筛选元素，类似于java流
print("\n=== 筛选元素，类似于java流")
print(x>15)
print(x[x>15])
# [ True  True False  True False False]
# [51 55 19]

```

<br>

---

## 【1.6】Matplotlib

1）Matplotlilb定义：Matplotlilb用于绘制图形的库。使用Matplotlib可以轻松绘制图形和实现数据的可视化；

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 1.6 Matplotlib
# 使用Matplotlib可以绘制图形和实现数据的可视化

# 1.6.1 绘制简单图形
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()

# 1.6.2 pyplot功能
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend()
plt.show()

# 1.6.3 显示图像  pyplot提供了用于显示图像的方法 imshow()
# from matplotlib.image import imread
img = imread('../img/doge.jpeg')
plt.imshow(img)
plt.show()

```

<br>

---

# 【2】感知机

感知机定义：接受多个输入信号，得到1个输出信号的算法；如与门，与非门，或门；

## 【2.3】感知机的实现

```python
import numpy as np

# 2.3.1 简单感知机 与门
print("\n=== 2.3.1 简单感知机 与门")


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1 * w1 + x2 * w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1


print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
# 0
# 0
# 0
# 1

print("\n=== 2.3.2 导入权重和偏置")
# 2.3.2 导入权重和偏置
x = np.array([0, 1])  # 输入
w = np.array([0.5, 0.5])  # 权重
b = -0.7
print(w * x)  # [0.  0.5]
print(np.sum(w * x))  # 0.5
print(np.sum(w * x) + b)  # -0.19999999999999996 大约为0.2

# 2.3.3 与门版本2 使用权重和偏置的实现与门
print("\n=== 2.3.3 与门版本2 使用权重和偏置的实现与门")
def AND_v2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    temp = np.sum(w * x) + b
    if temp <=0:
        return 0
    else:
        return 1
print(AND_v2(0, 0))
print(AND_v2(1, 0))
print(AND_v2(0, 1))
print(AND_v2(1, 1))
# 0
# 0
# 0
# 1

print("\n=== 非门 使用权重和偏置的实现非门")
def NON_AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 权重与偏置与 AND不同
    b = 0.7
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1
print(NON_AND(0, 0))
print(NON_AND(1, 0))
print(NON_AND(0, 1))
print(NON_AND(1, 1))
# 1
# 1
# 1
# 0

print("\n=== 或门 使用权重和偏置的实现或门")
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 权重与偏置与 AND不同
    b = -0.2
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
# 0
# 1
# 1
# 1

```

<br>

---

## 【2.4】感知机的局限性

1）感知机：无法表示异或门（当且仅当x1或x2中仅有一方等于1，才输出1，否则输出0）；

- 感知机的局限性：只能表示一条直线分割的空间，无法表示曲线分割的空间；
  - 非线性空间：被曲线分割的空间；
  - 线性空间：被直线分割的空间；

![image-20250907220402132](/Users/rong/studynote/01-ai/01_deep_learning_from_scratch/note/ch01_p24.png)

<br>

---

## 【2.5】多层感知机

1）多层感知机：多个感知机叠加或组合的感知机；

2）<font color=red>多层感知机的作用</font>：单层感知机无法实现异或门，但多个单层感知机叠加或组合后能够实现异或门；

<br>

---

### 【2.5.2】异或门的实现

```python
# 2.5.2 异或门的实现
print("\n=== 2.5.2 异或门的实现")


def XOR(x1, x2):
    s1 = NON_AND(x1, x2)
    s2 = OR(x2, x1)
    y = AND(s1, s2)
    return y


print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
# 0
# 1
# 1
# 0
```

<br>

3）用感知机表示异或门的图例

![image-20250907221326326](./ch02_p34.png)

<br>

## 【总结】

单层感知机只能表示线性空间，而多层感知机可以表示非线性空间； 

<br>

---

# 【3】神经网络

1）神经网络定义：自动从数据中学习到合适的权重参数的数学模型或计算模型；

---

## 【3.1】从感知机到神经网络

1）激活函数定义：把输入信号的总和转换为输出信号的函数；作用在于如何激活输入信号的总和；

如输入信号的总和为：  
$$
a = b + w1x1 + w2x2 
$$


输出信号为：
$$
y=h(a)
$$
则总和为a，函数h()把a转为输出y，则h()就是激活函数； 

![image-20250907220402132](./ch03_p41.png)

<br>2）朴素感知机与多层感知机：

- 朴素感知机：单层网络，指激活函数使用了阶跃函数的模型；（阶跃函数指一旦输入超过阈值，就切换输出的函数，如分段函数）

- 多层感知机：指神经网络，即使用sigmoid函数等平滑的激活函数的多层网络；


<br>

---

## 【3.2】激活函数

1）激活函数定义：把输入信号的总和转换为输出信号的函数；作用在于如何激活输入信号的总和；

### 【3.2.1】sigmoid函数

1）sigmoid函数：神经网络中常用的一种激活函数；
$$
h(x)=\frac{1}{1+exp(-x)}
$$
其中 exp(-x)表示$$ e^{-x} $$ ，e是纳皮尔常数 2.7182...

<font color=red>补充：感知机与神经网络的主要区别</font>

- 感知机：使用阶跃函数作为激活函数；
- 神经网络：使用sigmoid函数作为激活函数；

<br>

---

### 【3.2.2】阶跃函数的实现

```python
# 阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(int)

x = np.array([-1.0, 1.0, 2.0])
print(x)  # [-1.  1.  2.]
y = x > 0
print(y)  # [False  True  True]

# 布尔类型转为int型
print(y.astype(int)) # [0 1 1]
```

<br>

【阶跃函数的图形】

```python
import numpy as np
import matplotlib.pylab as plt

# 3.2.3 阶跃函数的图形
def step_function(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()

```

![image-20250907220402132](./ch03_p45.png)

<br>

---

### 【3.2.4】sigmoid函数的实现

1）sigmoid函数：

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### 【3.2.5】sigmoid函数和阶跃函数的比较

```python
import numpy as np
import matplotlib.pylab as plt

# 3.2.5 sigmoid函数和阶跃函数的比较
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 画图
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴范围

# 阶跃函数画图
def step_function(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y, linestyle='--')
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()

```

![image-20250907220402132](./ch03_p47.png)



<br>

### 【3.2.6】非线性函数

1）神经网络的激活函数，必须使用非线性函数； 

因为，为了发挥叠加层所带来的优势，神经网络的激活函数必须使用非线性函数；

<br> 

### 【3.2.7】ReLU函数

1）ReLU函数定义： 整流线性单位函数（Rectified Linear Unit, ReLU），又称修正线性单元，<font color=red>是一种人工神经网络中常用的激励函数（activation function）</font>，通常指代以斜坡函数及其变种为代表的非线性函数。
2）比较常用的线性整流函数有斜坡函数：$$f(x)=max(0,x)$$。

3）ReLU函数的实例如下：
$$
h(x)=\begin{cases}
x \quad (x>0)\\
0 \quad (x<=0)\\
\end{cases}
$$
![image-20250907220402132](./ch03_p50.png)

<br>

---

## 【3.4】3层神经网络的实现

### 【3.4.3】代码实现小结

1）神经网络代码实现小结

```python
import numpy as np

# 使用激活函数sigmoid转换神经元的加权和
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# identy_func : 是输出层的激活函数，也称恒等函数
def identity_func(x):
    return x

# 3.4.3 神经网络代码实现小结
def init_network():
    network = dict()
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# 前向传播
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_func(a3)

    return y

# 调用神经网络初始化函数，前向传播函数
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) #[0.31682708 0.69627909]

```

【函数介绍】

- init_network()：初始化神经网络的权重和偏置；把权重和偏置保存到字典变量network中；network字典变量保存了每一层所需的参数；
- forward()-前向函数：封装了将输入信号转为输出信号的过程；

2）神经网络输入信号到输出信号传递示意图

![image-20250913214342117](./ch03_p62_1copy.png)

<br>

---

### 【3.4.4】三层神经网络结构 

![image-20250913214539513](./ch03_p62_1copyb.png)

<br>

---

## 【3.5】输出层的设计 

### 【3.5.1】激活函数分类

1）神经网络可以用在分类问题与回归问题上，不过需要根据实际情况修改激活函数；

- 回归问题：使用恒等函数作为激活函数； （恒等函数： identity_function）
- 分类问题：使用softmax函数作为激活函数；

<br>

---

### 【3.5.2】恒等函数

1）恒等函数定义：把输入按照原样输出的函数；如 f(x)=x 

![image-20250913214539513](./ch03_p64.png)

<br>

---

### 【3.5.3】softmax函数

1）softmax函数定义：Softmax函数是一种将任意实数向量转化为概率分布的归一化指数函数，其输出向量的每个元素都在0到1之间，且所有元素的和为1。

- softmanx作用：它主要用于多分类问题中，作为神经网络的最后一层输出层，将原始分数转化为表示每个类别概率的向量，从而使模型能够对不同类别进行概率预测。﻿

2）softmax函数公式如下：
$$
y_k=\frac{exp(a_k)}{\sum_{i=1}^{n}exp(a_i)}
$$
softmax函数的图像如下。



![image-20250913214539513](./ch03_p65.png)



3）softmax函数的代码实现：

```python
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
```

<br>

---

### 【3.5.4】求解机器学习问题总结（学习+推理）

1）求解机器学习问题步骤：包括学习与推理；

- 学习阶段：在学习阶段进行模型的学习；
- 推理阶段：在推理阶段，用学到的模型对未知的数据进行推理（分类）；

<br>

---

## 【3.6】手写数字识别（推理实践）

1）前向传播： 我们使用学习到的参数，先实现神经网络的推理处理，这个推理处理的过程称为神经网络的前向传播； 

### 【3.6.0】使用神经网络解决问题的步骤

- 步骤1：使用训练数据进行权重参数的学习；
- 步骤2：推理时，使用步骤1学习得到的参数，对输入数据进行分类； 

<br>

---

### 【3.6.1】MNIST数据集

1）导入数据集

```python
import numpy as np
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而设定
from dataset.mnist import load_mnist
from PIL import Image # PIL -> Python Image Library # python图像库


# 3.6.1 mnist 数据集
def img_show(img):
    # np.uint8(img) 这一句是 把 img 转换成 numpy 的无符号 8 位整数类型
    pil_img = Image.fromarray(np.uint8(img)) # 把numpy数组的图像数据转为PIL对象
    pil_img.show()


# 读取mnist数据集 , flatten=True表示读入的图像是一维的，
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
img = x_train[0]
lable = t_train[0]
print(lable)  # 训练集的测试标签 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 读入的图像是一维的，显示时，需要把图像的形状变为原来的尺寸(28 * 28 )
print(img.shape)  # (28, 28)

# 显示图像
img_show(img)

```

【代码解说】

- sys.path.append(os.pardir) 作用：

  - sys.path：这是一个Python 列表，它包含了Python 解释器在导入（import）模块时会查找的所有目录路径。﻿

  - os.pardir：这是 os 模块中的一个常量，它代表字符串 '..'，表示当前目录的父目录或上一级目录。﻿

  - append() 方法：这是Python 列表的一个方法，用于在列表的末尾添加一个元素。﻿
  - 当前目录的父目录添加到python解释器用于查找模块的目录中去，使用以后，python就可以找到当前目录的父目录中的模块。

- 使用 load_minst函数可以轻松读入MNIST数据。

<br>

---

### 【3.6.2】神经网络的推理处理 

1）针对MNIST书记实现神经网络的推理处理：

- 神经网络输入层有784个神经元（28 * 28），因为图像大小=28*28；
- 输出层有 10 个神经元（数字0-9，共10个类别） ；
- 神经网络共有2个隐藏层：
  - 第1层有50个神经元；
  - 第2层有100个神经元；

2）推理处理代码：

【推理函数】

```python
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
```

【推理执行】

```python
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

```

【补充】

- sample_weight.pkl  权重参数文件，从 [源码官网](https://www.ituring.com.cn/book/1921)下载

<br>

---

### 【3.6.3】批处理 

1）批处理定义： 对输入信号批量处理，提高处理效率；

- 批：打包式的输入数据称为批； 

2）输出3.6.2中神经网络的各层形状； 

```python
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而设定
import test3_6_1_neural_network_infer_func as infer_func

# 3.6.3 批处理
# 打印神经网络各层形状
x, _ = infer_func.get_data()
network = infer_func.init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape)  # (10000, 784)
print(x[0].shape)  # (784,)  输入层，第0层
print(W1.shape)  # (784, 50)  中间层，第1层
print(W2.shape)  # (50, 100)  中间层，第2层
print(W3.shape)  # (100, 10)  输出层， 第3层
```

3）基于批处理的神经网络推理代码实现：

```python
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
```

<br>

---

# 【4】神经网络的学习

1）学习定义： 指从训练数据中自动获取最优权重参数的过程； 

### 【4.1.2】训练数据与测试数据

1）泛化能力：指处理未被观察过的数据的能力。获得泛化能力是机器学习的最终目标； 

2）为了准确评价模型的泛化能力，必须划分训练数据和测试数据；

- 训练数据：包括训练特征，训练标签；
- 测试数据：包括测试特征，测试标签；

<br>

---

## 【4.2】损失函数

1）损失函数定义： 神经网络寻找最优权重参数的指标；

- 损失函数：表示神经网络性能的恶劣程度（不拟合程度）的指标，即当前的神经网络对监督数据在多大程度上不拟合；

<br>

### 【4.2.1】均方误差

1）均方误差定义：表示预测值与真实值间差值的平方的均值。

公式如下：
$$
E=\frac{1}{2}\sum_{i=1}^{n}(y_i-t_i)^2
$$
2）代码实现：

```python
import numpy as np

# 均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

# 测试 均方误差函数
# 例1： 索引为2的概率最高，0.6
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(mean_squared_error(np.array(y1), np.array(t))) # 0.09750000000000003

# 例2： 索引为7的概率最高 0.6
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y2), np.array(t))) # 0.5975
```

<br>

---

### 【4.2.2】交叉熵误差

1）交叉熵误差定义：度量两个概率分布间的差异；

- 熵定义：无损编码事件信息的最小平均编码长度。

公式如下：
$$
E=-\sum_{i=1}^{n}t_i\log{y_i}
$$

- 其中， $y_i$是神经网络输出，$t_i$是正确解标签（取值1）； 
- $t_i$中只有正确解标签的索引为1，其他均为0（one-hot表示）
- 因此，上述公式实际上只计算对应正确解标签的输出的自然对数；
- <font color=red>交叉熵误差的值</font>：是由正确解标签所对应的输出结果决定的； 
  - 如，正确解标签索引是2，对应的神经网络输出是0.6，则交叉熵损失是 $-\log{0.6}$=0.51

2）$y=-\log{x}$图像（辅助理解交叉熵损失函数）：

```python
import numpy as np
import matplotlib.pylab as plt

# y = -log(x) 图像
def minus_log_func(x):
    return np.array(-np.log(x))

x = np.arange(0.0001, 1.1, 0.0001)
print(x)
y = minus_log_func(x)
print(y)
plt.plot(x, y)
plt.ylim(0.0, 5.0) # 指定y轴的范围
plt.show()

```

![-logx函数图像](./ch04_p87_minus_logx.png)

3）python代码实现：

```python
import numpy as np

# 交叉熵损失函数 cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 测试 交叉熵损失函数
# 例1： 索引为2的概率最高，0.6
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(cross_entropy_error(np.array(y1), np.array(t))) # 0.510825457099338

# 例2： 索引为7的概率最高 0.6
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y2), np.array(t))) # 2.302584092994546
```

【补充】

函数内部在计算np.log时，加上了一个微小值delta。这是因为当出现np.log(0)时，np.log(0)会变为负无限大的-inf，这会导致计算无法正常进行。

<br>

---

### 【4.2.3】mini-batch学习（小批量学习）

1）mini-batch学习定义： 神经网络从训练数据中选出一批数据（称为mini-batch，小批量），然后对每个mini-batch进行学习；

- 如，从60000个训练数据中随机选择100笔作为mini-batch，然后再针对这个mini-batch进行学习；

2）计算所有训练数据的交叉熵误差（<font color=red>把单个数据的交叉熵误差扩展到N份数据</font>）：
$$
E=-\frac{1}{N}\sum_{i}^{n}\sum_{j}^{m}t_{ij}\log{y_{ij}}
$$

- 其中 $t_{ij}$表示第i个数据项的第j个元素的值， 其中$y_{ij}$ 是神经网络输出，$t_{ij}$ 是标签数据；
- 通过除以n，可以计算单个数据的平均损失函数；
- 通过平均化，可以获得和训练数据的数据无关的统一指标； 

3）代码实现抽取mini-batch：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 导入mnist 数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# x_train 训练特征
# t_train 测试特征
# x_test 训练标签 
# t_test 测试标签 

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
```

【补充】

```python
# x_train 表示训练特征
# t_train 表示训练标签
# x_test  表示测试特征
# t_test  表示测试标签
```

<br>

---

### 【4.2.4】mini-batch版本的交叉熵误差实现 

1）实现可以同时处理单个和批量数据（mini-batch）的两种情况的函数；

函数1）one-hot标签的mini-batch版本的交叉熵损失函数 ：

```python
# one-hot标签的mini-batch版本的交叉熵损失函数 cross entropy error
def one_hot_mini_batch_cross_entropy_error(y, t):
    # 若轴个数为1，即一维数组，则转为二维数组 
    # ndim表示轴个数 
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # y向量在第0轴的长度
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

【补充】one-hot标签指0 1 标签，即标签数据t向量的取值为0或1 ； 



函数2）非one-hot标签的mini-batch版本的交叉熵损失函数：

```python
# 非one-hot标签的mini-batch版本的交叉熵损失函数 cross entropy error
def non_one_hot_mini_batch_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```











 

































