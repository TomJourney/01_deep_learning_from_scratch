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





















 

































