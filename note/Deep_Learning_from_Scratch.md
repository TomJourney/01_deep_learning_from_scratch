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

![image-20250907221326326](/Users/rong/studynote/01-ai/01_deep_learning_from_scratch/note/ch02_p34.png)

<br>

















