import numpy as np

# 创建一个3x3数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用NumPy的nditer函数创建了一个迭代器对象，用于高效地遍历多维数组
it = np.nditer(arr, flags=['multi_index'], op_flags=[['readwrite']])

# 遍历数组
while not it.finished:
    idx = it.multi_index  # 获取迭代器中的索引（一维）
    print(f"索引={idx}, 值={arr[idx]}")
    arr[idx] = arr[idx] * 2
    it.iternext()
print(arr)
# 索引=(0, 0), 值=1
# 索引=(0, 1), 值=2
# 索引=(0, 2), 值=3
# 索引=(1, 0), 值=4
# 索引=(1, 1), 值=5
# 索引=(1, 2), 值=6
# 索引=(2, 0), 值=7
# 索引=(2, 1), 值=8
# 索引=(2, 2), 值=9
# [[ 2  4  6]
#  [ 8 10 12]
#  [14 16 18]]
