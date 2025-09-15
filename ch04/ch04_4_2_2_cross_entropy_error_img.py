import numpy as np
from PIL import Image # PIL -> Python Image Library # python图像库

# 3.6.1 mnist 数据集
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 把numpy数组的图像数据转为PIL对象
    pil_img.show()

# 读取mnist数据集 , flatten=True表示读入的图像是一维的，
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
img = x_train[0]
lable = t_train[0]
print(lable)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 读入的图像是一维的，显示时，需要把图像的形状变为原来的尺寸(28 * 28 )
print(img.shape)  # (28, 28)

# 显示图像
img_show(img)
