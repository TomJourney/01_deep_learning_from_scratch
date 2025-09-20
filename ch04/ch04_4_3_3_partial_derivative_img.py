import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# f(x0,x1) = x0^2 + x1 ^2 图像
# 创建x0和x1的网格
x0 = np.linspace(-5, 5, 200)
x1 = np.linspace(-5, 5, 200)
X0, X1 = np.meshgrid(x0, x1)

# 计算z值
Z = X0**2 + X1**2

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图
surf = ax.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.8)

# 设置标签
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('z = x0² + x1²')
ax.set_title('z = x0² + x1²')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10)

# 显示图形
plt.tight_layout()
plt.show()