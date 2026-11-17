import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2

fig = plt.figure(figsize=(11, 4.5))

# 左图：三维曲面
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85)
ax1.set_title(r'$z=x^2-y^2$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# 右图：等高线
ax2 = fig.add_subplot(1, 2, 2)
levels = np.linspace(-4, 4, 17)
cs = ax2.contour(X, Y, Z, levels=levels, cmap='coolwarm')
ax2.clabel(cs, inline=True, fontsize=8)

# 标出鞍点
ax2.plot(0, 0, 'ko')
ax2.annotate('鞍点 (0,0)', xy=(0, 0), xytext=(0.5, 0.8),
             arrowprops=dict(arrowstyle='->'))

# 画出两个特殊方向
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)  # y=0，x方向
ax2.axvline(0, color='gray', linestyle='--', linewidth=1)  # x=0，y方向

ax2.set_aspect('equal')
ax2.set_title('等高线：原点为鞍点')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()