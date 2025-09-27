
# 计算图的乘法层类
class MulLayer():
    def __init__(self):
        self.x = None
        self.y = None

    # 正向传播， x与y都是输入信号
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 反向传播， dout是上游导数（从右到左）
    # backward()把从上游传来的导数dout乘以正向传播的翻转值，然后传给下游
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
