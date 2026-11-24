# sgd-随机梯度下降法 实现

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    # params: 权重； grads：梯度
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
