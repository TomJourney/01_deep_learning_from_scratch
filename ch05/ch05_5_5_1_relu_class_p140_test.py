import numpy

from ch05.ch05_5_5_1_relu_class_p140 import Relu

rule = Relu()

print("\n========== 前向传播： ")
x1 = numpy.array([-1, 0, 1, 2,3,4,5])
print(rule.forward(x1))

print("\n========== 反向传播： ")
print(rule.backward(x1))

print("\n========== mask： ")
print(rule.get_mask())