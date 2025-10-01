from ch05.ch05_5_4_1_multiply_layer_p135 import MulLayer
from ch05.ch05_5_4_2_add_layer_p137 import AddLayer

# 计算图的加法层类的测试用例
apple_price = 100
apple_num = 2
orange_price = 150
orange_num = 3
tax = 1.1

# 创建层实例
mul_apple_layer_ins = MulLayer()
mul_orange_layer_ins = MulLayer()
add_apple_layer_ins = AddLayer()
mul_tax_layer_ins = MulLayer()

# forward 前向传播
apple_price = mul_apple_layer_ins.forward(apple_price, apple_num) # (1)
orange_price = mul_orange_layer_ins.forward(orange_price, orange_num)  # (2)
all_price = add_apple_layer_ins.forward(apple_price, orange_price) #(3)
taxed_price = mul_tax_layer_ins.forward(all_price, tax)

# backward 后向传播
dprice = 1
dall_price = add_apple_layer_ins.forward(dprice, dall_price)


