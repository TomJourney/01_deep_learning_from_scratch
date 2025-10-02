from ch05.ch05_5_4_1_multiply_layer_p135 import MulLayer
from ch05.ch05_5_4_2_add_layer_p137 import AddLayer

# 计算图的加法层类的测试用例
apple_money_sum = 100
apple_num = 2
orange_money_sum = 150
orange_num = 3
tax = 1.1

# 创建乘法层与加法层实例
apple_mul_layer_ins = MulLayer() # 苹果乘法层
orange_mul_layer_ins = MulLayer()  # 橘子乘法层
apple_add_layer_ins = AddLayer()  # 苹果加法层
tax_mul_layer_ins = MulLayer()  # 税率乘法层 

# 苹果乘法层前向传播，得到苹果总价（支付总金额）
apple_money_sum = apple_mul_layer_ins.forward(apple_money_sum, apple_num) # (1)
# 橘子乘法层前向传播，得到橘子总价（支付总金额）
orange_money_sum = orange_mul_layer_ins.forward(orange_money_sum, orange_num)  # (2)
# 苹果加法层前向传播，得到苹果+橘子总价（支付总金额）
all_money_sum = apple_add_layer_ins.forward(apple_money_sum, orange_money_sum) #(3)
# 税率乘法层前向传播，得到计税后的支付总金额
taxed_total_money = tax_mul_layer_ins.forward(all_money_sum, tax)

# backward 后向传播
dtaxed_total_money = 1 # 计税后总金额导数为1
# 税率乘法层后向传播：得到计税后总金额对税前总金额偏导，税率偏导
duntax_total_money, dtax = tax_mul_layer_ins.backward(dtaxed_total_money)
# 苹果加法层后向传播： 得到苹果税前总金额偏导，橘子税前总金额偏导
duntax_apple_money_sum, duntax_orange_money_sum = apple_add_layer_ins.backward(duntax_total_money)
# 苹果加法层后向传播： 得到苹果单价偏导，苹果数量偏导
dapple_price, dapple_num = apple_mul_layer_ins.backward(duntax_apple_money_sum)
# 橘子加法层后向传播： 得到橘子单价偏导，橘子数量偏导
dorange_price, dorange_num = orange_mul_layer_ins.backward(duntax_orange_money_sum)

# 打印
print(f"后向传播：税前总金额偏导={duntax_total_money}, 税率偏导={dtax}")
print(f"后向传播：苹果税前总金额偏导={duntax_apple_money_sum}, 橘子税前总金额偏导={duntax_orange_money_sum}")
print(f"后向传播：苹果单价偏导={dapple_price}, 苹果数量偏导={dapple_num}")
print(f"后向传播：橘子单价偏导={dorange_price}, 橘子数量偏导={dorange_num}")
# 后向传播：税前总金额偏导=1.1, 税率偏导=650
# 后向传播：苹果税前总金额偏导=1.1, 橘子税前总金额偏导=1.1
# 后向传播：苹果单价偏导=2.2, 苹果数量偏导=110.00000000000001
# 后向传播：橘子单价偏导=3.3000000000000003, 橘子数量偏导=165.0