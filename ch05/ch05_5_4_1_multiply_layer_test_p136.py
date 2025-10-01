import ch05_5_4_1_multiply_layer_p135 as multiply_layer

# 计算图的乘法层类-测试用例
apple_price = 100
apple_num = 2
tax = 1.1

# layer
multi_apple_layer_instance = multiply_layer.MulLayer() # 苹果乘法层
multi_tax_layer_instance = multiply_layer.MulLayer()  # 税率乘法层

# 苹果-前向传播
apple_price = multi_apple_layer_instance.forward(apple_price, apple_num)
# 把苹果总金额与税额向前传播，得到总金额
total_money = multi_tax_layer_instance.forward(apple_price, tax)

# 打印正向传播计算的支付的总金额
print("\n\n===打印正向传播计算的支付的总金额")
print(total_money) # 220.00000000000003

# 后向传播： 计算各个变量的导数
dprice = 1
# 税率乘法层的后向传播，得到苹果总价偏导数，税率偏导数
dapple_total_money, dtax = multi_tax_layer_instance.backward(dprice)
# 苹果乘法层的后向传播，得到苹果单价偏导数，苹果个数偏导数
dapple_price, dapple_num = multi_apple_layer_instance.backward(dapple_total_money)

# 打印通过后向传播计算得到的各变量的导数
print("\n=== 打印通过后向传播计算得到的各变量的导数")
print(dapple_price, dapple_num, dtax) # 2.2 110.00000000000001 200

