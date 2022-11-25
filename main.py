# import Gradient


# 获取数据
# training_data, test_data = Gradient.load_data()
# x = training_data[:, :-1]
# y = training_data[:, -1:]
# print(x[0])
# print(y[0])

# # # 模型计算
# # # 权重
# net = test123.Network(13)
# z = net.forword(x)
# # print("prodict: ", z)

# loss = net.loss(y, z)
# print(loss)


# net = Gradient.Network(13)

# # 只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
# w5 = Network.np.arange(-160.0, 160.0, 1.0)
# w9 = test123.np.arange(-160.0, 160.0, 1.0)
# losses = test123.np.zeros([len(w5), len(w9)])
#
# # 计算设定区域内每个参数取值所对应的Loss
# for i in range(len(w5)):
#     for j in range(len(w9)):
#         net.w[5] = w5[i]
#         net.w[9] = w9[j]
#         z = net.forward(x)
#         loss = Gradient.loss(z, y)
#         losses[i, j] = loss
#
# # 使用matplotlib将两个变量和对应的Loss作3D图
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# ax = plt.figure().add_subplot(projection='3d')
#
# w5, w9 = Gradient.np.meshgrid(w5, w9)
#
# ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap = 'rainbow')
# plt.show()



# """
# 梯度计算
# """
# import Gradient
#
#
# # 获取数据
# training_data, test_data = Gradient.load_data()
# x = training_data[:, :-1]
# y = training_data[:, -1:]
#
#
# net = Gradient.Network(13)
# # 设置[w5, w9] = [-100., -100.]
# net.w[5] = -100.0
# net.w[9] = -100.0
#
# z = net.forward(x)
# losses = net.loss(y, z)
# gradient_w, gradient_b = net.gradient(x, y)
# gradient_w5 = gradient_w[5][0]
# gradient_w9 = gradient_w[9][0]
#
# print('point {}, losses {}'.format([net.w[5][0], net.w[9][0]], losses))
# print('gradient {}'.format([gradient_w5, gradient_w9]))
#
#
# """
# 梯度更新
# """
# # 在[w5, w9]的平面上，沿着梯度方向移动到下一个点P1
# # 定义移动的步长 eta
# eta = 0.1
# # 更新参数w5 w9
# net.w[5] = net.w[5] - eta * gradient_w5
# net.w[9] = net.w[9] - eta * gradient_w9
# print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], losses))
# print('gradient {}'.format([gradient_w5, gradient_w9]))




