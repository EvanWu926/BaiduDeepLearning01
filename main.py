import test123

d2o = test123.Data2One()

# 获取数据
training_data, test_data = d2o.load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]
# print(x[0])
# print(y[0])

# # # 模型计算
# # # 权重
# net = test123.Network(13)
# z = net.forword(x)
# # print("prodict: ", z)

# loss = net.loss(y, z)
# print(loss)


net = test123.Network(13)

# 只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = test123.np.arange(-160.0, 160.0, 1.0)
w9 = test123.np.arange(-160.0, 160.0, 1.0)
losses = test123.np.zeros([len(w5), len(w9)])

# 计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forword(x)
        loss = test123.loss(z, y)
        losses[i, j] = loss

# 使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure().add_subplot(projection='3d')

w5, w9 = test123.np.meshgrid(w5, w9)

ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap = 'rainbow')
plt.show()
