from Gradient import *
import matplotlib.pyplot as plt

# # 新建一个array
# a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# print('before shuffle', a)
# np.random.shuffle(a)
# print('after shuffle', a)

# # 新建一个array
# a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# a = a.reshape([6, 2])
# print('before shuffle\n', a)
# np.random.shuffle(a)
# print('after shuffle\n', a)

# # 获取数据
# train_data, test_data = load_data()
#
# # 打乱样本
# np.random.shuffle(train_data)
#
# # 将train_data分割成多个batch
# batch_size = 10
# n = len(train_data)
# mini_batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
# print(mini_batches)
#
# # 创建网络
# net = Network(13)
#
# # 依次使用mini_batch的数据
# for mini_batch in mini_batches:
#     x = mini_batch[:, :-1]
#     y = mini_batch[:, -1:]
#     loss = net.train(x, y, iterations= 100)

train_data, test_data = load_data()

net = Network(13)
losses = net.train(train_data, num_epochs=50, batch_size=10, eta=0.01)

# # 画出损失函数的变化趋势
# plot_x = np.arange(len(losses))
# plot_y = np.array(losses)
# plt.plot(plot_x, plot_y)
# plt.show()
np.save('w.npy', net.w)
np.save('b.npy', net.b)