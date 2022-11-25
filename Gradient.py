import numpy as np

def load_data():
    # 从文件导入数据
    datafile = 'D:\BaiduSyncdisk\Evan\Evan\Evan\Work\BaiduDeepLearning01\data\p1\housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), \
                               data.sum(axis=0) / data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
        # print(data)

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    testing_data = data[offset:]
    return training_data, testing_data






class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.0

    # 前向计算，公式XA+B
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    # 误差函数(Z-Y)^2
    def loss(self, y, z):
        error = z - y
        cost = error ** 2

        return np.mean(cost)

    # 梯度计算函数
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)

        return gradient_w, gradient_b

    def update(self, gradient_w5, gradient_w9, eta=0.01):
        self.w[5] = self.w[5] - eta * gradient_w5
        self.w[9] = self.w[9] - eta * gradient_w9

    def train(self, x, y, iterations=100, eta=0.01):
        points = []
        losses = []
        for i in range(iterations):
            points.append([self.w[5][0], self.w[9][0]])
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            gradient_w5 = gradient_w[5][0]
            gradient_w9 = gradient_w[9][0]
            self.update(gradient_w5, gradient_w9, eta)
            losses.append(L)
            if i % 50 == 0:
                print('iter {}, point {}, loss {}'.format(i, [self.w[5][0], self.w[9][0]], L))
        return points, losses

# net = Network(13)
#
# # 数据获取
# train_data, test_data = load_data()
# ix = train_data[:, :-1]
# iy = train_data[:, -1:]
# iz = net.forward(ix)
#
# # 取第一个数据作为样本
# x1 = ix[0]
# y1 = ix[0]
# z1 = net.forward(x1)
#
# # for x_i in x1:
# #     gradient_w = (z1 - y1) * x_i
# #     print('gradient_w {}'.format(gradient_w))
#
# # gradient_w = (z1 - y1) * x1
# # print('gradient_w_by_sample1 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
#
# # x3sample = x[0:3]
# # y3sample = y[0:3]
# # z3sample = net.forward(x3sample)
# #
# # gradient_w = (z3sample - y3sample) * x3sample
# # print('gradient_w {}, gradient_shape {}'.format(gradient_w, gradient_w.shape))
#
# gradient_w = (iz - iy) * ix
# # print(gradient_w)
# # axis = 0 表示把每一行做相加然后再除以总的行数
# gradient_w = np.mean(gradient_w, axis=0)
# gradient_w = gradient_w[:, np.newaxis]
#
# gradient_b  = (iz - iy)
# gradient_b = np.mean(gradient_b)
# # print(gradient_b)