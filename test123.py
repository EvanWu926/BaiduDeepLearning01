# 导入需要用到的package
import numpy as np
import json


class Data2One:

    def __init__(self) -> None:
        pass

    def load_data(self):
        # 从文件导入数据
        datafile = 'D:\BaiduSyncdisk\Evan\Evan\Evan\Work\PY\data\p1\housing.data'
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
        test_data = data[offset:]
        return training_data, test_data


# 误差函数(Z-Y)^2
def loss(y, z):
    error = z - y
    cost = error ** 2

    return np.mean(cost)


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.0

    # 前向计算，公式XA+B
    def forword(self, x):
        z = np.dot(x, self.w) + self.b
        return z


