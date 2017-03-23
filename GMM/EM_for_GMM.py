#coding:utf-8
#wangliang 2017/3/22

from matplotlib import pyplot as plt
import numpy as np
import random

'''
使用EM算法实现的GMM。 数据从文件中读取，由三个正态分布生成。
具体E步和M步推导的结果见《统计学习方法》。 本源码依据书上的推导结果写成。

'''

color = ['bo', 'go', 'ro'] #三种可能不够用，指定k>3时请补充，否则会溢出

def guassian(x, mu, sigma):
    '''
    多元正态分布
    '''
    norm_factor = 1.0 / np.linalg.det(sigma)
    res = norm_factor * np.exp(-0.5 * np.transpose(x - mu).dot(np.linalg.inv(sigma)).dot(x-mu))
    return res

class GMM(object):
    def __init__(self, data, k):
        '''
        :param data: numpy格式的数据，(n_samples, n_featuers)
        :param k: 指定高斯分布数目
        '''

        #数据的基本信息
        self.data = data
        self.k = k  #需要指定正态分布的个数
        self.n = data.shape[0]
        self.d = data.shape[1]

        ############gmm的参数
        #初始化参数，注意不能全都一样，否则会导致self.Q不满秩，求逆矩阵时出现singular matrix的错误
        self.mu = np.zeros((self.k, self.d), dtype='float64')
        for i in range(self.k):
            self.mu[i] = random.gauss(0,1)  #多跑几次就会发现，初值的选择真的很重要！
        self.sigma = np.zeros((self.k, self.d, self.d), dtype='float64')
        for i in range(self.k):
            self.sigma[i] = np.diag((1.0,)*self.d)
        self.alpha = np.array([1.0 / self.k for i in range(self.k)], dtype='float64')
        ############gmm的参数

        #存储第i个样本来自第j个高斯分布的后验概率，用于M步中更新参数
        self.Q = np.zeros((self.n, self.k), dtype='float64')

    def expectation(self):
        '''
        第i个样本来自第j个高斯分布的后验概率
        '''
        for i in range(self.n):
            sum_proba = 0
            for j in range(self.k):
                self.Q[i][j] = self.alpha[j] * guassian(self.data[i], self.mu[j], self.sigma[j])
                sum_proba += self.Q[i][j]
            for j in range(self.k):
                self.Q[i][j] = self.Q[i][j] / sum_proba

    def maximum(self):
        '''
        更新参数： alpha, mu, sigma
        '''
        for i in range(self.k):
            mu_s = np.zeros((1, self.d), dtype='float64')
            for j in range(self.n):
                mu_s += self.Q[j][i] * self.data[j]
            self.mu[i] = mu_s / np.sum(self.Q[:,i])

            sigma_s = np.zeros((self.d, self.d), dtype='float64')
            for j in range(self.n):
                sigma_s += self.Q[j][i] * np.transpose(np.mat(self.data[j] - self.mu[i])).dot(np.mat(self.data[j] - self.mu[i]))
                # sigma_s += self.Q[j][i] * np.transpose(self.data[j] - self.mu[i]).dot(self.data[j] - self.mu[i]) #为什么这样会导致singular matrix?
            self.sigma[i] = sigma_s / np.sum(self.Q[:, i])

            self.alpha[i] = sum(self.Q[:, i]) / self.n

    def EM(self, iter_times=20):
        for i in range(iter_times):
            self.expectation()
            self.maximum()

    def cluster(self):
        '''
        聚类
        '''
        proba = np.zeros((self.k), dtype='float64')
        classes = np.zeros((self.n),dtype='int64')
        for i in range(self.n):
            for j in range(self.k):
                proba[j] = self.alpha[j] * guassian(self.data[i], self.mu[j], self.sigma[j])
            classes[i] = proba.argmax()

        for i in range(self.k):  #高维的数据只能画出前两维, 所以高维数据的聚类结果可能看起来有“离群点”
            plt.plot(self.data[classes==i][:,0], self.data[classes==i][:,1], color[i])

        plt.show()

def read_data():
    dat = []
    f = open("./gmm_data.txt")
    for line in f:
        dat.append(map(float,line.strip().split(" "))[1:])
    return np.array(dat)

def iris_data():
    from sklearn import datasets
    iris = datasets.load_iris()
    return iris.data

if __name__ == '__main__':
    # data = read_data()
    data = iris_data()
    a = GMM(data, 3)
    a.EM(20)
    a.cluster()