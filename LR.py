#coding:utf-8
import numpy as np
import math
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

#生成二元正态分布
def td_normal(mu, sigma, sample_num):
    R = np.linalg.cholesky(sigma)
    data = np.dot(np.random.randn(sample_num, 2), R) + mu
    return data

#生成测试数据
def generate_traindata():
    data1 = td_normal(np.array([1,5]), np.array([[2,0.5],[1.5,3]]), 200)
    data2 = td_normal(np.array([2,3]), np.array([[2,0.5],[1.5,3]]), 200)

    train_X = np.vstack((data1, data2))
    train_y = np.array([1]*200 + [0]*200).reshape((400,1))   
    train_data = np.hstack((train_X, train_y))
    np.random.shuffle(train_data)
    return train_data

class LR(object):
    def __init__(self):
        pass

    def _label_encode(self, train_y):
        label_index = 0
        labels = {}
        indexs = {}
        for yi in train_y:
            if yi not in labels:
                labels[yi] = label_index
                indexs[label_index] = yi
                label_index += 1
        return labels, indexs        

    def fit(self, train_X, train_y):
        # self.labels, self.indexs = self._label_encode(train_y)
        self.train_X = train_X
        
        self.n = len(train_X)
        self.f = len(train_X[0]) if self.n else 0

        self.samples = np.hstack((train_X,np.ones((self.n, 1))))
        self.labels = train_y
        classes = 2 #现在只考虑了二元分类

        self.w = np.random.uniform(size=(self.f + 1))
        self.alpha = 0.1

        if classes == 2:
            self._sgd_sigmoid()            
            print u"skleanr准确率: ", LogisticRegression().fit(train_X,train_y).score(train_X, train_y)                        
        elif classes > 2:
            pass

    #sigmoid - h(x)
    def _h(self, x):
        return 1.0 / (1 + math.exp(-1.0 * self.w.dot(x)))

    #对数似然函数
    def _log_likelihood(self):
        res = 0
        for i in range(self.n):
            res += self.labels[i] * math.log(self._h(self.samples[i])) + (1 - self.labels[i]) * math.log((1 - self._h(self.samples[i])))
        return res

    #随机梯度下降
    def _sgd_sigmoid(self, max_iter=100):
        for _ in range(max_iter):
            for i in range(self.n):
                self.w = self.w + self.alpha * (self.labels[i] - self._h(self.samples[i])) * self.samples[i]
                #最大似然比较
                # print self._log_likelihood()

    def plot(self):
        index1 = train_y == 1
        index2 = train_y == 0
        plt.plot(self.train_X[index1][:,0], self.train_X[index1][:,1], 'b+')
        plt.plot(self.train_X[index2][:,0], self.train_X[index2][:,1], 'r+')

        x = np.arange(-3,7,0.5)
        y = (-self.w[2]-self.w[0]*x)/self.w[1]
        # y1 = (-self.wcf[0] * x) / self.wcf[1]

        plt.plot(x,y)
        # plt.plot(x,y1)
        plt.show()

    def score(self):
        test_X = self.samples
        wrong = 0
        nn = len(test_X)
        for i in range(nn):
            p1 = self._h(test_X[i])
            yi = 1 if p1 > 0.5 else 0
            if yi != self.labels[i]:
                wrong += 1
        print u'我的准确度: ', 1 - wrong * 1.0 / self.n

if __name__ == '__main__':
    train_data = generate_traindata()
    train_X = train_data[:,:-1]
    train_y = train_data[:,-1]

    lr = LR()
    lr.fit(train_X, train_y)        
    lr.score()

    lr.plot()