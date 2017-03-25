#coding:utf-8
import numpy as np
from matplotlib import pyplot as plt

#生成二元正态分布
def td_normal(mu, sigma, sample_num):
    R = np.linalg.cholesky(sigma)
    data = np.dot(np.random.randn(sample_num, 2), R) + mu
    return data

#生成200条属于同一个二元正态分布的数据
def generate_traindata():
    data = td_normal(np.array([1,5]), np.array([[1,0.5],[1.5,3]]), 200)
    return data

class PCA(object):
    def __init__(self, r):
        self.r = r  #降维后的维度

    def transform(self, X):
        '''
        :param data: (n_samples, n_features)
        :return x_r: 降维后的r维数据
        :return x_recon: 降维后再还原成原来维度的数据
        '''        
        means = np.mean(X, axis=0)     #每个特征的均值
        X_norm = X - means             #X零均值化
        sigma = (X_norm.T.dot(X_norm)) #协方差矩阵， 省略了前面的1.0 / m
        # sigma = np.cov(X_norm, rowvar=0)  #或者调用np.cov求协方差矩阵也可        

        w, v = np.linalg.eig(sigma)  #特征值，已经归一化的特征向量(按列排布)            
        w_in = np.argsort(w) #可以使特征值从小到大排序的下标序列
        w_in = w_in[-(self.r):] #前r大的特征值的下标
        v_need = v[:, w_in]                  #前r大的特征向量
        X_r = X_norm.dot(v_need)             #降维后的数据
        X_recon = X_r.dot(v_need.T) + means  #将降维后的数据映射到原来的空间
        return X_r, X_recon

    def plot(self, X, X_recon):
        plt.scatter(X[:,0], X[:,1], c='blue')
        plt.scatter(X_recon[:,0], X_recon[:,1], c='red')
        plt.show()

if __name__ == '__main__':
    X = generate_traindata()
    pca = PCA(1)
    X_r, X_recon = pca.transform(X)
    pca.plot(X, X_recon)