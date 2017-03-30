#coding:utf-8
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.cluster import spectral_clustering

'''
谱聚类：
优点：
(1)适合高维特征稀疏的样本，此时速度快
(2)相当于非线性变换，不要求簇的形状为凸形

缺点：
(1)只适合簇的种类不多的情况
(2)只适合每个簇样本数目较为均匀的情况

(3)同样难以预先确认分几个簇
'''

#生成二元正态分布
def td_normal(mu, sigma, sample_num):
    R = np.linalg.cholesky(sigma)
    data = np.dot(np.random.randn(sample_num, 2), R) + mu
    return data

#生成测试数据 3类
def generate_traindata():
    data1 = td_normal(np.array([1,5]), np.array([[3,1.5],[1.5,3]]), 200)
    data2 = td_normal(np.array([0,1]), np.array([[1,-1.5],[-1.5,3]]), 200)
    data3 = td_normal(np.array([8,0]), np.array([[1,0],[0,6]]), 200)

    data = np.vstack((data1, data2, data3))
    return data

def plot(data, cluster_classes):
        index0 = cluster_classes == 0
        plt.scatter(data[index0][:,0], data[index0][:,1], c='red')
        index1 = cluster_classes == 1
        plt.scatter(data[index1][:,0], data[index1][:,1], c='blue')
        index2 = cluster_classes == 2
        plt.scatter(data[index2][:,0], data[index2][:,1], c='green')

        plt.show()

class SC(object):
    def __init__(self, k, data):
        self.k = k
        self.data = data

    '''
    距离矩阵如何转化为相似性矩阵？
    If you have an affinity matrix, such as a distance matrix, 
    for which 0 means identical elements, and high values means very dissimilar elements, 
    it can be transformed in a similarity matrix that is well suited for the algorithm 
        by applying the  Gaussian (RBF, heat) kernel:
    np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

    similarity = np.exp(-beta * distance / distance.std())
    '''
    def similarity_matrix(self):
        n = self.data.shape[0]
        W = np.zeros((n, n), dtype='float64')
        for i in range(n):
            for j in range(i+1, n):
                W[i][j] = W[j][i] = distance.euclidean(self.data[i], self.data[j])

        W = np.exp(-W / W.std())
        return W

    '''
    相似矩阵 --> 拉普拉斯矩阵
    邻接矩阵 --> 拉普拉斯矩阵
    貌似不一样？
    '''
    def cluster(self):
        W = self.similarity_matrix()
        n = W.shape[0]

        W_column_sums = np.sum(W, axis=0)
        D = np.diag(W_column_sums)

        L = D - W
        Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
        L = np.dot(np.dot(Dn, L), Dn)

        w, v = np.linalg.eig(L)
        w_sort = np.argsort(w)
        w_first_k = w_sort[:self.k]
        spectral_data = v[:, w_first_k]

        KM = K_means(self.k, spectral_data)
        cluster_classes = KM.cluster()
        
        return cluster_classes    

class K_means(object):
    def __init__(self, k, data):
        self.k = k
        self.data = data

    def cluster(self):
        n, m = self.data.shape        

        centroids = np.zeros((self.k, m))
        column_mins = np.min(self.data, axis=0)
        column_maxs = np.max(self.data, axis=0)
        for i in range(m):
            centroids[:, i] = np.random.uniform(column_mins[i], column_maxs[i], (self.k))        

        cluster_changed = True
        cluster_assigned = np.zeros((n))
        while cluster_changed:
            cluster_changed = False
            for i in range(n):
                min_dist = float('inf')
                min_class = -1

                for j in range(self.k):
                    dist = distance.euclidean(self.data[i], centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_class = j

                if min_class != cluster_assigned[i]:
                    cluster_assigned[i] = min_class
                    cluster_changed = True

            for j in range(self.k):
                index = cluster_assigned == j                
                centroids[j] = np.mean(self.data[index], axis=0)               
        return cluster_assigned

def sklearn_sc(W, k):
    labels = spectral_clustering(W, 2)
    return labels


if __name__ == '__main__':
    data = generate_traindata()

    #测试k-means
    # km = K_means(2, data)
    # cluster_classes = km.cluster()    
    # plot(data, cluster_classes)
    
    #测试谱聚类
    sc = SC(3, data)
    cluster_classes = sc.cluster()
    plot(data, cluster_classes)

    #测试sklearn中的spectral_clustering
    # sc = SC(2, data)
    # W = sc.similarity_matrix()
    # cluster_classes = sklearn_sc(W, 2)
    # plot(data, cluster_classes)