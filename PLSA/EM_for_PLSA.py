#coding:utf-8
import numpy as np
from collections import defaultdict
import math

class PLSA(object):
    def __init__(self, corpus, topic):
        def get_w(corpus):
            words = defaultdict(int)
            for doc in corpus:
                for word in doc:
                    words[word] = 1
            return words

        #数据基本信息
        self.corpus = corpus      
        self.topic = topic   #主题个数

        self.d = len(self.corpus)  #文档数目
        self.words = get_w(corpus)    #所有词汇， 不计重复;
        self.w = len(self.words)  #总词汇数目，不计重复; 
        self.each = map(sum, map(lambda x:x.values(), corpus)) #每篇文档的词汇数目，在M步需要用到，便于计算

        #######plsa的参数
        #初始化参数
        self.dz = np.random.uniform(low=0.0, high=1.0, size=(self.d, self.topic))  #p(z|d)， 对每篇文档，需归一化
        row_sum = np.sum(self.dz, axis=1)
        for i in range(self.d):
            self.dz[i] /= row_sum[i]

        self.zw = np.random.uniform(low=0.0, high=1.0, size=(self.topic, self.w))  #p(w|z), 对每个主题，需归一化
        row_sum = np.sum(self.zw, axis=1)
        for i in range(self.topic):
            self.zw[i] /= row_sum[i]
        #######plsa的参数

        #p(z|w,d), 用于存放E步的后验概率:  一篇文档中的一个词来自于某个主题的后验概率
        self.z_wd = np.zeros((self.d, self.w, self.topic))

    def _expectation(self):
        for i in range(self.d):
            for j in range(self.w):
                sum_proba = 0
                for k in range(self.topic):
                    self.z_wd[i][j][k] = self.dz[i][k] * self.zw[k][j]
                    sum_proba += self.z_wd[i][j][k]
                for k in range(self.topic):
                    self.z_wd[i][j][k] /= sum_proba

    def _maximum(self):
        for k in range(self.topic):            
            for i in range(self.d):                
                self.dz[i][k] = 0
                for j in range(self.w):
                    w = self.words[j]
                    count = self.corpus[i][w] if w in self.corpus[i] else 0
                    self.dz[i][k] += count * self.z_wd[i][j][k]
                self.dz[i][k] /= self.each[i]

        for k in range(self.topic):                        
            for j in range(self.w):
                self.zw[k][j] = 0
                w = self.words[j]
                count = self.corpus[i][w] if w in self.corpus[i] else 0 
                for i in range(self.d):
                    self.zw[k][j] += count * self.z_wd[i][j][k]
            norm = np.sum(self.zw[k])
            for j in range(self.w):
                self.zw[k][j] /= norm

    def _log_likelihood(self):
        likelihood = 0
        for i in range(self.d):
            for j in range(self.w):
                tmp = 0
                w = self.words[j]
                count = self.corpus[i][w] if w in self.corpus[i] else 0 
                for k in range(self.topic):       #从各个主题中生成一个该词的概率之和
                    tmp += count * self.dz[i][k] * self.zw[k][j]
                if tmp > 0:                       #生成count个该词的log概率
                    likelihood += self.corpus[i][w] * math.log(tmp)
        return likelihood


    def train_with_EM(self, max_iter=20):
        cur = 0
        for i in range(max_iter):
            print "iter %d time" % i
            self._expectation()
            self._maximum()
            print "log likelihood: ", self._log_likelihood()

if __name__ == '__main__':
    a = PLSA()
    a.train_with_EM()