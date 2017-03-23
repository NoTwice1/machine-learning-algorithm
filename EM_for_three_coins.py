#coding:utf-8

'''
假设有3枚硬币，分别记做A,B,C. 这些硬币正面出现的概率为phi, p, q. 进行如下抛掷实验： 先掷硬币A，如果为正面，则
继续掷硬币B； 反之，则继续掷硬币C。 对于A选出的硬币B或C，记录其抛掷的结果，正面为1，反面为0. 独立地重复10次的结果
如下：
    1101001011
假设只能看到掷硬币的结果，不能看到过程。 估计phi, p, q.

E步：  p(z = b|x,theta)
M步： argmax 
'''


#结果: EM算法对初始值太敏感了..

result = '1101001011'
iter_times = 10

def Expectation(Y, phi, p, q):  #P(z | Y,theta_i) z == B: 每个观测样本的观测值来自骰子B的后验概率. 则来自骰子C的后验概率(1 - u)
    n = len(Y)
    u = [0.0] * n
    for i in range(n):
        y = int(Y[i])
        a = phi * p ** y * (1 - p) ** (1 - y) 
        b = a + (1 - phi) * q ** y * (1 - q) ** (1 - y)
        u[i] = a * 1.0 / b
    return u

def Maximum(Y, u):
    n = len(Y)
    phi = 1.0 / n * sum(u)

    a = 0
    for i in range(n):
        a += u[i] * int(Y[i])
    p = a * 1.0 / sum(u)

    a = 0
    for i in range(n):
        a += (1 - u[i]) * int(Y[i])
    q = a * 1.0 / (n - sum(u))
    return phi, p, q

def EM(Y):
    # phi, p, q = 0.5, 0.5, 0.5
    phi, p, q = 0.4, 0.6, 0.7
    for i in range(iter_times):
        u = Expectation(Y, phi, p, q)

        phi, p, q = Maximum(Y, u)
        print phi, p, q
    return phi, p, q

if __name__ == "__main__":
    EM(result)