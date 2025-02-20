# SIR模型的模拟函数
# 预测函数
from tqdm import tqdm
import math
import numpy as np
def map(x):
    return math.tan((x-1/2)*math.pi) / 300

def reverse_map(y):
    return math.atan(300*y) / math.pi + 1/2

def sir(I, R, beta, gamma, dt=1):
    dI = beta * (1-I-R) * I  - gamma * I
    dR = gamma * I
    return I + dI*dt, R + dR*dt

# 观测矩阵，这里假设观测数据是I和R的线性组合
H = np.diag([1,1,0,0])

# 还是我自己写吧
# 状体 I, R, beta, gamma
# trends = sir_model.trends
def ensemble_kalman_filter(observations, Q_beta, Q_gamma, R_cov, fraction, ensemble_size=50, steps=3000):
    beta_ensemble = [map(i) if i>=0 else map(0.01) for i in np.random.normal(0.01, Q_beta, ensemble_size)]
    gamma_ensemble = [map(i) if i>=0 else map(0.01) for i in np.random.normal(0.01, Q_gamma, ensemble_size)]
    # beta_ensemble = [map(0.01) for i in range(ensemble_size)]
    # gamma_ensemble = [map(0.01) for i in range(ensemble_size)]
    # 初始化状态集合
    I_ensemble = np.full(ensemble_size, fraction)
    R_ensemble = np.full(ensemble_size, 0.)
    
    x_ensemble = np.vstack((I_ensemble, R_ensemble, beta_ensemble, gamma_ensemble))
    print(x_ensemble.shape)
    estimates = []
    process = []
    
    for t in tqdm(range(steps)):
        # if t%500 == 0:
        #     R_cov/=5
        R_matrix = np.diag([R_cov, R_cov])
        # 模拟SIR模型 预测步
        process.append(x_ensemble)
        for j in range(ensemble_size):
            x_ensemble[0,j], x_ensemble[1,j] = sir(x_ensemble[0,j], x_ensemble[1,j], reverse_map(x_ensemble[2,j]), reverse_map(x_ensemble[3,j]))
        #计算状态均值
        
        y = x_ensemble[0:2,:]
        # 计算集合协方差
        P = np.cov(y) + R_matrix
        # print(x_ensemble.mean(axis=0).shape)
        # 计算集合观测状态协方差
        Pxy = (x_ensemble-x_ensemble.mean(axis=1).reshape(-1,1))@(y - y.mean(axis=1).reshape(-1,1)).T/(ensemble_size-1)
        # 计算卡尔曼增益
        K = Pxy@np.linalg.inv(P)
        # 生成随机数
        eta = np.random.multivariate_normal(np.zeros(2), np.diag([Q_beta, Q_gamma]), ensemble_size).T
        # 更新参数
        i=1
        # print((observations[[t],:].T - y[:,i] + eta[:,i]).shape)
        # print(y[:,i].shape)
        # print(eta[:,i].shape)
        # print(observations[t,:].T.shape)
        # print(K.shape
        for i in range(ensemble_size):
            x_ensemble[:,i] +=  K@(observations[t,:].T - y[:,i] + eta[:,i])
        estimates.append(x_ensemble.mean(axis=1))
    return np.array(estimates), np.array(process)
        
        
        
# 生成观测
# observations = np.vstack((trends[0]["trends"]['node_count'][1], trends[0]["trends"]['node_count'][2])).T/8846
        
# estimated, p = ensemble_kalman_filter(observations, Q_beta = 0.02, Q_gamma= 0.008, R_cov = 0.01, fraction = 0.002, ensemble_size=50, steps=3000)    