import copy
import random
from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
from filterpy.common import pretty_str, outer_product_sum
# 贝叶斯滤波器的库
from numpy.random import multivariate_normal
from .utils import reverse_map
import random
from collections import Counter

class EnKFGamma(object):
    """
    """
    def __init__(self, x, P, dim_z, N, hx, sirmodels, n_nodes, task):
        """
        x:list 每个样本的 平均感染率, 平均转化率, 以及当前模型的beta或者gamma
        P: 初始的随机数
        dim_z: 需要估计的参数的数量
        N: 样本数量
        hx: 函数，看返回哪个观测值
        sirmodels: 需要用的已知网络的模型
        n_nodes: 节点数量
        task: 观测任务的种类
        """
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')
        
        dim_x = len(x[0])
        self.n_nodes = n_nodes
        self.dim_x = dim_x
        self.dim_z = dim_z
        # self.dt = dt
        self.N = N
        self.hx = hx
        self.sirmodels = sirmodels
        self.K = zeros((dim_x, dim_z))
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))   # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty
        self.x = x
        self.P = P
        self.Is_last = dict()

        self.initialize(x, P)
        self.Q = eye(dim_x)     # process uncertainty
        self.R = eye(dim_z)      # state uncertainty
        self.inv = np.linalg.inv
        self.delta_status = {}

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

        self.task = task
        self.steps = 0

    def initialize(self, x, P):

        # self.sigmas = multivariate_normal(mean=self.x, cov=self.P, size=self.N) #sample
        self.sigmas = x #sample 初值
        self.x = np.mean(x, axis=0) # 初始均值
        self.P = P # 初始方差

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy() # x_hat
        self.P_prior = self.P.copy() # P_hat

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy() #
        self.P_post = self.P.copy()
    
    def update(self, z, R=None):
        # z是measurement 观测 在这里是感染率
        if z is None:
            self.z = array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R

        N = self.N
        dim_z = len(z)
        sigmas_h = zeros((N, dim_z))

        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i]) #observation

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = (outer_product_sum(sigmas_h - z_mean) / (N-1)) + R
        P_xz = outer_product_sum(
            self.sigmas - self.x, sigmas_h - z_mean) / (N - 1)

        self.S = P_zz
        self.SI = self.inv(self.S)
        self.K = dot(P_xz, self.SI)

        e_r = multivariate_normal(self._mean_z, R, N)
        for i in range(N):
            self.sigmas[i] += dot(self.K, z + e_r[i] - sigmas_h[i])

            if 1:
                '''
                rescale
                '''
                if self.sigmas[i][0] < 0:
                    self.sigmas[i][0] = 0
                if self.sigmas[i][1] < 0:
                    self.sigmas[i][1] = 0
                if self.sigmas[i][0] > 1:
                    self.sigmas[i][0] = 1
                if self.sigmas[i][1] > 1:
                    self.sigmas[i][1] = 1

        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy() # x_{k+1}
        self.P_post = self.P.copy() # P_{k+}

        self.steps += 1
    
    def predict(self, Q=None, windows=1):
        """ Predict next position. """
        if Q is None:
            Q = self.Q
        if np.isscalar(Q):
            Q = eye(self.dim_x) * Q

        N = self.N
        modified_i_list = []
        modified_r_list = []
        for i, s in enumerate(self.sigmas):
            self.sirmodels[i].params['model']['gamma'] = np.random.exponential(reverse_map(self.sigmas[i][-1]), self.n_nodes)
            tmp_i_nums = int(self.n_nodes * self.sigmas[i][0])
            tmp_r_nums = int(self.n_nodes * self.sigmas[i][1])
            tmp_s_nums = self.n_nodes - tmp_i_nums - tmp_r_nums
            counter = Counter(list(self.sirmodels[i].status.values()))
            # print('-------------')
            if self.steps > 0 and (self.steps+1) % windows==0:
                # 以下开关可以选择开或不开，不开效果也很好
                if 0 and counter[1] > tmp_i_nums:
                    # if 0!! case 1
                    # 如果实际的感染人数人数大于修正的人数，则gamma估小了，任选转为r
                    diff_r = counter[1] - tmp_i_nums
                    candidates = []
                    # for key, value in self.sirmodels[i].status.items():
                    #     if value == 1:
                    #         candidates.append(key)
                    first_candidates = []
                    tmp_nums = 0
                    for history_infected in self.delta_status[i]:
                        # 后感染，先退回
                        if tmp_nums >= diff_r:
                            break
                        tmp_removed = []
                        tmp_nums_bak = tmp_nums
                        for idx in history_infected:
                            if self.sirmodels[i].status[idx] == 1:
                                tmp_nums += 1
                                tmp_removed.append(idx)
                        if tmp_nums < diff_r:
                            # 如果还不够
                            first_candidates += tmp_removed
                        else:
                            # 假如超过了
                            r2i_list = np.random.choice(tmp_removed, diff_r-tmp_nums_bak, replace=False)
                            first_candidates += r2i_list.tolist()
                        if tmp_nums >= diff_r:
                            break
                    # if len(candidates) > 0:
                    #     i2r_list = np.random.choice(candidates, min(diff_r, len(candidates)), replace=False) # 不可重复
                    i2r_list = first_candidates
                    for n_k in i2r_list:
                        #if random.random() < 0.8:
                        self.sirmodels[i].status[n_k] = 2 # i->r
                        # else:
                        #     self.sirmodels[i].status[n_k] = 0 # i->r
                    diff_r = None
                counter = Counter(list(self.sirmodels[i].status.values()))
                # if counter[1] > tmp_i_nums:
                #     diff_r = counter[1] - tmp_i_nums
                #     for key, value in self.sirmodels[i].status.items():
                #         if value == 1:
                #             candidates.append(key)
                #     i2r_list = np.random.choice(candidates, min(diff_r, len(candidates)), replace=False)
                #     for n_k in i2r_list:
                #         self.sirmodels[i].status[n_k] = 0 # i->r

                if counter[1] < tmp_i_nums:
                    # 如果实际的感染人数人数小于修正的人数，则gamma估计大了，由Rs返回，则将最新的一部分2回退到1，如果不够，则任选一些2，回退到1
                    diff_i = - counter[1] + tmp_i_nums
                    # print('s->i=',diff_s)
                    candidates = []
                    first_candidates = []
                    tmp_nums = 0
                    for history_infected in self.delta_status[i][::-1]:
                        # 后感染，先退回
                        if tmp_nums >= diff_i:
                            break
                        tmp_removed = []
                        tmp_nums_bak = tmp_nums
                        for idx in history_infected:
                            if self.sirmodels[i].status[idx] == 2:
                                tmp_nums += 1
                                tmp_removed.append(idx)
                        if tmp_nums < diff_i:
                            # 如果还不够
                            first_candidates += tmp_removed
                        else:
                            # 假如超过了
                            r2i_list = np.random.choice(tmp_removed, diff_i-tmp_nums_bak, replace=False)
                            first_candidates += r2i_list.tolist()
                        if tmp_nums >= diff_i:
                            break
                    r2i_list_1 = np.array([])
                    if len(first_candidates) < diff_i:
                        for key, value in self.sirmodels[i].status.items():
                            if value == 2 and key not in first_candidates:
                                candidates.append(key)
                        if candidates:
                            r2i_list_1 = np.random.choice(candidates, min(diff_i-len(first_candidates), len(candidates)), replace=False) # 不可重复

                    r2i_list = r2i_list_1.tolist() + first_candidates
                    self.delta_status[i] = self.delta_status[i] + [r2i_list] # !!
                    for n_k in r2i_list:
                        self.sirmodels[i].status[n_k] = 1 # r->i
                    diff_i = None
            counter = Counter(list(self.sirmodels[i].status.values()))
            modified_i = counter[1] / self.n_nodes
            modified_r = counter[2] / self.n_nodes
            modified_i_list.append(modified_i)
            modified_r_list.append(modified_r)

            iterations = self.sirmodels[i].iteration_bunch(bunch_size=1)
            if self.steps == 0:
                self.delta_status[i] = []
            if iterations[0]['status'] and self.steps > 0:
                self.delta_status[i].append(iterations[0]['status']) # dict, 哪些node从1->2，或者从0->1,或者从
            self.delta_status[i] = self.delta_status[i]
            trends = self.sirmodels[i].build_trends(iterations)
            # self.sigmas[i][0] = trends[0]['trends']['node_count'][0][-1] / self.n_nodes
            # self.sigmas[i][1] = trends[0]['trends']['node_count'][1][-1] / self.n_nodes
            # self.sigmas[i][2] = trends[0]['trends']['node_count'][2][-1] / self.n_nodes
            self.sigmas[i][0] = trends[0]['trends']['node_count'][1][-1] / self.n_nodes
            self.sigmas[i][1] = trends[0]['trends']['node_count'][2][-1] / self.n_nodes
            self.Is_last[i] = [key for key, value in self.sirmodels[i].status.items() if (value==1 and (key not in iterations[0]['status'] or iterations[0]['status'][key]==0))] # 感染，但不是这一步被感染的样本

        # print('average modified i ={}'.format(np.mean(modified_i_list)))
        # print('average modified r ={}'.format(np.mean(modified_r_list)))
        # e = multivariate_normal(self._mean, self.Q, N)
        e = multivariate_normal(self._mean, Q, N)
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0) 
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        def __repr__(self):
            return '\n'.join([
                'EnsembleKalmanFilter object',
                pretty_str('dim_x', self.dim_x),
                pretty_str('dim_z', self.dim_z),
                pretty_str('dt', self.dt),
                pretty_str('x', self.x),
                pretty_str('P', self.P),
                pretty_str('x_prior', self.x_prior),
                pretty_str('P_prior', self.P_prior),
                pretty_str('Q', self.Q),
                pretty_str('R', self.R),
                pretty_str('K', self.K),


                pretty_str('S', self.S),
                pretty_str('sigmas', self.sigmas),
                pretty_str('hx', self.hx),
                pretty_str('fx', self.fx)
        ])