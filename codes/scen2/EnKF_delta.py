import copy
import random
from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
from numpy.random import multivariate_normal
from filterpy.common import pretty_str, outer_product_sum
from utils import reverse_map
from collections import Counter
# 要注意设定中，这里beta是感染概率
class EnKFBeta(object):
    """
    """

    def __init__(self, x, P, dim_z, N, hx, sirmodels, n_nodes, task):  #hx是函数
        #x是状态 It, Rt, beta
        #dim_z 是观测量的维数
        #N是样本数量
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x[0])
        self.n_nodes = n_nodes
        self.dim_x = dim_x
        self.dim_z = dim_z
        # self.dt = dt
        self.N = N  #the number of particles
        self.hx = hx
        self.sirmodels = sirmodels # models
        self.K = zeros((dim_x, dim_z))  # z is the dimension of obeservation
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))   # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty
        self.x = x   # initial mean
        self.P = P   #initial covariance
        self.Is_last = dict()

        self.initialize(x, P)
        self.Q = eye(dim_x)     # process uncertainty
        self.R = eye(dim_z)      # state uncertainty
        self.inv = np.linalg.inv  # solve the inv
        self.delta_status = {}

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

        self.task = task
        self.steps = 0


    def initialize(self, x, P):
        """
        Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__
        Parameters
        ----------
        x : np.array(dim_z)
            state mean
        P : np.array((dim_x, dim_x))
            covariance of the state
        """

        # if self.x.ndim != 1:
        #     raise ValueError('x must be a 1D array')

        # self.sigmas = multivariate_normal(mean=self.x, cov=self.P, size=self.N) #sample
        self.sigmas = x #sample
        self.x = np.mean(x, axis=0)
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()  #Pminus

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, R=None):
        """
        update step
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.
        Parameters
        ----------
        z : np.array
            measurement for this update.  observation?
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        """

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
        sigmas_h = zeros((N, dim_z))  #H

        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i])  # observation

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = (outer_product_sum(sigmas_h - z_mean) / (N-1)) + R  #生成样本协方差矩阵
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
                # if self.sigmas[i][0] + self.sigmas[i][1] > 1:
                #     # self.sigmas[i][0] /= self.sigmas[i][0] + self.sigmas[i][1]
                #     # self.sigmas[i][1] /= self.sigmas[i][0] + self.sigmas[i][1]
                #     self.sigmas[i][1] = 1 - self.sigmas[i][0] # 感染率更为准确

        self.x = np.mean(self.sigmas, axis=0)
        # if self.x[1] > 0.1:
        #     print('over 0.1 before: ', self.x_prior)
        #     print('over 0.1 after: ', self.x)
        self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

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

        #更新参数
        for i, s in enumerate(self.sigmas):
            self.sirmodels[i].params['model']['beta'] = np.random.exponential(reverse_map(self.sigmas[i][-1]), self.n_nodes)
            tmp_i_nums = int(self.n_nodes * self.sigmas[i][0])  # 模拟理论应该的感染比率
            tmp_r_nums = int(self.n_nodes * self.sigmas[i][1])
            tmp_s_nums = self.n_nodes - tmp_i_nums - tmp_r_nums
            counter = Counter(list(self.sirmodels[i].status.values()))   # 模拟实际的感染比例
            # print('-------------')
            # print('raw: ', counter)
            if self.steps > 0 and (self.steps+1) % windows==0:
                if counter[1] < tmp_i_nums:
                    diff_s = - counter[1] + tmp_i_nums # 相差的人数
                    # print('i={}, s->i='.format(i),diff_s)
                    s2i_list = np.array([])
                    s2i_list2 = np.array([])
                    candidates2 = []
                    for key in self.Is_last[i]:
                        if self.sirmodels[i].status[key] == 1:
                            # 寻找其邻居
                            susceptible_neighbors = [v for v in self.sirmodels[i].graph.neighbors(key) if self.sirmodels[i].status[v] == 0]
                            candidates2 += susceptible_neighbors  #寻找所有感染节点的易感邻居
                    candidates2 = [ty for ty in candidates2]
                    if len(candidates2) > 0:
                        s2i_list = np.random.choice(candidates2, max(min(diff_s//1, len(candidates2)//1), 1), replace=False) # 不可重复
                    if len(s2i_list) < diff_s:
                        candidates = []
                        for key, value in self.sirmodels[i].status.items():
                            if value == 0 and key not in s2i_list:
                                candidates.append(key)
                        if len(candidates) > 0:
                            s2i_list2 = np.random.choice(candidates, min(diff_s-len(s2i_list), len(candidates)), replace=False) # 不可重复

                    s2i_list = s2i_list.tolist() + s2i_list2.tolist()
                    for n_k in s2i_list:
                        self.sirmodels[i].status[n_k] = 1 # s->i  逐个修改模型状态
                if counter[1] > tmp_i_nums:
                    # 如果实际的感染人群人数大于修正的人数，则将最新的一部分1回退到0，如果不够，则任选一些1，回退到0
                    diff_i = counter[1] - tmp_i_nums
                    # print('s->i=',diff_s)
                    candidates = []
                    first_candidates = []
                    tmp_nums = 0
                    for history_infected in self.delta_status[i][::-1]:  #把iteration 倒序
                        if tmp_nums >= diff_i:
                            break
                        tmp_infected = []
                        tmp_nums_bak = tmp_nums
                        for idx in history_infected:
                            if self.sirmodels[i].status[idx] == 1:
                                tmp_nums += 1
                                tmp_infected.append(idx)
                        if tmp_nums < diff_i:
                            # 如果还不够
                            first_candidates += tmp_infected
                        else:
                            # 假如超过了
                            i2s_list = np.random.choice(tmp_infected, diff_i-tmp_nums_bak, replace=False)
                            first_candidates += i2s_list.tolist()
                        if tmp_nums >= diff_i:
                            break
                    i2s_list_1 = np.array([])
                    if len(first_candidates) < diff_i:
                        for key, value in self.sirmodels[i].status.items():
                            if value == 1 and key not in first_candidates:
                                candidates.append(key)
                        i2s_list_1 = np.random.choice(candidates, diff_i-len(first_candidates), replace=False) # 不可重复
                        i2s_list = i2s_list_1.tolist() + first_candidates
                    else:
                        i2s_list = first_candidates
                    # print('diff_i={}, first_nums={}, others={}'.format(diff_i, len(first_candidates), len(i2s_list_1)))
                    for n_k in i2s_list:
                        self.sirmodels[i].status[n_k] = 0 # i->s
            counter = Counter(list(self.sirmodels[i].status.values()))
            modified_i = counter[1] / self.n_nodes
            modified_r = counter[2] / self.n_nodes
            modified_i_list.append(modified_i)  #被动表完成，是修正之后的感染频率
            modified_r_list.append(modified_r)
            # iterations是每个step的iteration，iteration储存了每个点的状态
            iterations = self.sirmodels[i].iteration_bunch(bunch_size=1)
            if self.steps == 0:
                self.delta_status[i] = []
            if iterations[0]['status'] and self.steps > 0:
                self.delta_status[i].append(iterations[0]['status']) # dict, 哪些node从1->2，或者从0->1,或者从  因为iteration里面放的都是每次演进的变化量
            trends = self.sirmodels[i].build_trends(iterations)
            # self.sigmas[i][0] = trends[0]['trends']['node_count'][0][-1] / self.n_nodes
            # self.sigmas[i][1] = trends[0]['trends']['node_count'][1][-1] / self.n_nodes
            # self.sigmas[i][2] = trends[0]['trends']['node_count'][2][-1] / self.n_nodes
            self.sigmas[i][0] = trends[0]['trends']['node_count'][1][-1] / self.n_nodes  #最新的演化后的结果
            self.sigmas[i][1] = trends[0]['trends']['node_count'][2][-1] / self.n_nodes
            # self.Is_last[i] = [key for key, value in self.sirmodels[i].status.items() if (value==1 and (key not in iterations[0]['status'] or iterations[0]['status'][key]==0 ))] # 感染，但不是这一步被感染的样本
            self.Is_last[i] = [key for key, value in self.sirmodels[i].status.items() if (value==1 and (key not in iterations[0]['status']))]
            # iteration 记录的是每次发生变化的节点当前的状态
        # print('average modified i ={}'.format(np.mean(modified_i_list)))
        # print('average modified r ={}'.format(np.mean(modified_r_list)))
        # e = multivariate_normal(self._mean, self.Q, N)
        e = multivariate_normal(self._mean, Q, N)  #均值是0
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0)  #对所有样本求平均
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)  #计算样本方差

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