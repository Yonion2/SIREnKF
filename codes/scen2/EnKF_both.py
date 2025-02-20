import copy
import random
from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
from numpy.random import multivariate_normal
from filterpy.common import pretty_str, outer_product_sum
from utils import reverse_map
import random
from collections import Counter

class EnKFBoth(object):
    """
    """

    def __init__(self, x, P, dim_z, N, hx, sirmodels, n_nodes, task):
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
        self.status_time = dict() # self.status_time[i][j][1]: 第i个粒子第j个node由0->1的时间；self.status_time[i][j][2]: 由1->2的时间

        self.initialize(x, P)
        self.Q = eye(dim_x)     # process uncertainty
        self.R = eye(dim_z)      # state uncertainty
        self.inv = np.linalg.inv
        self.delta_status = {i:None for i in range(N)}

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

        self.task = task
        self.steps = 0
        self.ratio = [0] * N
        self.diff = [0] * N
        self.s2i_nums = [0] * N
        self.i2r_nums = [0] * N


    def initialize(self, x, P):

        # self.sigmas = multivariate_normal(mean=self.x, cov=self.P, size=self.N) #sample
        self.sigmas = x #sample
        self.x = np.mean(x, axis=0)
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # 初始化的time step为0
        for i in range(self.N):
            self.status_time[i] = dict()
            for j in range(self.n_nodes):
                self.status_time[i][j] = {1: None, 2: None}
            keys1 = [key for key, value in self.sirmodels[i].status.items() if value == 1]
            keys2 = [key for key, value in self.sirmodels[i].status.items() if value == 2]
            for key in keys1:
                self.status_time[i][key][1] = 0
            for key in keys2:
                self.status_time[i][key][2] = 0

    def update(self, z, R=None):

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
            delta_sigmas =  dot(self.K, z + e_r[i] - sigmas_h[i])
            delta_sigmas[1] = min(delta_sigmas[1], self.sigmas[i][0]*min(reverse_map(delta_sigmas[-1]), 0.1))
            self.sigmas[i] += delta_sigmas

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
            # 更新参数

        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z) 
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def run_i2r(self, diff_i2r, i):
        times_node = dict()
        for key, value in self.status_time[i].items():
            # 必须要求当前状态为1
            if value[1] is not None and self.sirmodels[i].status[key]==1:
                if value[1] not in times_node:
                    times_node[value[1]] = set()
                times_node[value[1]].add(key)
        keys_sorted = sorted(list(times_node.keys()), reverse=False)
        # 先感染的先前进
        i2r_list = []
        for key in keys_sorted:
            if len(i2r_list) >= diff_i2r:
                break
            values = list(times_node[key])
            if len(i2r_list) + len(values) > diff_i2r:
                tmp_list = np.random.choice(values, diff_i2r-len(i2r_list), replace=False)
                i2r_list += tmp_list.tolist()
            else:
                i2r_list += list(values)
        for n_k in i2r_list:
            self.sirmodels[i].status[n_k] = 2 # i->r
            self.status_time[i][n_k][2] = self.steps + 0.5 # 移除时刻

    def run_i2s(self, diff_i2s, i):
        times_node = dict()
        for key, value in self.status_time[i].items():
            # 必须要求当前状态为1
            if value[1] is not None and self.sirmodels[i].status[key]==1:
                if value[1] not in times_node:
                    times_node[value[1]] = set()
                times_node[value[1]].add(key)
        keys_sorted = sorted(list(times_node.keys()), reverse=True)
        # 后感染的先回退
        i2s_list = []
        for key in keys_sorted:
            if len(i2s_list) >= diff_i2s:
                break
            values = list(times_node[key])
            if len(i2s_list) + len(values) > diff_i2s:
                tmp_list = np.random.choice(values, diff_i2s-len(i2s_list), replace=False)
                i2s_list += tmp_list.tolist()
            else:
                i2s_list += list(values)
        for n_k in i2s_list:
            self.sirmodels[i].status[n_k] = 0 # r->i
            self.status_time[i][n_k][1] = None # 感染状态恢复None

    def run_r2i(self, diff_r2i, i):
        times_node = dict()
        for key, value in self.status_time[i].items():
            if value[2] is not None:
                assert self.sirmodels[i].status[key] == 2
                if value[2] not in times_node:
                    times_node[value[2]] = set()
                times_node[value[2]].add(key)
        keys_sorted = sorted(list(times_node.keys()), reverse=True)
        # 后恢复的先回退
        r2i_list = []
        for key in keys_sorted:
            if len(r2i_list) >= diff_r2i:
                break
            values = list(times_node[key])
            if len(r2i_list) + len(values) > diff_r2i:
                tmp_list = np.random.choice(values, diff_r2i-len(r2i_list), replace=False)
                r2i_list += tmp_list.tolist()
            else:
                r2i_list += list(values)
        for n_k in r2i_list:
            self.sirmodels[i].status[n_k] = 1 # r->i
            self.status_time[i][n_k][2] = None # 恢复None

    def run_s2i(self, diff_s2i, i):
        candidates = []
        for key, value in self.sirmodels[i].status.items():
            if self.sirmodels[i].status[key] == 1:
                # 寻找其邻居，所有感染节点在当前步认为是平等的
                susceptible_neighbors = [v for v in self.sirmodels[i].graph.neighbors(key) if self.sirmodels[i].status[v] == 0]
                candidates += susceptible_neighbors
        s2i_list = np.random.choice(candidates, min(diff_s2i, len(candidates)), replace=False) # 不可重复
        s2i_list = s2i_list.tolist()
        # 是否开启当不够时，任选易感进行感染, 暂时不开启
        if len(s2i_list) < diff_s2i:
            candidates2 = []
            for key, value in self.sirmodels[i].status.items():
                if value == 0 and key not in s2i_list:
                    candidates2.append(key)
            if len(candidates2) > 0:
                s2i_list2 = np.random.choice(candidates, min(diff_s2i-len(s2i_list), len(candidates)), replace=False) # 不可重复
                s2i_list += s2i_list2.tolist()
        for n_k in s2i_list:
            self.sirmodels[i].status[n_k] = 1 # s->i
            self.status_time[i][n_k][1] = self.steps + 0.5 # 前进到1

    def predict(self, Q=None, windows=5):
        """ Predict next position. """
        if Q is None:
            Q = self.Q
        if np.isscalar(Q):
            Q = eye(self.dim_x) * Q

        N = self.N
        modified_i_list = []
        modified_r_list = []
        self.steps += 1
        for i, s in enumerate(self.sigmas):
            if self.task == 'all':
                self.sirmodels[i].params['model']['beta'] = np.random.exponential(reverse_map(self.sigmas[i][-2]), self.n_nodes)
                self.sirmodels[i].params['model']['gamma'] = np.random.exponential(reverse_map(self.sigmas[i][-1]), self.n_nodes)

            tmp_i_nums = int(self.n_nodes * self.sigmas[i][0])
            tmp_r_nums = int(self.n_nodes * self.sigmas[i][1])
            counter = Counter(list(self.sirmodels[i].status.values()))
            if self.steps > 1 and self.steps % windows==0:
                if 1 and counter[2] > tmp_r_nums:
                    self.run_r2i(counter[2] - tmp_r_nums, i)
                elif 1 and counter[2] < tmp_r_nums:
                    self.run_i2r(-counter[2] + tmp_r_nums, i)
                counter = Counter(list(self.sirmodels[i].status.values()))
                if 1 and counter[1] > tmp_i_nums:
                    self.run_i2s(counter[1] - tmp_i_nums, i)
                elif 1 and counter[1] < tmp_i_nums:
                    self.run_s2i(-counter[1] + tmp_i_nums, i)
            counter = Counter(list(self.sirmodels[i].status.values()))
            modified_i = counter[1] / self.n_nodes
            modified_r = counter[2] / self.n_nodes
            modified_i_list.append(modified_i)
            modified_r_list.append(modified_r)
            # self.sirmodels[i].params['model']["beta"] = np.random.exponential(reverse_map(self.sigmas[i][-2]), self.n_nodes)
            # self.sirmodels[i].params['model']["gamma"] = np.random.exponential(reverse_map(self.sigmas[i][-1]), self.n_nodes)
        
            iterations = self.sirmodels[i].iteration_bunch(bunch_size=1)

            if iterations[0]['status'] and self.steps > 1:
                for key, value in iterations[0]['status'].items():
                    assert value >= 1
                    if value == 1:
                        assert self.sirmodels[i].status[key] == 1
                        assert self.status_time[i][key][1] is None
                        self.status_time[i][key][1] = self.steps
                    else:
                        assert self.sirmodels[i].status[key] == 2
                        assert self.status_time[i][key][2] is None and self.status_time[i][key][1] is not None
                        self.status_time[i][key][2] = self.steps
            else:
                keys = [key for key, value in iterations[0]['status'].items() if value == 1]
                for key in keys:
                    if self.sirmodels[i].status[key] == 1 and self.status_time[i][key][1] is None:
                        self.status_time[i][key][1] = self.steps
                    if self.sirmodels[i].status[key] == 2 and self.status_time[i][key][2] is None:
                        self.status_time[i][key][2] = self.steps
            trends = self.sirmodels[i].build_trends(iterations)
            self.sigmas[i][0] = trends[0]['trends']['node_count'][1][-1] / self.n_nodes
            self.sigmas[i][1] = trends[0]['trends']['node_count'][2][-1] / self.n_nodes
        e = multivariate_normal(self._mean, Q, N)
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def __repr__(self):
        # 修改实例化的输出
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