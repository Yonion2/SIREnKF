# toymodel的网络参数和真实网络相同
from EnKF_delta import EnKFBeta
from EnKF_delta_gamma import EnKFGamma
# from EnKF_delta_both import EnsembleKalmanFilter
from EnKF_both import EnKFBoth
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import load_obj, hx, save_obj, map, reverse_map, hx2
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
from generate_data import gene_obserbation
from numpy.random import multivariate_normal

graph_params = {'type':'er','p': 0.1, 'k': 5,  'n_nodes': 5000}

def generate_graph(graph_params):
    if graph_params['type'] == 'er':
        # p随机图
        p = graph_params['p']
        n_nodes = graph_params['n_nodes']
        g = nx.erdos_renyi_graph(n_nodes, p)
        # self.g = load_obj(g_path)
    elif graph_params['type'] == 'ws':
        k = graph_params['k']
        g = nx.watts_strogatz_graph(n_nodes, k, p)
    return g
    


def generate_modes_hyper(N, graph_params, Is, params, mode, beta_gt, gamma_gt):
    # 生成预测模型
    # params 里面有初始值
    sirmodels = []
    if mode == 'beta':
        betas = params #生成的模型的均值是真实值
        gammas = gamma_gt * np.ones(N)  # _gt代表真实值
    elif mode == 'gamma':  # 估计gamma
        betas = beta_gt * np.ones(N)
        gammas = params
    else:
        betas = params[0].reshape(-1)
        gammas = params[1].reshape(-1)
    for i in range(N):  # N是节点数目
        graph = generate_graph(graph_params)
        model = ep.SIRModel(graph)
        # model = ep.SIRModel(graph, seed=seed)
        # model = ep.SIRModel(graph)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', betas[i])
        cfg.add_model_parameter('gamma', gammas[i])
        cfg.add_model_parameter("fraction_infected", Is[i]) # 用模拟值做初始感染率
        model.set_initial_status(cfg)
        sirmodels.append(model)
    return sirmodels

