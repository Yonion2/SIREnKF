# toymodel的网络参数和真实网络相同
from EnKF_delta import EnKFBeta
from EnKF_delta_gamma import EnKFGamma
from EnKF_delta_both import EnsembleKalmanFilterBoth
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_obj, hx, save_obj, map, reverse_map, hx2
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import os
from numpy.random import multivariate_normal

graph_para = {'type':'er','p_e': 0.1, 'p_w':0.1, 'd':5, 'm': 5,  'n_nodes': 5000}



def generate_modes(N, graph_para, seed, Is, params, task, gt_beta, gt_gamma):
    # 生成预测模型
    # params 里面有初始值
    # 这里取消了随机种子的设定
    g_tpye = graph_para['type']
    p_e = graph_para['p_e']
    p_w = graph_para['p_w']
    d = graph_para['d']
    m = graph_para['m']
    n_nodes = graph_para['n_nodes']
    sirmodels = []
    betas = params[0].reshape(-1)
    gammas = params[1].reshape(-1)
    for i in range(N):  # N是节点数目
        if g_tpye == 'er':
            graph = nx.erdos_renyi_graph(n_nodes, p_e, seed=seed)
        elif g_tpye == 'ba':
            graph = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        elif g_tpye == 'ws':
            graph = nx.random_graphs.watts_strogatz_graph(n_nodes, k=d, p=p_w, seed=seed)
        else:
            raise ValueError('Invalid graph type')
        model = ep.SIRModel(graph)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', betas[i])
        cfg.add_model_parameter('gamma', gammas[i])
        cfg.add_model_parameter("fraction_infected", Is[i]) # 用模拟值做初始感染率
        model.set_initial_status(cfg)
        sirmodels.append(model)
    return sirmodels


def get_initial_state(models):
    Is = []
    Rs = []
    for model in models:
        i = 0
        r = 0
        cnt = 0
        for key, val in model.initial_status.items():
            if val == 1:  #如果是在感染状态
                i += 1
            elif val == 2:  #如果是在恢复状态
                r += 1
            cnt += 1
        Is.append(i/cnt)
        Rs.append(r/cnt)
    return Is, Rs
# trend 的结构是一个字典，下面有”trends“， node_count,是0，1，2三个状态每个迭代轮次的数量变化


def run(gt_param, beta, gamma, task, Q_x, Q_param, P_x, P_param, R_x, N, windows, rounds, measurement_mode='infection', name = None):
    gt_beta = gt_param['beta_gt']
    gt_gamma = gt_param['gamma_gt']
    gts = gt_param['gts']
    n_nodes = gt_param['n_nodes']
    Is = gt_param['Is']
    graph_para = gt_param['graph_para']
    save_path = 'beta{}_gamma{}_mea{}_Qx{}_Qp{}_Px_{}_Pp{}_Rx{}_N{}_L{}'.format(
                        beta, gamma, measurement_mode, Q_x, Q_param, P_x, P_param, R_x, N, windows)+ name

    if task in ['beta', 'gamma']:
        if task == 'beta':
            x_mean = np.array([10/n_nodes, 0/n_nodes, beta]) # infected, removed, beta
        else:
            x_mean = np.array([10/n_nodes, 0/n_nodes, gamma])
        Q = np.diag([Q_x**2, Q_x**2, Q_param**2])
        P = np.diag([P_x**2, P_x**2, P_param**2])
    else:
        x_mean = np.array([Is, 0/n_nodes, beta, gamma])
        Q = np.diag([Q_x**2, Q_x**2, Q_param**2, Q_param**2])   #Q是系统误差协方差
        P = np.diag([P_x**2, P_x**2, P_param**2, P_param**2])   # P是中间变量 是进入系统进化后的误差
    if measurement_mode == 'both':
        R = np.diag([R_x**2, R_x**2])  # R是观测误差协方差
    else:
        R = np.diag([R_x**2])
    states_init = multivariate_normal(mean=x_mean, cov=P, size=N)  #size为样本大小  shape(N,3)
    Is = np.clip(states_init[:, 0].reshape(-1), 0.001, 0.003)  # 限制下初始感染率  reshape(-1)改成一串 没有行列 限制在0.001 到0.003之间
    if task == 'beta':
        params = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 只同化一个参数
    elif task == 'gamma':
        params = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 只同化一个参数
    else:
        params1 = np.clip(states_init[:, -2].reshape(-1), 0.0001, 0.5) # 同时同化两个参数
        params2 = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 同时同化两个参数
        params = [params1, params2]
    # 模型要求初始移除率为0
    sirmodels = generate_modes(N, graph_para, 0, Is, params, task, gt_beta, gt_gamma)

    dim_z = R.shape[0]
    current_Is, current_Rs = get_initial_state(sirmodels)
    if task in ['beta', 'gamma']:
        x = np.array([current_Is, current_Rs, [map(l) for l in states_init[:, -1].reshape(-1).tolist()]]).T
        if task == 'beta':
            enkf = EnKFBeta(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task) # x代表初始状态  hx返回numpy形式的第一个值
        else:
            enkf = EnKFGamma(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task)
    else:
        current_betas = [map(l) for l in states_init[:, -2].reshape(-1).tolist()]
        current_gammas = [map(l) for l in states_init[:, -1].reshape(-1).tolist()]
        x = np.array([current_Is, current_Rs, current_betas, current_gammas]).T
        if measurement_mode == 'both':
            enkf = EnsembleKalmanFilterBoth(x, P, dim_z, N, hx2, sirmodels, n_nodes, task=task)
        else:
            raise NotImplementedError

    post_states = list()
    prior_states = list()
    save_dir = './res_1209-{}/{}/'.format(windows, task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    full_path = os.path.join(save_dir, save_path)
    print('save to:---> ', full_path)
    with tqdm(range(rounds), desc='Test') as tbar:
        for i in tbar:
            if measurement_mode == 'both':
                measurement = np.array([gts[0]['trends']['node_count'][j][i] / n_nodes for j in range(1, 3)])
            else:
                measurement = np.array([gts[0]['trends']['node_count'][1][i] / n_nodes]) # 感染率
            if task == 'all':
                # 如果同时估计两个参数，则开启自适应噪声
                Q[0][0] = max(min((measurement[0] * 3e-2)**2, 3e-3**2), 5e-6**2)
                Q[1][1] = max(min((measurement[-1] * 3e-2)**2, 3e-3**2), 5e-6**2)
                R[0][0] = max(min((measurement[0] * 4e-2)**2, 6e-3**2), 5e-6**2)
                enkf.predict(Q, windows=windows)
            elif task == 'beta' or task == 'gamma':
                if i % 250 == 0:
                    Q[-1][-1] /= (5.0**2)
                    R[-1] /= (5.0**2)
                enkf.predict(Q, windows=windows)
            else:
                if i % 500 == 0:
                    Q[-1][-1] /= (5.0**2)
                    Q[-2][-2] /= (5.0**2)
                    R[-1][-1] /= (5.0**2)
                    R[-2][-2] /= (5.0**2)
                enkf.predict(Q, windows=windows)
            enkf.update(measurement, R)
            post_states.append(enkf.x)
            prior_states.append(enkf.x_prior)
            # if i % 50 == 0:
            #     np.save(full_path, post_states) # 保存EnKF修正后结果  #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
            #     np.save(full_path+'_before', prior_states) # 保存预测结果
            if task == 'beta':
                tbar.set_postfix(param =  reverse_map(post_states[-1][-1]))
            elif task == 'gamma':
                tbar.set_postfix(param =  reverse_map(post_states[-1][-1]))
            else:
                tbar.set_postfix(param =  (reverse_map(post_states[-1][-2]), reverse_map(post_states[-1][-1])))
    np.save(full_path, post_states)
    np.save(full_path+'_before', prior_states)

if __name__ == '__main__':
    beta_t = 0.002
    gamma_t = 0.001
    Is = 0.002
    for type in ['er','ba','ws']:
        if type == 'er':
            for p_e in [0.002, 0.02, 0.04, 0.06]:
                graph_para = {'type':'er','p_e': p_e, 'p_w':0.1, 'd':5, 'm': 5,  'n_nodes': 5000}
                g = nx.random_graphs.erdos_renyi_graph(5000, p_e) # 生成网络
                n_nodes = 5000
                model = ep.SIRModel(g)
                cfg = mc.Configuration()
                cfg.add_model_parameter('beta', beta_t)
                cfg.add_model_parameter('gamma', gamma_t)
                cfg.add_model_parameter("fraction_infected", Is) # 用模拟值做初始感染率
                model.set_initial_status(cfg)
                iterations = model.iteration_bunch(bunch_size=3000)
                trends = model.build_trends(iterations)
                save_obj(trends, './datasets/scen3_data/beta_{}gamma_{}p_{}type{}'.format(beta_t, gamma_t,p_e,type))
                gt_param = {'graph_para': graph_para, 'Is':0.002, 'beta_gt':beta_t, 'gamma_gt':gamma_t, 'gts': trends, 'n_nodes':5000, 'save_dir':'./datasets/scen3_data/res_1209-10'}
                run(gt_param, beta = 0.01, gamma=0.01, task = "all", Q_x = 1e-4, Q_param = 1e-4, P_x = 5e-4, P_param = 1e-2, R_x= 5e-3, N = 50, windows = 10, rounds =3000, measurement_mode='both', name="{}{}{}{}".format(type, p_e,beta_t, gamma_t))