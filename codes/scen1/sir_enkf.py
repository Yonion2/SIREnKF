# from EnKF_random import EnsembleKalmanFilter
# from EnKF_multirun import EnsembleKalmanFilter
from EnKF_delta import EnsembleKalmanFilterBeta
from EnKF_delta_gamma import EnsembleKalmanFilterGamma
# from EnKF_delta_both import EnsembleKalmanFilter
from EnKF_delta_both import EnsembleKalmanFilterBoth
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_obj, hx, save_obj, map, reverse_map, hx2
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
from numpy.random import multivariate_normal


def generate_modes(N, graph, seed, Is, params, mode, beta_gt, gamma_gt):
    # 生成预测模型
    # params 里面有初始值
    # 这里取消了随机种子的设定
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


def run(gt_param, beta, gamma, task, Q_x, Q_param, P_x, P_param, R_x, N, windows, rounds, measurement_mode='infection'):
    gt_graph = gt_param['graph']
    gt_beta = gt_param['beta_gt']
    gt_gamma = gt_param['gamma_gt']
    gts = gt_param['gts']
    n_nodes = gt_param['n_nodes']
    Is = gt_param['Is']
    save_path = 'beta{}_gamma{}_mea{}_Qx{}_Qp{}_Px_{}_Pp{}_Rx{}_N{}_L{}'.format(
                        beta, gamma, measurement_mode, Q_x, Q_param, P_x, P_param, R_x, N, windows)

    if task in ['beta', 'gamma']:
        if task == 'beta':
            x_mean = np.array([10/n_nodes, 0/n_nodes, beta]) # infected, removed, beta
        else:
            x_mean = np.array([10/n_nodes, 0/n_nodes, gamma])
        Q = np.diag([Q_x**2, Q_x**2, Q_param**2])
        P = np.diag([P_x**2, P_x**2, P_param**2])
    else:
        x_mean = np.array([10/n_nodes, 0/n_nodes, beta, gamma])
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
    sirmodels = generate_modes(N, gt_graph, 0, Is, params, task, gt_beta, gt_gamma)

    dim_z = R.shape[0]
    current_Is, current_Rs = get_initial_state(sirmodels)
    if task in ['beta', 'gamma']:
        x = np.array([current_Is, current_Rs, [map(l) for l in states_init[:, -1].reshape(-1).tolist()]]).T
        if task == 'beta':
            enkf = EnsembleKalmanFilterBeta(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task) # x代表初始状态  hx返回numpy形式的第一个值
        else:
            enkf = EnsembleKalmanFilterGamma(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task)
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
            # print(i)
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

def main():
    # 观测平均感染率，估计beta，超参数重要性倒序: windows、N、Q_x、Q_param、P_x、P_param、R_x
    run(beta=0.1, gamma=0.002, task='beta', Q_x=1e-3, Q_param=2.5e-2, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50, windows=1, measurement_mode='infection')  # N是样本个数
    # 观测平均感染率，估计gamma，一般window=1即可，增大可能更好，但是没试过
    run(beta=0.005, gamma=0.1, task='gamma', Q_x=1e-3, Q_param=2.5e-2, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50, windows=10, measurement_mode='infection')
    # # 观测平均感染率和平均移除率，估计beta和gamma, windows设置为10～50都可
    run(beta=0.01, gamma=0.01, task='all', Q_x=1e-4, Q_param=2.5e-3, P_x=5e-4, P_param=1e-2, R_x=1e-3, N=50, windows=10, measurement_mode='both')
    # for beta in [0.05, 0.008, 0.01, 0.012, 0.015]:
    #     for gamma in [0.001, 0.003, 0.005, 0.008, 0.01]:
    #         idx += 1
    #         print('test: ', idx)
    #         run(beta=beta, gamma=gamma, Q_x=2e-4, Q_param=1, P_x=5e-4, P_param=50, R_x=1e-4, N=64)
    # for qx in [2e-4, 5e-4, 1e-3]:
    #     for P_x in [2e-4, 5e-4, 1e-3]:
    #         idx += 1
    #         print('test: ', idx)
    #         run(beta=0.010, gamma=0.005, Q_x=qx, Q_param=1, P_x=P_x, P_param=50, R_x=1e-4, N=64)
    # for Q_p in [0.1, 1, 10]:
    #     for P_p in [0.1, 1, 10]:
    #         idx += 1
    #         print('test: ', idx)
    #         run(beta=0.010, gamma=0.005, Q_x=2e-4, Q_param=Q_p, P_x=5e-4, P_param=P_p, R_x=1e-4, N=64)
    # for N in [50, 64, 100, 128, 256]:
    #     idx += 1
    #     print('test: ', idx)
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-4, P_x=5e-3, P_param=1e-2, R_x=5e-3, N=50) # beta, good!! gamma相对差一些
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-4, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50) # beta, good!! gamma相对差一些
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-4, P_x=5e-3, P_param=1e-2, R_x=5e-3, N=50)
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-4, P_x=5e-3, P_param=1e-2, R_x=5e-4, N=50)
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-4, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50) # 前面挺好，后边飞了。
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-3, Q_param=2.5e-3, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50) # 前面挺好，后边飞了。
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-3, P_x=5e-4, P_param=1e-2, R_x=5e-3, N=50) # 前面挺好，后边飞了。
    # run(beta=0.1, gamma=0.1, task='all', Q_x=1e-4, Q_param=2.5e-3, P_x=5e-3, P_param=1e-2, R_x=5e-3, N=50) # 前面挺好，后边飞了。
    # for beta in [0.01, 0.05, 0.10]:
    #     for Q_x in [1e-4, 1e-6, 1e-8, 1e-12]:
    #         for Q_param in [8e-5, 1e-12, 1e-6, 1e-3]:
    #             for P_param in [4e-4, 1e-2, 1e-4, 1]:
    #                 for P_x in [1e-3, 1e-1]:
    #                     for R_x in [5e-8, 1e-6, 5e-4, 1e-3]:
    #                         for N in [50, 100, 200]:
    #                             run(beta=beta, gamma=0.001, task='beta', Q_x=Q_x, Q_param=Q_param, P_x=P_x, P_param=P_param, R_x=R_x, N=N)



if __name__ == '__main__':
    main()