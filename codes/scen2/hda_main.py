from EnKF_both import EnKFBoth
from EnKF_delta import EnKFBeta
from EnKF_delta_gamma import EnKFGamma
from utils import save_obj, load_obj, hx2, reverse_map, map, hx
from tqdm.notebook import tqdm as tqdm
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
# from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
# from generate_data import gene_obserbation
from tqdm import tqdm
# from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
# from generate_data import gene_obserbation
from numpy.random import multivariate_normal
import numpy as np
import networkx as nx
from hSIR import HyperSIRModel

def generate_modes(N, graph, seed, Is, params, mode, beta_gt, gamma_gt):
    # 生成预测模型
    # params 里面有初始值
    # 这里取消了随机种子的设定
    n_nodes = nx.number_of_nodes(graph)
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
        model = HyperSIRModel(graph)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', np.random.exponential(betas[i], int(n_nodes)))
        cfg.add_model_parameter('gamma', np.random.exponential(gammas[i], int(n_nodes)))
        cfg.add_model_parameter("fraction_infected", Is[i]) # 用模拟值做初始感染率
        model.set_initial_status(cfg)
        sirmodels.append(model)
    return sirmodels


def get_initial_state(models):
    # 对每个样本计算当前的感染率和转化率
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
        Is.append(i/cnt)  # 平均感染率
        Rs.append(r/cnt)  # 平均回复率
    return Is, Rs

def run(gt_param, beta, gamma, task, Q_x, Q_param, P_x, P_param, R_x, N, windows, rounds, measurement_mode='infection', name = None):
    # 把生成观测和预测放在一起
    assert measurement_mode in ['infection', 'both'] # [仅观测感染率，观测感染率和移除率]
    gt_graph = gt_param['graph']
    gt_beta = gt_param['beta_gt']
    gt_gamma = gt_param['gamma_gt']
    gts = gt_param['gts']
    n_nodes = gt_param['n_nodes']
    if task in ['beta', 'gamma']:
        if task == 'beta':
            x_mean = np.array([10/n_nodes, 0/n_nodes, beta]) # infected, removed, beta
        else:
            x_mean = np.array([10/n_nodes, 0/n_nodes, gamma]) # 这里的beta不是beta_gt 是随机的
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
        params = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 最后一个参数是beta 对它做处理
    elif task == 'gamma':
        params = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 只同化一个参数
    else:
        params1 = np.clip(states_init[:, -2].reshape(-1), 0.0001, 0.5) # 同时同化两个参数
        params2 = np.clip(states_init[:, -1].reshape(-1), 0.0001, 0.5) # 同时同化两个参数
        params = [params1, params2]
    # 模型要求初始移除率为0
    sirmodels = generate_modes(N, gt_graph, 0, Is, params, task, gt_beta, gt_gamma)

    dim_z = R.shape[0]  # 需要估计的参数数量
    current_Is, current_Rs = get_initial_state(sirmodels)  # 每个样本的平均感染率
    if task in ['beta', 'gamma']:
        x = np.array([current_Is, current_Rs, [map(l) for l in states_init[:, -1].reshape(-1).tolist()]]).T
        if task == 'beta':
            enkf = EnKFBeta(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task) # x代表初始状态  hx返回numpy形式的第一个值
        else:
            enkf = EnKFGamma(x, P, dim_z, N, hx, sirmodels, n_nodes, task=task) # hx是函数，返回更新后的观测值
    else:
        current_betas = [map(l) for l in states_init[:, -2].reshape(-1).tolist()]  # 把值映到R上
        current_gammas = [map(l) for l in states_init[:, -1].reshape(-1).tolist()]
        x = np.array([current_Is, current_Rs, current_betas, current_gammas]).T # x的每个元素的长度是样本量
        if measurement_mode == 'both':
            enkf = EnKFBoth(x, P, dim_z, N, hx2, sirmodels, n_nodes, task=task)
        else:
            raise NotImplementedError
    post_states = list()
    prior_states = list()
    save_path = 'beta{}_gamma{}_mea{}_Qx{}_Qp{}_Px_{}_Pp{}_Rx{}_N{}_L{}'.format(beta, gamma, measurement_mode, Q_x, Q_param, P_x, P_param, R_x, N, windows) +name
    save_dir = gt_param['save_dir']+task
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
            else:
                if i % 500 == 0:
                    Q[-1][-1] /= (5.0**2)
                    R[-1] /= (5.0**2)
                enkf.predict(Q, windows=windows)
            enkf.update(measurement, R)
            post_states.append(enkf.x)
            prior_states.append(enkf.x_prior)
            # if i % 50 == 0:
            #     np.save(full_path, post_states) # 保存EnKF修正后结果  #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
            #     np.save(full_path+'_before', prior_states) # 保存预测结果
            if task == 'beta':
                tbar.set_postfix(param = reverse_map(post_states[-1][-1]))
                # print('param: ', reverse_map(post_states[-1][-1]))
            elif task == 'gamma':
                tbar.set_postfix(gama_param = reverse_map(post_states[-1][-1]))
                # print('gamma param: ', reverse_map(post_states[-1][-1]))
            else:
                tbar.set_postfix(param = reverse_map(post_states[-1][-2]), gama_param = reverse_map(post_states[-1][-1]))
                # print('beta estimate={}, gamma estimate={}'.format(reverse_map(post_states[-1][-2]), reverse_map(post_states[-1][-1])))
    np.save(full_path, post_states)
    np.save(full_path+'_before', prior_states)
    # 要保存每次估计的参数， 已经保存过了
    ## 应该加上生成图片的模块
    
if __name__ == '__main__':
    m = 6
    path = "./datasets/graphs/p2p-Gnutella05.txt"
    def read_txt_direct(data):
        g = nx.read_edgelist(data,  nodetype=int, create_using=nx.DiGraph())
        return g
    g = read_txt_direct(path)
    # trends = load_obj(r"C:\Users\xinji\Documents\理论论文\卡尔曼滤波\paper_code\scen2\scen2_data\trends")
    # gt_param = {'graph': g, 'Is':0.002, 'beta_gt':0.006, 'gamma_gt':0.003, 'gts': trends, 'n_nodes':8846, 'save_dir':'./scen2_data/res/'}
    # run(gt_param = gt_param, beta=0.01, gamma=0.01, task='all', Q_x = 1e-4, Q_param = 1e-4, P_x = 5e-4, P_param = 1e-2, R_x= 5e-3, N = 50, windows = 10, rounds = 3000, measurement_mode='both',name = '5')
    beta = 0.006
    gamma = 0.003
    round = 3000
    Is = 0.002
    n_nodes = nx.number_of_nodes(g)
    model = HyperSIRModel(g)
    save_obj(g, './scen2/scen2_data/netwwork/er_8846')
    beta_list = np.random.exponential(beta, n_nodes)
    gamma_list = np.random.exponential(gamma, n_nodes)
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta_list)
    cfg.add_model_parameter('gamma', gamma_list)
    cfg.add_model_parameter("fraction_infected", Is) # 用模拟值做初始感染率
    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(bunch_size=3000)
    trends = model.build_trends(iterations)
    save_obj(trends, './datasets/scen2_data/beta_{}gamma_{}er8846trends'.format(beta, gamma))
    gt_param = {'graph': g, 'Is':0.002, 'beta_gt':beta, 'gamma_gt':gamma, 'gts': trends, 'n_nodes':8846, 'save_dir':'./datasets/scen2_data/'}
    run(gt_param = gt_param, beta=0.01, gamma=0.01, task='all', Q_x = 1e-4, Q_param = 1e-4, P_x = 5e-4, P_param = 1e-2, R_x= 5e-3, N = 50, windows = 10, rounds = 3000, measurement_mode='both', name = 'er8846')