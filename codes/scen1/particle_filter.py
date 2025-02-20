from collections import Counter
import numpy as np
from utils import map, reverse_map
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import math
from tqdm import tqdm 
import pickle
import networkx as nx
def map(x):
    return math.tan((x-1/2)*math.pi) / 300

def reverse_map(y):
    return math.atan(300*y) / math.pi + 1/2


def generate_particle_group(particle_size, fraction_infected, g):
    # 一个用来保存模型，一个用来保存状态
    particle_group = []
    state_group = []
    beta = np.random.uniform(0, 0.015, particle_size)
    gamma = np.random.uniform(0, 0.02, particle_size)
    print(min(beta))
    print(min(gamma))
    for i in range(particle_size):
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta[i])
        cfg.add_model_parameter('gamma',gamma[i])
        cfg.add_model_parameter("fraction_infected", fraction_infected) # 用模拟值做初始感染率
        model = ep.SIRModel(g)
        model.set_initial_status(cfg)
        particle_group.append(model)
        count = Counter(model.status.values())
        # print(count[1]/8846)
        state_group.append((model.status, count[1]/8846, count[2]/8846, map(beta[i]), map(gamma[i])))
        
    return particle_group, np.array(state_group)


def update_particle_group(particle_group, state_group):
    for i in range(len(particle_group)):
        state = state_group[i]
        particle_group[i].status = state[0]
        particle_group[i].params['model']['beta'] = reverse_map(state[-2])
        particle_group[i].params['model']['gamma'] = reverse_map(state[-1])
    return particle_group, state_group

# 预测步向前走一步，更新状态
def predict(model):
    iterations = model.iteration_bunch(bunch_size=1)
    trends = model.build_trends(iterations)
    return model, model.status, trends


    

def cal_prob(obe, R, state):
    # 二维高斯分布, obe是均值
    prob = np.exp(-0.5 * ( ((obe[0] - state[0]) ** 2 + (obe[1] - state[1])**2 )/ R) )
    if prob < 1e-60:
        prob = 1e-60
    if prob > 1:
        prob = 1
    return prob


def resample(state_group, weights):
    index = np.random.choice(list(np.arange(len(state_group))), size=len(state_group), replace=True, p=weights)
    return state_group[index,:]

def single_particle(particle, state, observe, R_cov, Q_beta, Q_gamma):
    _, new_state, trends = predict(particle)
    beta = state[-2]
    gamma = state[-1]
    # 对参数做扰动
    ave_i = trends[-1]['trends']['node_count'][1][-1]/8846
    ave_r = trends[-1]['trends']['node_count'][2][-1]/8846
    new_beta = np.random.normal(beta, Q_beta, 1)
    new_gamma = np.random.normal(gamma, Q_gamma, 1)
    weight = cal_prob(observe, R_cov, (ave_i, ave_r))
    return (new_state, ave_i, ave_r, new_beta, new_gamma), weight
    

def particle_filter(observations, Q_beta, Q_gamma, R_cov, particle_size, fraction_infected, g):
    # 初始化粒子
    estimated = []
    particle_group, state_group = generate_particle_group(particle_size, fraction_infected, g)
    # 初始化权重
    weights = np.ones(particle_size) / particle_size
    with tqdm(range(len(observations)), desc='Test') as tbar:
    # with tqdm(range(10), desc='Test') as tbar:
        for t in tbar:
        # 预测步骤
            if t % 600 == 0:
                Q_beta *= 0.4
                Q_gamma *= 0.4
            estimated_i = []
            estimated_r = []
            estimated_beta = []
            esitmated_gamma = []
            for i in range(particle_size):
                # new_state是当前状态, trends是做的统计
                _, new_state, trends = predict(particle_group[i])
                beta = state_group[i][-2]
                gamma = state_group[i][-1]
                # 对参数做扰动
                ave_i = trends[-1]['trends']['node_count'][1][-1]/8846
                ave_r = trends[-1]['trends']['node_count'][2][-1]/8846
                # print( trends[-1]['trends']['node_count'][1])
                new_beta = np.random.normal(beta, Q_beta, 1)
                new_gamma = np.random.normal(gamma, Q_gamma, 1)
                state_group[i] = (new_state, ave_i, ave_r, new_beta, new_gamma)
                # 不做扰动
                # state_group[i] = (new_state, ave_i, ave_r, beta, gamma)
                actual_state = np.array([ave_i, ave_r])
                # 权重更新
                weights[i] = cal_prob(observations[t], R_cov, actual_state)
            weights/= sum(weights)
            # 重采样
            state_group = resample(state_group, weights)
            # 更新模型
            particle_group, state_group = update_particle_group(particle_group, state_group)
            for i in range(particle_size):
                estimated_i.append(state_group[i][1])
                estimated_r.append(state_group[i][2])
                estimated_beta.append(reverse_map(state_group[i][-2]))
                esitmated_gamma.append(reverse_map(state_group[i][-1]))
                
            tbar.set_postfix(param =  (np.mean(estimated_i), np.mean(estimated_r), np.mean(estimated_beta), np.mean(esitmated_gamma)))
            estimated.append((np.mean(estimated_i), np.mean(estimated_r), np.mean(estimated_beta), np.mean(esitmated_gamma)))
            # 重采样步骤
    return np.array(estimated)
  
if __name__ == '__main__':
    Q_beta = Q_gamma = 0.03
    R_cov = 0.0000001
    particle_size = 100
    path = r"C:\Users\xinji\Documents\理论论文\卡尔曼滤波\paper_code\scen1\paper_data\network\p2p-Gnutella05.txt"
    def read_txt_direct(data):
        g = nx.read_edgelist(data,  nodetype=int, create_using=nx.DiGraph())
        return g
    g = read_txt_direct(path)
    trends = pickle.load(open(r'C:\Users\xinji\Documents\理论论文\卡尔曼滤波\paper_code\scen1\paper_data\case1\obe\trends_addedd_50beta0.005_gamma0.002_fraction0.002.pkl', 'rb'))
    observations = np.vstack((trends[0]['trends']['node_count'][1], trends[0]['trends']['node_count'][2])).T/8846
    e = particle_filter(observations, Q_beta, Q_gamma, R_cov, particle_size, 0.002, g)
    np.save(r"C:\Users\xinji\Documents\理论论文\卡尔曼滤波\paper_code\scen1\paper_data\case1\obe\enutalla0.0050.002Q_beta{}Q_gamma{}size{}Rcov{}.npy".format(Q_beta, Q_gamma, particle_size, R_cov), e)