import os

import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

import numpy as np

from utils import save_obj, load_obj

class GenerateSIRData():
    def __init__(self, graph_type, graph_params, model_params, seed):
        self.graph_type = graph_type
        self.graph_params = graph_params
        self.beta  = model_params['beta']
        self.gamma = model_params['gamma']
        self.fraction_infected = model_params['fraction_infected']
        self.seed = seed

    def generate_graph(self):
        if self.graph_type == 'er':
            p = self.graph_params['p']
            n_nodes = self.graph_params['n_nodes']
            self.g = nx.erdos_renyi_graph(n_nodes, p, seed=self.seed)
        elif self.graph_type == 'ws':
            g_path = self.graph_params['g_path']
            self.g = load_obj(g_path)
        else:
            raise NotImplemented
    def generate_model(self):
        self.model = ep.SIRModel(self.g, seed=self.seed)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', self.beta)
        cfg.add_model_parameter('gamma', self.gamma)
        cfg.add_model_parameter("fraction_infected", self.fraction_infected)
        self.model.set_initial_status(cfg)

    def print_info(self):
        print('info: ', self.model.get_info())

    def generate_data(self, bunch_size):
        self.generate_graph()
        self.generate_model()
        self.print_info()
        self.iterations = self.model.iteration_bunch(bunch_size)
        self.trends = self.model.build_trends(self.iterations)

    def save_data(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = '{}_p{}_n{}_beta{}_gamma{}_fraction{}_seed_{}'.format(
                                self.graph_type, self.graph_params['p'],self.graph_params['n_nodes'],
                                            self.beta, self.gamma, self.fraction_infected, self.seed
                                             )
        save_obj(self.trends, os.path.join(save_dir, 'trends_'+save_name))
        save_obj(self.iterations, os.path.join(save_dir, 'iterations_'+save_name))
        # save_obj(self.g, os.path.join(save_dir, 'graphs_'+save_name))


if __name__ == '__main__':
    # graph_type = 'er'
    # graph_params = {'p': 0.1, 'n_nodes': 1000}
    # model_params = {'beta': 0.01, 'gamma': 0.005, 'fraction_infected': 0.05}
    # seed = 0
    seed = 0
    graph_type = 'ws'
    # graph_params = {'p': 0.01, 'k': 3,  'n_nodes': 1000, 'g_path': './1126_ws_graph_n1000_k3_p0.01'}
    graph_params = {'p': 0.1, 'k': 5,  'n_nodes': 5000, 'g_path': './1201_ws_graph_n5000_k5_p0.1_seed0'}
    # model_params = {'beta': 0.04, 'gamma': 0.001, 'fraction_infected': 0.05}
    model_params = {'beta': 0.005, 'gamma': 0.002, 'fraction_infected': 0.002}
    sir_model = GenerateSIRData(graph_type, graph_params, model_params, seed)
    sir_model.generate_data(3000)
    save_dir = '../sir_ws/1201/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sir_model.save_data(save_dir)






