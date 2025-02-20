from utils import save_obj, load_obj
import networkx as nx


n_nodes = 5000
k = 5
p = 0.1
seed = 0
# p = 0.1
# g = None
if 1:
# while (not g) or (not g.is_directed()):
    g = nx.watts_strogatz_graph(n_nodes, k, p, seed)
    # g = nx.erdos_renyi_graph(n_nodes, p)
# save_obj(g, '1201_ws_graph_n{}_k{}_p{}_seed{}'.format(n_nodes, k, p, seed))
save_obj(g, '1201_ws_graph_n{}_k{}_p{}_seed{}'.format(n_nodes, k, p, seed))