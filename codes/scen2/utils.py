import pickle
import numpy as np
import math

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# def hx(states):
#     return np.array(states[:3])

def hx2(states):
    return np.array(states[:2])  # 返回更新值

def hx(states):
    return np.array([states[0]])

def hx_sis(states):
    return np.array([states[0]])

def compute_dist(dict1, key1, key2):
    dis = 0
    # return abs(dict1[1]-key1) + abs(dict1[2]-key2)
    return abs(dict1[1]-key1)
# def map(x):
#     return math.tan((x-1/2)*math.pi)
#
# def reverse_map(y):
#     return math.atan(y) / math.pi + 1/2

def map(x):
    # return math.tan((x-1/2)*math.pi) / 300
    return min(0.02,x)*50

def reverse_map(y):
    return max(0,y)/50

def map_regular(x):
    return math.tan((x-1/2)*math.pi)*20

def reverse_map_regular(y):
    return math.atan(y/20) / math.pi + 1/2

if __name__ == '__main__':
    print(map(0.012))
    print(map(0.004))
