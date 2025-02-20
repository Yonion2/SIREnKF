import os

from utils import load_obj, reverse_map
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

trends_path = '../datasets/trends/trends_ws_p0.1_n5000_beta0.005_gamma0.002_fraction0.002_seed_0'
res_path = './res_1209-10/gamma/beta0.005_gamma0.1_meainfection_Qx0.001_Qp0.025_Px_0.0005_Pp0.01_Rx0.005_N50_L10.npy'

def parser(res_path):
    res_path_prior = res_path.replace('.npy', '_before.npy')
    task = res_path.split('/')[-2]
    save_name = res_path.split('/')[-1][:-4]
    save_dir_all = res_path.split('/')[1] + '_figure'
    beta = 0.005
    gamma = 0.002

    nodes_n = 5000

    gts = load_obj(trends_path)
    res = np.load(res_path)
    s_gt = gts[0]['trends']['node_count'][0]
    s_gt = [s / nodes_n for s in s_gt][1:]
    i_gt = gts[0]['trends']['node_count'][1]
    i_gt = [i / nodes_n for i in i_gt][1:]
    r_gt = gts[0]['trends']['node_count'][2]
    r_gt = [r / nodes_n for r in r_gt][1:]
    beta_gt = beta * np.ones_like(s_gt)
    gamma_gt = gamma* np.ones_like(s_gt)

    i_pre = res[:, 0]
    r_pre = res[:, 1]
    xx = [i for i in range(i_pre.shape[0])]

    if 1:
        res_before = np.load(res_path_prior)
        i_pre_before = res_before[:, 0]
        r_pre_before = res_before[:, 1]

        if task == 'all':
            beta_pre_before = [reverse_map(beta) for beta in res_before[:, -2].tolist()]
            gamma_pre_before = [reverse_map(gamma) for gamma in res_before[:, -1].tolist()]
        elif task == 'beta':
            beta_pre_before = [reverse_map(beta) for beta in res_before[:, -1].tolist()]
        else:
            assert task == 'gamma'
            gamma_pre_before = [reverse_map(gamma) for gamma in res_before[:, -1].tolist()]

        plt.plot(xx, i_pre_before, 'b--', label='infected pre before', )
        plt.plot(xx, r_pre_before, 'r--', label='removed pre before', )


    if task == 'all':
        beta_pre = [reverse_map(beta) for beta in res[:, -2].tolist()]
        gamma_pre = [reverse_map(gamma) for gamma in res[:, -1].tolist()]
    elif task == 'beta':
        beta_pre = [reverse_map(beta) for beta in res[:, -1].tolist()]
    else:
        assert task == 'gamma'
        gamma_pre = [reverse_map(gamma) for gamma in res[:, -1].tolist()]


    length = len(xx)
    plt.plot(xx, i_gt[:length], 'c-', label='infected gt', )
    plt.plot(xx, i_pre, 'k--', label='infected pre', )
    plt.plot(xx, r_gt[:length], 'y-', label='removed gt', )
    plt.plot(xx, r_pre, 'm--', label='removed pre', )

    # plt.ylim(0, 0.1)
    plt.legend()
    plt.title(save_name)
    save_dir = os.path.join(save_dir_all, 'state/{}/'.format(task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_name+'.jpg'), dpi=300, bbox_inches='tight')
    # plt.ylim(0.3, 0.32)
    plt.show()
    plt.close()

    if task != 'beta':
        plt.plot(xx, gamma_gt[:length], 'y-', label='gamma gt')
        plt.plot(xx, gamma_pre, 'm--', label='gamma pre')

        plt.legend()
        plt.title(save_name)
        if task == 'all':
            save_dir = os.path.join(save_dir_all, 'param/all/gamma/')
            # plt.ylim(0.0, 0.010)
        else:
            save_dir = os.path.join(save_dir_all, 'param/gamma/')
            # plt.ylim(0.0, 0.010)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, save_name+'.jpg'), dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.close()
        plt.ylim(0.0, 0.02)
        plt.savefig(os.path.join(save_dir, save_name+'_detail.jpg'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    if task != 'gamma':
        plt.plot(xx, beta_gt[:length], 'c-', label='beta gt')
        plt.plot(xx, beta_pre, 'm--', label='beta pre')
        plt.title(save_name)
        plt.legend()
        if task == 'all':
            save_dir = os.path.join(save_dir_all, 'param/all/beta/')
        else:
            save_dir = os.path.join(save_dir_all, 'param/beta/')
            # plt.ylim(0.0, 0.01)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # plt.ylim(0.0, 0.10)
        plt.savefig(os.path.join(save_dir, save_name+'.jpg'), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.ylim(0.0, 0.02)
        plt.savefig(os.path.join(save_dir, save_name+'_detail.jpg'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    # except:
        # print('Error')

if __name__ == '__main__':

    if 0:
        res_dir = '../../SIR_logs/res_1209-50/all/'
        for file in os.listdir(res_dir):
            if 'before' not in file:
                if 1:
                    res_path = os.path.join(res_dir, file)
                    parser(res_path)
                # except:
                #     print(res_path)
    if 1:
        parser(res_path)
    # res_path = './res_1130/beta/beta0.15_gamma0.001_Qx1e-06_Qp1e-12_Px_0.001_Pp0.01_Rx1e-06_N200.npy'