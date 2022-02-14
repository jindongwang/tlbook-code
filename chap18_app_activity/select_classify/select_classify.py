import numpy as np
import scipy.io as io
import os
import argparse

import calc_dist
import classifier


parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='RA')
parser.add_argument('--k', type=int, default=1)
args = parser.parse_args()

root_path = '/data/jindwang/Dataset_PerCom18_STL'
datasets = ['dsads', 'pamap', 'opp_loco']
body_parts = {
    'dsads':  ['T', 'RA', 'LA', 'RL', 'LL']
}
body_map = {
    'dsads': {
        'T': np.arange(0, 81),
        'RA': np.arange(81, 162),
        'LA': np.arange(162, 243),
        'RL': np.arange(243, 324),
        'LL': np.arange(324, 405),
    }
}

def get_data_by_position(dataset, position):
    datas = data[0]
    if dataset == 'pamap':
        datas = data[1]
    elif dataset == 'opp':
        datas = data[2]
    return datas[:, body_map[dataset][position]], datas[:, -2]


def gen_train_data(d_src, d_tar, k=1):
    x_tar, y_tar = d_tar
    x_src, y_src = d_src[0]
    y_src = np.reshape(y_src, (len(y_src), 1))
    for i in range(k-1):
        x, y = d_src_list[i]
        x_src = np.vstack((x_src, x))
        y_src = np.vstack((y_src, y.reshape((len(y), 1))))
    y_src = y_src.squeeze()
    return x_src, y_src, x_tar, y_tar


def source_selection(target_pos='RA'):
    # weights given by human, for the semantic similarity
    weights = [.2, .5, .15, .15] if args.target == 'RA' else [.5, .2, .15, .15]
    d_tar = get_data_by_position('dsads', target_pos)
    x_tar = d_tar[0]
    t = body_parts['dsads'].copy()
    t.remove(target_pos)
    d_src_list = [get_data_by_position('dsads', item) for item in t]
    print('Source candidates:', [item for item in t])
    a_dist = [calc_dist.proxy_a_distance(
        x_tar, item[0]) for item in d_src_list]
    cos_dist = [1 - (calc_dist.cosine_sim(x_tar, d_src_list[i]
                                          [0]).mean() * weights[i]) for i in range(len(d_src_list))]
    total_dist = np.array(a_dist) + np.array(cos_dist)
    print(f'Distance to target: ', total_dist)
    return total_dist, d_src_list, d_tar, t


def classify(alldata):
    x_src, y_src, x_tar, y_tar = alldata
    cls = classifier.Classifier((x_src, y_src - 1, x_tar, y_tar - 1))
    cls.fit_predict_all()


if __name__ == '__main__':
    SEED = 100
    np.random.seed(SEED)
    data = [io.loadmat(os.path.join(root_path, datasets[i]))[
        f'data_{datasets[i]}'] for i in range(len(datasets))]
    print(f'Target position: {args.target}')
    # Source selection to get distance between each source-target pair
    total_dist, d_src_list, d_tar, source_pos = source_selection(args.target)
    # # Sort distances to get the best k source domains
    ind = np.argsort(total_dist)
    dsrc = [d_src_list[ind[i]] for i in range(len(source_pos))]
    sort_source = [
          source_pos[ind[i]] for i in range(len(source_pos))]
    print('Source sorted positions: ', sort_source)
    print(f'The best source to target is: {sort_source[0]}')
    # Classify
    x_src, y_src, x_tar, y_tar = gen_train_data(dsrc, d_tar, args.k)

    print('Classification using KNN...')
    classify((x_src, y_src, x_tar, y_tar))
    
    print('Classification using TCA+KNN...')
    from tca import TCA
    t = TCA(kernel_type='primal', dim=40, lamb=.1, gamma=1)
    acc_tca, _ = t.fit_predict(x_src, y_src, x_tar, y_tar)
    print(f'Acc of TCA: {acc_tca}')
