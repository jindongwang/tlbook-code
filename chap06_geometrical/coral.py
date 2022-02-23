# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import os
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors


def load_data(folder, domain):
    from scipy import io
    data = io.loadmat(os.path.join(folder, domain + '_fc6.mat'))
    return data['fts'], data['labels']


def load_csv(folder, src_domain, tar_domain):
    data_s = np.loadtxt(f'{folder}/amazon_{src_domain}.csv', delimiter=',')
    data_t = np.loadtxt(f'{folder}/amazon_{tar_domain}.csv', delimiter=',')
    Xs, Ys = data_s[:, :-1], data_s[:, -1]
    Xt, Yt = data_t[:, :-1], data_t[:, -1]
    return Xs, Ys, Xt, Yt

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(
            scipy.linalg.fractional_matrix_power(cov_src, -0.5),
            scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.real(np.dot(Xs, A_coral))
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


# if __name__ == '__main__':
#     folder = '../../office31'
#     src_domain = 'amazon'
#     tar_domain = 'webcam'
#     Xs, Ys = load_data(folder, src_domain)
#     Xt, Yt = load_data(folder, tar_domain)
#     print('Source:', src_domain, Xs.shape, Ys.shape)
#     print('Target:', tar_domain, Xt.shape, Yt.shape)
#     coral = CORAL()
#     acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
#     print(acc)


if __name__ == "__main__":
    # download the dataset here: https://www.jianguoyun.com/p/DcNAUg0QmN7PCBiF9asD (Password: qqLA7D)
    folder = '../../office31_resnet50'
    src_domain = 'amazon'
    tar_domain = 'webcam'
    Xs, Ys, Xt, Yt = load_csv(folder, src_domain, tar_domain)
    print('Source:', src_domain, Xs.shape, Ys.shape)
    print('Target:', tar_domain, Xt.shape, Yt.shape)
    coral = CORAL()
    acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
    print(acc)