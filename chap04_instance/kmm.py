"""
Kernel Mean Matching
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
"""

import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()


def load_csv(folder, src_domain, tar_domain):
    data_s = np.loadtxt(f'{folder}/amazon_{src_domain}.csv', delimiter=',')
    data_t = np.loadtxt(f'{folder}/amazon_{tar_domain}.csv', delimiter=',')
    Xs, Ys = data_s[:, :-1], data_s[:, -1]
    Xt, Yt = data_t[:, :-1], data_t[:, -1]
    return Xs, Ys, Xt, Yt


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma)
                       * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -
                   np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1),
                   self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


def knn_classify(Xs, Ys, Xt, Yt, k=1, norm=False):
    model = KNeighborsClassifier(n_neighbors=k)
    Ys = Ys.ravel()
    Yt = Yt.ravel()
    if norm:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.fit_transform(Xt)
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    acc = accuracy_score(Yt, Yt_pred)
    print(f'Accuracy: {acc * 100:.2f}%')


if __name__ == "__main__":
    folder = '../../office31_resnet50'
    src_domain = 'amazon'
    tar_domain = 'webcam'
    Xs, Ys, Xt, Yt = load_csv(folder, src_domain, tar_domain)
    print('Source:', src_domain, Xs.shape, Ys.shape)
    print('Target:', tar_domain, Xt.shape, Yt.shape)

    kmm = KMM(kernel_type='rbf', B=18)
    beta = kmm.fit(Xs, Xt)
    Xs_new = beta * Xs
    knn_classify(Xs_new, Ys, Xt, Yt, k=1, norm=args.norm)
