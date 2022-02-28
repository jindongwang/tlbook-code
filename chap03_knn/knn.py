import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()


def load_csv(folder, src_domain, tar_domain):
    data_s = np.loadtxt(f'{folder}/amazon_{src_domain}.csv', delimiter=',')
    data_t = np.loadtxt(f'{folder}/amazon_{tar_domain}.csv', delimiter=',')
    Xs, Ys = data_s[:, :-1], data_s[:, -1]
    Xt, Yt = data_t[:, :-1], data_t[:, -1]
    return Xs, Ys, Xt, Yt


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

    knn_classify(Xs, Ys, Xt, Yt, k=1, norm=args.norm)
