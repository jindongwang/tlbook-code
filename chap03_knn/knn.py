import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()

def load_data(folder, domain):
    from scipy import io
    data = io.loadmat(os.path.join(folder, domain + '_fc6.mat'))
    return data['fts'], data['labels']


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
    print(f'Accuracy using kNN: {acc * 100:.2f}%')


if __name__ == "__main__":
    # download the dataset here: https://www.jianguoyun.com/p/DcNAUg0QmN7PCBiF9asD (Password: qqLA7D)
    folder = './office31'
    src_domain = 'amazon'
    tar_domain = 'webcam'
    Xs, Ys = load_data(folder, src_domain)
    Xt, Yt = load_data(folder, tar_domain)
    print('Source:', src_domain, Xs.shape, Ys.shape)
    print('Target:', tar_domain, Xt.shape, Yt.shape)

    knn_classify(Xs, Ys, Xt, Yt, k=1, norm=args.norm)

