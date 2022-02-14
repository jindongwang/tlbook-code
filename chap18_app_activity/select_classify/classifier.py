from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self, data) -> None:
        self.x_src, self.y_src, self.x_tar, self.y_tar = data

    def fit_predict_all(self):
        for algo in ['knn']:
            print(f'Algo: {algo}:')
            exec(f'self.{algo}()')

    def svm(self, verbose=False):
        from sklearn import svm
        best_acc, best_c, best_ker = 0, 0, 0
        for c in [0.1, 0.5, 1, 3, 5, 10]:
            for ker in ['linear', 'rbf']:
                clf = svm.SVC(C=c, kernel=ker)
                clf.fit(self.x_src, self.y_src)
                ypred = clf.predict(self.x_tar)
                acc = accuracy_score(ypred, self.y_tar)
                if verbose:
                    print(f'C: {c}, ker: {ker}, acc: {acc}')
                if acc > best_acc:
                    best_acc = acc
                    best_c, best_ker = c, ker
        print(f'Best acc: {best_acc}, C: {best_c}, ker: {best_ker}')

    def knn(self, verbose=False):
        from sklearn.neighbors import KNeighborsClassifier
        best_acc, best_k = 0, 0
        for k in [1, 3, 5]:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(self.x_src, self.y_src)
            ypred = clf.predict(self.x_tar)
            acc = accuracy_score(ypred, self.y_tar)
            if verbose:
                print(f'K: {k}, acc: {acc}')
            if acc > best_acc:
                best_acc = acc
                best_k = k
        print(f'Best acc: {best_acc}, K: {best_k}')

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        best_acc, best_n = 0, 0
        for n in [2, 5, 10, 20, 30, 50, 100]:
            clf = RandomForestClassifier(n_estimators=n)
            clf.fit(self.x_src, self.y_src)
            ypred = clf.predict(self.x_tar)
            acc = accuracy_score(self.y_tar, ypred)
            if acc > best_acc:
                best_acc = acc
                best_n = n
        print(f'Best acc: {best_acc}, n: {best_n}')


if __name__ == '__main__':
    import numpy as np
    x_src, x_tar = np.random.randn(200, 2), np.random.randn(200, 2)
    y_src, y_tar = np.random.randint(
        0, 2, (200,)), np.random.randint(0, 2, (200,))
    print(x_src, y_src, x_tar, y_tar)
    cls = Classifier((x_src, y_src, x_tar, y_tar))
    cls.fit_predict_all()
