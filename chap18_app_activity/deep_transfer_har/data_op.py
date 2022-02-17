import numpy as np
import torch
from scipy import io
import os

class DSADS27(torch.utils.data.Dataset):
    def __init__(self, data):
        self.samples = data[:, :405]
        self.labels = data[:, -2]

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        # from sklearn.preprocessing import StandardScaler
        # sample = StandardScaler().fit_transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


def load_27data(batch_size=100):
    root_path = '/D_data/jindwang/Dataset_PerCom18_STL'
    data = io.loadmat(os.path.join(root_path, 'dsads'))['data_dsads']
    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, test_size=.1)
    data_train, data_val = train_test_split(data_train, test_size=.2)
    train_set, test_set, val_set = DSADS27(
        data_train), DSADS27(data_test), DSADS27(data_val)
    train_loader, test_loader, val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True), torch.utils.data.DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False, drop_last=False), torch.utils.data.DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # merge_data()
    # get_data()
    a, _, _ = load_27data()
    for data, label in a:
        print(data.shape, label)
        break
