import numpy as np
import torch
from torch.utils.data import Dataset

def get_data_medmnist(file='./data/organcmnist.npz'):
    data=np.load(file)
    train_data=np.vstack((data['train_images'],data['val_images'],data['test_images']))
    y=np.hstack((np.squeeze(data['train_labels']),np.squeeze(data['val_labels']),np.squeeze(data['test_labels'])))
    return train_data,y

class MedMnistDataset(Dataset):
    def __init__(self, filename='./data/organcmnist.npz'):
        self.data,self.targets=get_data_medmnist(filename)
        self.targets=np.squeeze(self.targets)

        self.data=torch.Tensor(self.data)
        self.data=torch.unsqueeze(self.data,dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
