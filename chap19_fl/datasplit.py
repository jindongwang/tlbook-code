#coding=utf-8
import numpy as np
import torch
import torch.distributed as dist

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return (self.data[data_idx][0],self.data[data_idx][1])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def set_targets(self,replaced_targets):
        self.replaced_targets=replaced_targets

    def get_targets(self):
        return self.replaced_targets

    def clean_replaced_targets(self):
        self.replaced_targets = None

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self, conf, data, partition_sizes, partition_type, consistent_indices=True
    ):
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices
        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)
        if self.partition_type=='evenlysplit':
            classes=np.unique(self.data.targets)
            lp=len(self.partition_sizes)
            ti=indices[:,0]
            ttar=indices[:,1]
            for i in range(lp):
                self.partitions.append(np.array([]))
            for c in classes:
                tindice=np.where(ttar==c)[0]
                lti=len(tindice)
                from_index = 0
                for i in range(lp):
                    partition_size=self.partition_sizes[i]
                    to_index = from_index + int(partition_size * lti)
                    if i==(lp-1):
                        self.partitions[i]=np.hstack((self.partitions[i],ti[tindice[from_index:]]))
                    else:
                        self.partitions[i]=np.hstack((self.partitions[i],ti[tindice[from_index:to_index]]))
                    from_index=to_index
            for i in range(lp):
                self.partitions[i]=self.partitions[i].astype(np.int).tolist()
        else:
            from_index = 0
            for partition_size in self.partition_sizes:
                to_index = from_index + int(partition_size * self.data_size)
                self.partitions.append(indices[from_index:to_index])
                from_index = to_index

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            self.conf.random_state.shuffle(indices)
        elif self.partition_type=='evenlysplit':
            indices =np.array([
                (idx, target)
                for idx, target in enumerate(self.data.targets)
                if idx in indices
            ])
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


def define_data_loader(conf, dataset,data_partitioner=None):
    world_size = conf.n_clients
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if data_partitioner is None:
        data_partitioner = DataPartitioner(
            conf, dataset, partition_sizes, partition_type=conf.partition_data
        )
    return data_partitioner

def getdataloader1(conf,dataall,root_dir='./split/'):
    file=root_dir+conf.data+'/partion_'+str(conf.non_iid_alpha)+'.npy'
    conf.partition_data='origin'
    data_part=define_data_loader(conf,dataall)
    data_part.partitions=np.load(file,allow_pickle=True).tolist()
    clienttrain_list=[]
    clienttest_list=[]
    for i in range(conf.n_clients):
        clienttrain_list.append(data_part.use(2*i))
        clienttest_list.append(data_part.use(2*i+1))
    return clienttrain_list,clienttest_list


def define_pretrain_dataset(conf, train_dataset):
    partition_sizes = [
        0.3,0.7
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenlysplit",
    )
    return data_partitioner.use(0)
