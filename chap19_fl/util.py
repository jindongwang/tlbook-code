#coding=utf-8
import numpy as np

class config(object):
    def __init__(self,percent=0.1,data='organcmnist',save_path='./checkpoint/',lr=1e-2,batch=32,epochs=100,
    log=False,test=False,iters=300,wk_iters=1,mu=1e-3,mode='fedavg',resume=False,
    update_style='avg',serverdata=4,num_classes=10,model_momentum=0.5,device='cuda',sdata=False,
    n_clients=20,partition_data='non_iid_dirichlet',non_iid_alpha=0.05,random_state=np.random.RandomState(1),
    avg_opt='best',round=10):
        super(config,self).__init__()
        self.round=round
        self.percent=percent
        self.data=data
        self.save_path=save_path
        self.batch=batch
        self.lr=lr
        self.epochs=epochs
        self.log=log
        self.test=test
        self.iters=iters
        self.wk_iters=wk_iters
        self.mode=mode
        self.mu=mu
        self.resume=resume
        self.update_style=update_style
        self.serverdata=serverdata
        self.num_classes=num_classes
        self.model_momentum=model_momentum
        self.device=device
        self.sdata=sdata
        self.n_clients=n_clients
        self.non_iid_alpha=non_iid_alpha
        self.partition_data=partition_data
        self.random_state=random_state
        self.avg_opt=avg_opt