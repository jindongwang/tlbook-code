#coding=utf-8
import os,sys
import numpy as np
import math
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from models import *
from util import *
from dataloader_medmnist import *
import datasplit
import copy
def prepare_data(args):
    ld=args.n_clients
    tmpdata=MedMnistDataset()
    traindata,testdata=datasplit.getdataloader1(args,tmpdata)
    traindataloader,testdataloader=[],[]
    for i in range(ld):
        traindataloader.append(torch.utils.data.DataLoader(traindata[i],batch_size=args.batch,shuffle=True))
        testdataloader.append(torch.utils.data.DataLoader(testdata[i],batch_size=args.batch,shuffle=False))
    return traindataloader,testdataloader

def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,_ = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_loader), correct/num_data

def test1(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            targets.append(target.detach().cpu().numpy())

            output,_ = model(data)
            
            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]

            correct += pred.eq(target.view(-1)).sum().item()
        
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def getwasserstein(m1,v1,m2,v2,mode='nosquare'):
    w=0
    bl=len(m1)
    for i in range(bl):
        tw=0
        tw+=(np.sum(np.square(m1[i]-m2[i])))
        tw+=(np.sum(np.square(np.sqrt(v1[i])- np.sqrt(v2[i]))))
        if mode=='square':
            w+=tw
        else:
            w+=math.sqrt(tw)
    return w

def fedhealth_copy(args,server_model,models,weight_m,sharew='no'):
    client_num=len(models)
    tmpmodels=[]
    for i in range(client_num):
        tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
    with torch.no_grad():
        for cl in range(client_num):
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += weight_m[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                if ('bn' not in key) or (sharew=='yes'):
                    models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return models

def communication(args, server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                if 'bn' not in key:
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower()=='fedbn_m':
            models=fedhealth_copy(args,server_model,models,client_weights,'no')
        elif args.mode.lower()=='fedavg':
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def pretrained(args,filename,device='cuda'):
    print('===training pretrained model===')
    model=lenet5v1().to(device)
    data=MedMnistDataset()
    predata=datasplit.define_pretrain_dataset(args,data)
    traindata=torch.utils.data.DataLoader(predata,batch_size=args.batch,shuffle=True)
    loss_fun = nn.CrossEntropyLoss()  
    opt=optim.SGD(params=model.parameters(), lr=args.lr)
    for i in range(args.iters):
        loss,acc=train(model,traindata,opt,loss_fun,0,device)
    torch.save({
        'state':model.state_dict(),
        'acc':acc
    },filename)
    print('===done!===')

def get_form(model):
    tmpm=[]
    tmpv=[]
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm,tmpv

class metacount(object):
    def __init__(self,numpyform):
        super(metacount,self).__init__()
        self.count=0
        self.mean=[]
        self.var=[]
        self.bl=len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self,m,tm,tv):
        tmpcount=self.count+m
        for i in range(self.bl):
            tmpm=(self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i]=(self.count*(self.var[i]+np.square(tmpm-self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i]=tmpm
        self.count=tmpcount
    
    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var

def get_weight_matrix1(args,bnmlist,bnvlist,client_weights):
    client_num=len(bnmlist)
    weight_m=np.zeros((client_num,client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i==j:
                weight_m[i,j]=0
            else:
                tmp=getwasserstein(bnmlist[i],bnvlist[i],bnmlist[j],bnvlist[j])
                if tmp==0:
                    weight_m[i,j]=100000000000000
                else:
                    weight_m[i,j]=1/tmp
    weight_s=np.sum(weight_m,axis=1)
    weight_s=np.repeat(weight_s,client_num).reshape((client_num,client_num))
    weight_m=(weight_m/weight_s)*(1-args.model_momentum)
    for i in range(client_num):
        weight_m[i,i]=args.model_momentum
    return weight_m

def get_weight_preckpt(args,preckpt,trainloadrs,client_weights,device='cuda'):
    model=lenet5v1().to(device)
    model.load_state_dict(torch.load(preckpt)['state'])
    model.eval()
    bnmlist1,bnvlist1=[],[]
    for i in range(args.n_clients):
        avgmeta=metacount(get_form(model)[0])
        with torch.no_grad():
            for data,_ in trainloadrs[i]:      
                data=data.to(device).float()     
                _,fea=model(data)
                nl=len(data)
                tm,tv=[],[]
                for item in fea:
                    if len(item.shape)==4:
                        tm.append(torch.mean(item,dim=[0,2,3]).detach().to('cpu').numpy())
                        tv.append(torch.var(item,dim=[0,2,3]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(item,dim=0).detach().to('cpu').numpy())
                        tv.append(torch.var(item,dim=0).detach().to('cpu').numpy())
                avgmeta.update(nl,tm,tv)
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m=get_weight_matrix1(args,bnmlist1,bnvlist1,client_weights)
    return weight_m

def fed_train(args,datasetname,device):
    train_loaders, test_loaders = prepare_data(args)
    client_num = len(datasetname)
    client_weights = [1/client_num for i in range(client_num)]

    if args.mode=='fedhealth':
        os.makedirs('./checkpoint/'+'pretrained/',exist_ok=True)
        preckpt='./checkpoint/'+'pretrained/'+args.data+'_'+str(args.batch)
        if not os.path.exists(preckpt):
            pretrained(args,preckpt)
        weight_m=get_weight_preckpt(args,preckpt,train_loaders,client_weights)
        args.mode='fedbn_m'

    loss_fun = nn.CrossEntropyLoss()   
    server_model =lenet5v1().to(device)
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    
    for a_iter in range(args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train(model, train_loader, optimizer, loss_fun, client_num, device)
        
        if args.mode=='fedbn_m':
            server_model, models = communication(args, server_model, models, weight_m)
        else:
            server_model, models = communication(args, server_model, models, client_weights)
        
        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            train_loss, train_acc = test1(model, train_loader, loss_fun, device) 
            print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasetname[client_idx] ,train_loss, train_acc))
        tmp=[]
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test1(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasetname[test_idx], test_loss, test_acc))
            tmp.append(test_acc)
        print(np.mean(np.array(tmp)))

def evaluate(args,datasetname,device,method):
    _, test_loaders = prepare_data(args)
    loss_fun = nn.CrossEntropyLoss()   
    modelpath='./OCmodels/'+method
    modeldata=torch.load(modelpath)
    server_model =lenet5v1().to(device)
    client_num=args.n_clients
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]    
    if method=='fedavg':
        for i in range(client_num):
            models[i].load_state_dict(modeldata['server_model'])
    else:
        for i in range(client_num):
            models[i].load_state_dict(modeldata['model_'+str(i)])
    print('===start evaluating===')
    accsum=0
    for i in range(client_num):
        print('===client num: %d ==='%i)
        test_loss, test_acc =test1(models[i],test_loaders[i],loss_fun,device)
        print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasetname[i], test_loss, test_acc))
        accsum+=test_acc
    print('=== average acc: %.4f ==='%(accsum/client_num))

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)
    args=config(batch=32,wk_iters=1,iters=400)
    args.data='organcmnist'
    datasetname=[]
    for i in range(args.n_clients):
        datasetname.append('organcmnist_'+str(i))
    args.num_classes=10
    test=int(sys.argv[1])
    method=sys.argv[2]
    args.mode=method

    if test==1:
        evaluate(args,datasetname,device,method)
    else:
        fed_train(args,datasetname,device)
