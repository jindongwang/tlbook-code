import torch
import torch.nn as nn
import numpy as np
import argparse

from network import TNNAR
from mmd import MMD_loss, CORAL_loss
from data_op import load_27data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--nclass', type=int, default=19)
    parser.add_argument('--nepochs', type=int, default=5)
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--loss', type=str, default='coral')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--mode', type=str, default='ratola')
    args = parser.parse_args()
    return args


def train(model, loaders, optimizer):
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.nepochs):
        model.train()
        total_loss = 0
        correct = 0
        for _, (data, label) in enumerate(loaders[0]):
            data, label = data[:,243:324].float().cuda(), label.long().cuda() - 1
            data = data.view(-1, 9, 9, 1)
            _, out = model(data)
            loss = criterion(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == label).sum()
        train_acc = float(correct) / len(loaders[0].dataset)
        train_loss = total_loss / len(loaders[0])
        val_acc = test(model, loaders[1])
        test_acc = test(model, loaders[2])
        if best_acc < test_acc:
            best_acc = test_acc
        print(f'Epoch: [{epoch:2d}/{args.nepochs}] loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
    print(f'Best acc: {best_acc}')


def train_da(model, loaders, optimizer, mode='ratola'):
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.nepochs):
        model.train()
        total_loss = 0
        correct = 0
        for (src, tar) in zip(loaders[0][0], loaders[1][0]):
            xs, ys = src
            xt, yt = tar
            xs = xs[:, 81:162] if mode == 'ratola' else xs[:, 243:324]
            xt = xt[:, 162:243] if mode == 'ratola' else xt[:, 324:405]
            xs, ys, xt, yt = xs.float().cuda(), ys.long().cuda() - 1, xt.float().cuda(), yt.float().cuda() - 1
            # data, label = data[:,243:324].float().cuda(), label.long().cuda() - 1
            xs, xt = xs.view(-1, 9, 9, 1), xt.view(-1, 9, 9, 1)
            # data = data.view(-1, 9, 9, 1)
            fs, outs = model(xs)
            ft, _ = model(xt)
            loss_cls = criterion(outs, ys)
            mmd = MMD_loss(kernel_type='rbf')(fs, ft) if args.loss == 'mmd' else CORAL_loss(fs, ft)
            # mmd = 
            loss = loss_cls + args.lamb * mmd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outs.data, 1)
            correct += (predicted == ys).sum()
        train_acc = float(correct) / len(loaders[0][0].dataset)
        # train_acc = 0
        train_loss = total_loss / len(loaders[0])
        val_acc = test(model, loaders[1][1])
        #test_acc = test(model, loaders[1][2])
        test_acc = 0
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model4.pkl')
        print(f'Epoch: [{epoch:2d}/{args.nepochs}] loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
    acc_final = test(model, loaders[1][2], 'model4.pkl')
    print(f'Best acc: {acc_final}')



def test(model, loader, model_path=None):
    if model_path:
        print('Load model...')
        model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.float().cuda(), label.long().cuda() - 1
            data = data[:, 162:243] if args.mode == 'ratola' else data[:, 324:405]
            data = data.view(-1, 9, 9, 1)
            pred = model.predict(data)
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == label).sum()
    acc = float(correct) / len(loader.dataset)
    return acc


if __name__ == '__main__':
    args = get_args()
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loaders = load_27data(args.batchsize)
    loaders2 = load_27data(args.batchsize)
    net = TNNAR(n_class=args.nclass).cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    train_da(net, (loaders, loaders2), optimizer, mode=args.mode)
    
