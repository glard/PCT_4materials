from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

import time

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ModelNet40(partition='validation', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                        batch_size=args.test_batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    print(str(model))
    print(f"The total trainable parameter number of current model is: {_count_parameters(model)} ")
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*5, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    # criterion = cal_loss
    reg_loss = nn.MSELoss()
    criterion = reg_loss
    #best_test_acc = 0
    best_mae = 9999

    for epoch in range(args.epochs):
        print('************************************epoch start************************************')
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            # print('************************************training start************************************')
            preds = model(data)
            # print(label.size())

            label = torch.reshape(label, (-1,1))
            # print(label.size())
            # print(preds.size())
            loss = criterion(preds, label)
            loss.backward()
            
            # added to adapt deeper layer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            #preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        # outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
        #                                                                         train_loss*1.0/count,
        #                                                                         metrics.accuracy_score(
        #                                                                         train_true, train_pred),
        #                                                                         metrics.balanced_accuracy_score(
        #                                                                         train_true, train_pred))
        outstr = 'Train %d, loss: %.6f, mean_absolute_error: %.6f, r2_score: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.mean_absolute_error(
                                                                                     train_true, train_pred),
                                                                                 metrics.r2_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Validation
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).squeeze()
            # added
            label = torch.reshape(label, (-1, 1))
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            preds = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(preds, label)
            #preds = logits.max(dim=1)[1]
            # preds = logits

            count += batch_size
            test_loss += loss.item() * batch_size
            # test_true.append(label.cpu().numpy())
            # test_pred.append(preds.detach().cpu().numpy())
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        # test_acc = metrics.accuracy_score(test_true, test_pred)
        # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        mean_absolute_error = metrics.mean_absolute_error(test_true, test_pred)
        r2_score = metrics.r2_score(test_true, test_pred)
        # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
        #                                                                     test_loss*1.0/count,
        #                                                                     test_acc,
        #                                                                     avg_per_class_acc)
        outstr = 'Validation %d, loss: %.6f, mean_absolute_error: %.6f, r2_score: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              mean_absolute_error,
                                                                              r2_score)
        io.cprint(outstr)
        if mean_absolute_error <= best_mae:
            print('model with lower MAE: %.6f found!'% mean_absolute_error)
            best_mae = mean_absolute_error
            torch.save(model.state_dict(), 'checkpoints/' + str(args.exp_name) + '/models/model_' + str(args.num_points) + '.t7')


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        # preds = logits.max(dim=1)[1]
        preds = logits
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_true, test_pred)

    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    r2_score = metrics.r2_score(test_true, test_pred)
    # outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    outstr = 'Test :: test MAE: %.6f, test r^2: %.6f'%(mean_absolute_error, r2_score)

    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='pctm_20k_20sa_encoding_dim_d_0806', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    # revised based on bandgap custom num_points original 1024 67039 27431
    parser.add_argument('--num_points', type=int, default=27431,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='/model_27431.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
