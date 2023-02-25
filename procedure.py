from config import *
from model import *
from dataset import DataSet, Feeder_semi
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np
import random
from tqdm import tqdm
import copy
from einops import rearrange, repeat
from math import pi, cos

from module.gcn.st_gcn import Model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
setup_seed(1)

class BaseProcessor:

    @ex.capture
    def load_data(self,train_list,train_label,test_list,test_label,batch_size,label_percent):
        self.dataset = dict()
        self.data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        # self.dataset['semi'] = Feeder_semi(train_list, train_label, label_percent)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=32,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            num_workers=32,
            shuffle=False)
        
        # self.data_loader['semi'] = torch.utils.data.DataLoader(
        #     dataset=self.dataset['semi'],
        #     batch_size=batch_size,
        #     num_workers=32,
        #     shuffle=True)
        
    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()
    
    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.test_epoch()
    
    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(self):
        
        pass

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

# %%
class RecognitionProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, lp_lr, lp_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
             lr=lp_lr,
             )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, lp_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self, epoch, lp_epoch, lp_lr):
        self.encoder.eval()
        self.classifier.train()

        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        Z = self.encoder(data)
        Z = Z.detach()
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/train/cls_acc", acc.item())
        self.log.update_batch("log/train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_lp):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)

            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_lp:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)

    def save_model(self):
        
        pass
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class SemiProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['semi']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label): 

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/semi_train/cls_acc", acc.item())
        self.log.update_batch("log/semi_train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_semi=True):
        self.encoder.eval()
        self.classifier.eval()

        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_semi10_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/semi_test/cls_acc", acc.item())
            self.log.update_batch("log/semi_test/cls_loss", loss.item())

        if save_semi:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class FTProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/finetune/cls_acc", acc.item())
        self.log.update_batch("log/finetune/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_finetune):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_finetune_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_finetune:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class BTProcessor(BaseProcessor):
    
    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,dropout,
                    graph_args,edge_importance_weighting):

        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                    hidden_dim=hidden_dim,dropout=dropout, 
                                    graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    )
        self.encoder = self.encoder.cuda()
        self.btwins_head = BTwins().cuda()

    @ex.capture
    def load_optim(self, pretrain_lr, pretrain_epoch, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.btwins_head.parameters()},
            ], 
            weight_decay=weight_decay,
            lr=pretrain_lr)

    def btwins_batch(self, feat1, feat2, mode):
        BTloss = self.btwins_head(feat1, feat2)
        BTloss = torch.mean(BTloss)
        self.log.update_batch("log/pretrain/"+mode+"_bt_loss", BTloss.item())
        return BTloss

    @ex.capture
    def train_epoch(self, epoch, pretrain_epoch, pretrain_lr):
        self.encoder.train()
        self.btwins_head.train()

        loader = self.data_loader['train']
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=pretrain_epoch, lr_max=pretrain_lr)
        
        for data, label in tqdm(loader):
            # load data
            n,c,t,v,m = data.shape
            data = data.type(torch.FloatTensor).cuda()
            data = get_stream(data)

            # get ignore joint
            ignore_joint = central_spacial_mask()

            # input1
            input1 = shear(crop(data))
            input1 = random_rotate(input1)
            input1 = random_spatial_flip(input1)
            feat1 = self.encoder(input1)

            # input2
            input2 = shear(crop(data))
            input2 = random_rotate(input2)
            input2 = random_spatial_flip(input2)
            # MATM
            input2 = motion_att_temp_mask(input2)
            feat2 = self.encoder(input2)

            # input3
            input3 = shear(crop(data))
            input3 = random_rotate(input3)
            input3 = random_spatial_flip(input3)
            # CSM
            feat3 = self.encoder(input3, ignore_joint)

            # loss
            loss_bt1 = self.btwins_batch(feat1, feat2, mode='temp_mask')
            loss_bt2 = self.btwins_batch(feat1, feat3, mode='joint_mask')

            loss = loss_bt1 + loss_bt2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
  
    @ex.capture
    def save_model(self,epoch,version):
        torch.save(self.encoder.state_dict(), f"output/multi_model/xsub/v"+version+"_epoch_"+str(epoch+1)+"_pretrain.pt")
        
    @ex.capture
    def optimize(self, pretrain_epoch):
        for epoch in range(pretrain_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
            self.train_epoch(epoch=epoch)
            if epoch+1 == pretrain_epoch:
                self.save_model(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,pretrain_epoch,lr=lr)
            
    @ex.capture
    def start(self):
        self.initialize()
        self.optimize()


# %%
@ex.automain
def main(train_mode):
    if "pretrain" in train_mode:
        p = BTProcessor()
    elif "lp" in train_mode:
        p = RecognitionProcessor()
    elif "finetune" in train_mode:
        p = FTProcessor()
    elif "semi" in train_mode:
        p = SemiProcessor()
    else:
        print('train_mode error')
    p.start()