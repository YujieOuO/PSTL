from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from math import sin, cos
from einops import rearrange, repeat

def init_weights(m):
    class_name=m.__class__.__name__

    if "Conv2d" in class_name or "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Linear(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, dataset): 
        super(Linear, self).__init__()
        if "ntu60" in dataset:
            label_num = 60
        elif "ntu120" in dataset:
            label_num = 120
        elif "pku" in dataset:
            label_num = 51
        else:
            raise ValueError
        self.classifier = nn.Linear(hidden_size, label_num)
        self.apply(init_weights)

    def forward(self, X):
        X = self.classifier(X)
        return X

class BTwins(nn.Module):

    @ex.capture
    def __init__(self, hidden_size, lambd, pj_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )
        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.lambd = lambd

    def forward(self, feat1, feat2):
        
        feat1 = self.projector(feat1)
        feat2 = self.projector(feat2)
        feat1_norm = self.bn(feat1)
        feat2_norm = self.bn(feat2)

        N, D = feat1_norm.shape
        c = (feat1_norm.T @ feat2_norm).div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        BTloss = on_diag + self.lambd * off_diag

        return BTloss 

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
@ex.capture 
def get_stream(data, view):
    N, C, T, V, M = data.shape

    if view == 'joint':
        pass

    elif view == 'motion':
        motion = torch.zeros_like(data)
        motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

        data = motion

    elif view == 'bone':
        Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        bone = torch.zeros_like(data)

        for v1, v2 in Bone:
            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

        data = bone
    
    else:

        return None

    return data

@ex.capture
def shear(input_data, shear_amp):
    # n c t v m
    temp = input_data.clone()
    amp = shear_amp
    Shear       = np.array([
                    [1, random.uniform(-amp, amp), 	random.uniform(-amp, amp)],
                    [random.uniform(-amp, amp), 1, 	random.uniform(-amp, amp)],
                    [random.uniform(-amp, amp), 	random.uniform(-amp, amp),1]
                    ])
    Shear = torch.Tensor(Shear).cuda()
    output =  torch.einsum('n c t v m, c d -> n d t v m',[temp,Shear])

    return output
    
def reverse(data,p=0.5):

    N,C,T,V,M = data.shape
    temp = data.clone()

    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp
        
@ex.capture 
def crop(data, temperal_padding_ratio=6):
    input_data = data.clone()
    N, C, T, V, M = input_data.shape
    #padding
    padding_len = T // temperal_padding_ratio
    frame_start = torch.randint(0, padding_len * 2 + 1,(1,))
    first_clip = torch.flip(input_data[:,:,:padding_len],dims=[2])
    second_clip = input_data
    thrid_clip = torch.flip(input_data[:,:,-padding_len:],dims=[2])
    out = torch.cat([first_clip,second_clip,thrid_clip],dim=2)
    out = out[:, :, frame_start:frame_start + T]
    
    return out

def random_rotate(data):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        R = torch.Tensor(R).cuda()
        output =  torch.einsum('n c t v m, c d -> n d t v m',[seq,R])
        return output

    # n c t v m
    new_seq = data.clone()
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    return new_seq

@ex.capture
def get_ignore_joint(mask_joint):

    ignore_joint = random.sample(range(25), mask_joint)
    return ignore_joint

@ex.capture
def get_ignore_part(mask_part):

    left_hand = [8,9,10,11,23,24]
    right_hand = [4,5,6,7,21,22]
    left_leg = [16,17,18,19]
    right_leg = [12,13,14,15]
    body = [0,1,2,3,20]
    all_joint = [left_hand, right_hand, left_leg, right_leg, body]
    part = random.sample(range(5), mask_part)
    ignore_joint = []
    for i in part:
        ignore_joint += all_joint[i]

    return ignore_joint

def gaus_noise(data, mean= 0, std = 0.01):
    temp = data.clone()
    n, c, t, v, m = temp.shape
    noise = np.random.normal(mean, std, size=(n, c, t, v, m))
    noise = torch.Tensor(noise).cuda()

    return temp + noise

def gaus_filter(data):
    temp = data.clone()
    g = GaussianBlurConv(3).cuda()
    return g(temp)

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        kernel =  kernel.float()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,5)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.weight = self.weight.cuda()

        prob = np.random.random_sample()
        if prob < 0.5:
            #x = x.permute(3,0,2,1) # M,C,V,T
            x = rearrange(x, 'n c t v m -> (n m) c v t')
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            #x = x.permute(1,-1,-2, 0) #C,T,V,M
            x = rearrange(x, '(n m) c v t -> n c t v m', m = 2)

        return x

@ex.capture
def temporal_cropresize(input_data,max_frame,output_size,l_ratio=[0.1,1]):

    num_of_frames = max_frame
    
    n, c, t, v, m = input_data.shape
    min_crop_length = 64
    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)
    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:, :,start:start+temporal_crop_length, :, :]
    temporal_context = rearrange(temporal_context,'n c t v m -> n (c v m) t')
    temporal_context=temporal_context[: , :, :,None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=-1)
    temporal_context = rearrange(temporal_context,'n (c v m) t -> n c t v m',c=c,v=v,m=m)
    return temporal_context

def random_spatial_flip(data, p=0.5):
    temp = data.clone()
    order = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 
    17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
    if random.random() < p:
        temp = temp[:, :, :, order, :]

    return temp

def random_time_flip(temp, p=0.5):
    # temp = data.clone()
    T = temp.shape[2]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp

@ex.capture
def motion_att_temp_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :]

    return output

@ex.capture
def central_spacial_mask(mask_joint):

    # Degree Centrality
    degree_centrality = [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
                        2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 1, 2]
    all_joint = []
    for i in range(25):
        all_joint += [i]*degree_centrality[i]

    ignore_joint = random.sample(all_joint, mask_joint)

    return ignore_joint


def semi_mask(mask_num):

    p = random.random()
    if p<0.5:
        ignore_joint = central_spacial_mask(mask_num)
    else:
        ignore_joint = []
    
    return ignore_joint