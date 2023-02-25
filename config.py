import os
from sacred import Experiment

ex = Experiment("PSTL", save_git_info=False) 

@ex.config
def my_config():
    ############################## setting ##############################
    version = "ntu60_xsub_j"
    dataset = "ntu60"
    split = "xsub"
    view = "joint"
    save_lp = False
    save_finetune = False
    save_semi = False
    pretrain_epoch = 150
    ft_epoch = 150
    lp_epoch = 150
    pretrain_lr = 5e-3
    lp_lr = 0.01
    ft_lr = 5e-3
    label_percent = 0.1
    weight_decay = 1e-5
    hidden_size = 256
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################ down stream ############################
    weight_path = './output/multi_model/xsub/v'+version+'_epoch_150_pretrain.pt'
    train_mode = 'lp'
    # train_mode = 'finetune'
    # train_mode = 'pretrain'
    # train_mode = 'semi'
    log_path = './output/log/v'+version+'_'+train_mode+'.log'
    ############################# 3s stream #############################
    result_path = './result/'+dataset+'/'+split+'/'+view+'/'+version+'_'
    label_path = './result/'+dataset+'/'+split+'/label/label.pkl'
    ################################ GPU ################################
    # gpus = "0"
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    ########################## Skeleton Setting #########################
    batch_size = 128
    channel_num = 3
    person_num = 2
    joint_num = 25
    max_frame = 50
    train_list = '/mnt/petrelfs/zhouyujie/data/'+dataset+'_frame50/'+split+'/train_data_joint.npy'
    test_list = '/mnt/petrelfs/zhouyujie/data/'+dataset+'_frame50/'+split+'/val_data_joint.npy'
    train_label = '/mnt/petrelfs/zhouyujie/data/'+dataset+'_frame50/'+split+'/train_label.pkl'
    test_label = '/mnt/petrelfs/zhouyujie/data/'+dataset+'_frame50/'+split+'/val_label.pkl'
    ########################### Data Augmentation #########################
    temperal_padding_ratio = 6
    shear_amp = 1
    mask_joint = 8
    mask_frame = 10
    ############################ Barlow Twins #############################
    pj_size = 6144
    lambd = 2e-4