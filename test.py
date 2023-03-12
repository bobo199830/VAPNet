from random import betavariate, shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from network import VAPNet
import ipdb
import argparse
import numpy as np
import random
from torch.optim import lr_scheduler
import itertools
from tqdm import tqdm
import os
import warnings
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("ViA")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=10)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--manualSeed', type=int, default=557, help='manual seed')
parser.add_argument('--dataset', choices=['UCF101', 'HMDB51'], type=str, default='UCF101', help='test dataset')
parser.add_argument('--clip_num', choices=[1, 25], type=int, default=1, help='clip number for each video')
parser.add_argument('--in_dim', type=int, default=384, help='input_dim for video attribute generation')
parser.add_argument('--num_heads', type=int, default=8, help='attention heads for video attribute generation')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/VAPNet662_checkpoint_UCF.pth',help='load checkpoint')
opt = parser.parse_args()
print(opt)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

if opt.dataset == 'UCF101':
    unseen_dataset = VideoDataset(feature='./data/UCF101/ucf_video_feature_'+str(opt.clip_num)+'.npy', label='./data/UCF101/ucf_video_label.npy',cap='./data/UCF101/ucf_cap_clean.npy',class_embed='./data/UCF101/ucf_cls.npy',des_embed='./data/UCF101/ucf_des.npy')
elif opt.dataset == 'HMDB51':
    unseen_dataset = VideoDataset(feature='./data/HMDB51/hmdb_video_feature_'+str(opt.clip_num)+'.npy', label='./data/HMDB51/hmdb_video_label.npy',cap='./data/HMDB51/hmdb_cap_clean.npy',class_embed='./data/HMDB51/hmdb_cls.npy',des_embed='./data/HMDB51/hmdb_des.npy')

unseen_loader = DataLoader(unseen_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.workers)

def compute_accuracy(predicted_embed, true_cal_class_embedding, true_label, write_name):
    y_pred = []
    for k in range(predicted_embed.shape[0]):
        pred = cdist(predicted_embed[k][np.newaxis,:], true_cal_class_embedding[k], 'cosine').argsort(1)
        y_pred.append(pred)
    y_pred = np.array(y_pred).squeeze()
    y = true_label
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5

model = VAPNet(opt)
model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.checkpoint))

model.eval()
unseen_cls = unseen_dataset.class_embed[unseen_dataset.class_index].cuda()
unseen_des = unseen_dataset.des_embed[unseen_dataset.class_index].cuda()
print('Testing...')
with torch.no_grad():
    n_samples = len(unseen_dataset)
    predicted_embed = np.zeros([n_samples, 2048], 'float32')
    if opt.dataset == 'UCF101':
        true_cal_class_embedding = np.zeros([n_samples,101,2048],'float32')
    elif opt.dataset == 'HMDB51':
        true_cal_class_embedding = np.zeros([n_samples,51,2048],'float32')
    true_label = np.zeros(n_samples, 'int')
    fi = 0
    for ii,(data,cap,_,label,des) in enumerate(tqdm(unseen_loader)):
        torch.cuda.empty_cache()
        feature = model.module.vhead(data.cuda()).cpu().detach().numpy()
        unseen_att = []
        for tmp in cap:
            _, mean, _ = model.module.un(tmp.repeat(unseen_cls.shape[0],1).cuda())
            tmp1 = model.module.ca1(mean, unseen_cls, unseen_cls)
            tmp2 = model.module.ca2(mean, unseen_des, unseen_des)
            tmp1 = (tmp1+tmp2)
            unseen_att.append(tmp1)
        unseen_att1 = torch.stack(unseen_att)
        unseen_att = model.module.ahead(unseen_att1.flatten(end_dim=1))
        unseen_att = unseen_att.reshape(unseen_att1.shape[0],unseen_att1.shape[1],-1).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        predicted_embed[fi:fi+len(label)] = feature
        true_label[fi:fi+len(label)] = label
        true_cal_class_embedding[fi:fi+len(label)] = unseen_att
        fi += len(label)
predicted_embed = predicted_embed[:fi]
true_label = true_label[:fi]
accuracy, accuracy_top5 = compute_accuracy(predicted_embed, true_cal_class_embedding, true_label, 'Protocol1')
print(opt.dataset+" with clip_num="+str(opt.clip_num)+": Unseen top-1 acc:", '%.1f'%accuracy)
print(opt.dataset+" with clip_num="+str(opt.clip_num)+": Unseen top-5 acc:", '%.1f'%accuracy_top5)

