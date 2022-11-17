from operator import index
from collections import Counter
import torch
import os
import cv2
cv2.setNumThreads(0)
import numpy as np
from torch.utils.data import Dataset
import time
import pandas as pd

def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):
    def __init__(self, feature, label, cap, class_embed, des_embed, load_clips=load_clips_tsn,
                 clip_len=16, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):

        self.data = torch.from_numpy(np.load(feature)).float()
        label_array = np.load(label)
        index_list= label_array.tolist()
        self.freq = Counter(index_list)
        self.class_index = np.unique(index_list)
        self.label_array = torch.from_numpy(label_array).long()
        self.cap_feature = torch.from_numpy(np.load(cap)).float()
        self.class_embed = torch.from_numpy(np.load(class_embed)).float()
        self.des_embed = torch.from_numpy(np.load(des_embed)).float()

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        buffer = self.data[idx]
        label = self.label_array[idx]
        cap_feature = self.cap_feature[idx]
        return buffer, cap_feature, self.class_embed[label], label, self.des_embed[label]

    def __len__(self):
        return len(self.data)
    
    def statistics(self,df):
        tmp = df.to_dict('index')
        frame_list = []
        index_list = []
        for i in range(len(tmp)):
            frame_list.append(tmp[i]['path'])
            index_list.append(tmp[i]['label'])
        freq = Counter(index_list)
        class_index = np.unique(index_list)
        return np.array(frame_list), np.array(index_list), class_index, freq

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels