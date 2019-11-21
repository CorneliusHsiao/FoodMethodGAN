from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch
import json
from new_args import get_parser

parser = get_parser()
opts = parser.parse_args()

def default_loader(folder, path):

    img_path = os.path.join(path[0],path[1],path[2],path[3],path)
    img_path = os.path.join(folder,img_path)

    try:
        im = Image.open(img_path).convert('RGB')
        return im
    except Exception as e:
        print(img_path)
        print(e)
        return Image.new('RGB', (112, 112), 'white')
       
class ImageLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition


        with open(os.path.join(data_path,self.partition+".json")) as f:
            self.data = json.load(f)
            if self.partition=='train':
                self.data = self.data[:100]
            else:
                self.data = self.data[:1000]

        self.img_dir = os.path.join(data_path,"img_data",self.partition)

        with open(os.path.join(data_path,"ingrs2num.json")) as f:
            self.ingrs2num = json.load(f)

        with open(os.path.join(data_path,"instrs2num.json")) as f:
            self.instrs2num = json.load(f)

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 10
        self.maxIngr = 20

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):

        # we force 80 percent of them to be a mismatch
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        target = match and 1 or -1

        # image
        if target == 1:
            imgs = self.data[index]["images"]
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
            else:
                imgIdx = 0

            img_path = imgs[imgIdx]

        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.data))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            rndimgs = self.data[rndindex]['images']

            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
            else:
                imgIdx = 0

            img_path = rndimgs[imgIdx]
            # path = self.imgPath + rndimgs[imgIdx]['id']

        #load image
        img = default_loader(self.img_dir, img_path)
        
        #instructions
        instrs_encoding = np.zeros((self.maxInst))
        instrs = self.data[index]['instructions']
        for i,v in enumerate(instrs):
            if i >= self.maxInst:
                break
            instrs_encoding[i] = self.instrs2num[v]
        instrs_len = len(instrs) if len(instrs) < self.maxInst else self.maxInst
        instrs_encoding = torch.LongTensor(instrs_encoding)


        #ingredients
        ingrs_encoding = np.zeros((self.maxIngr))
        ingrs = self.data[index]['ingredients']
        for i,v in enumerate(ingrs):
            if i >= self.maxIngr:
                break
            ingrs_encoding[i] = self.ingrs2num[v]
        ingrs_len = len(ingrs) if len(ingrs) < self.maxIngr else self.maxIngr
        ingrs_encoding = torch.LongTensor(ingrs_encoding)


        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        # output
        return [img, instrs_encoding, instrs_len, ingrs_encoding, ingrs_len], [target]


    def __len__(self):
    	return len(self.data)