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
import torchvision.transforms as transforms
from miscc.config import cfg


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
                 loader=default_loader, square=False, data_path=None, partition=None,
                 base_size=64):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition


        with open(os.path.join(data_path,self.partition+".json")) as f:
            self.data = json.load(f)
            if self.partition=='train':
                self.data = self.data[:10000]
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

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):


        imgs = self.data[index]["images"]
        if self.partition == 'train':
            # We do only use the first five images per recipe during training
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            imgIdx = 0

        img_path = imgs[imgIdx]

        #load image
        img = default_loader(self.img_dir, img_path)

        wrong_imgs = []

        if cfg.GAN.B_CONDITION:
            # loading wrong images
            wrong_index = np.random.randint(0,len(self.data))
            while wrong_index==index:
                wrong_index = np.random.randint(0,len(self.data))

            wrong_imgs = self.data[wrong_index]["images"]
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                wrong_imgIdx = np.random.choice(range(min(5, len(wrong_imgs))))
            else:
                wrong_imgIdx = 0

            wrong_img_path = wrong_imgs[wrong_imgIdx]

            wrong_img = default_loader(self.img_dir,wrong_img_path)

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

        if self.transform is not None:
            img = self.transform(img)
            if cfg.GAN.B_CONDITION:
                wrong_img = self.transform(wrong_img)

        imgs = []
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(self.imsize[i])(img)
            else:
                re_img = img
            imgs.append(self.norm(re_img))

        if cfg.GAN.B_CONDITION:
            wrong_imgs = []
            for i in range(cfg.TREE.BRANCH_NUM):
                if i < (cfg.TREE.BRANCH_NUM - 1):
                    re_img = transforms.Scale(self.imsize[i])(wrong_img)
                else:
                    re_img = wrong_img
                wrong_imgs.append(self.norm(re_img))
        # output

        return imgs, wrong_imgs, instrs_encoding, instrs_len, ingrs_encoding, ingrs_len


    def __len__(self):
    	return len(self.data)