from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
from miscc.config import cfg, cfg_from_file
from torch.autograd import Variable

from model import G_NET
from PIL import Image, ImageFont, ImageDraw
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import math
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Generate an image.')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()
    return args

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_checkpoint(modelpath):
    # s_gpus = cfg.GPU_ID.split(',')
    # gpus = [int(ix) for ix in s_gpus]
    # torch.cuda.set_device(gpus[0])
    # state_dict = torch.load(modelpath, map_location=lambda storage, loc: storage)
    # netG = G_NET()
    # netG.apply(weights_init)
    # netG = torch.nn.DataParallel(netG, device_ids=gpus)
    # netG.load_state_dict(state_dict)
    # netG.eval()
    state_dict = torch.load(modelpath, map_location='cpu')
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k[7:]] = v
    netG = G_NET()

    netG.load_state_dict(new_state_dict)
    netG.eval()

    return netG

def prepare_data_with_text(data):
   
    instrs_emb = data[0]
    ingrs_emb = data[2]

    if cfg.CUDA:
        vembedding_instrs = Variable(instrs_emb).cuda()
        vembedding_ingrs = Variable(ingrs_emb).cuda()
        v_instrs_len = Variable(data[1]).cuda()
        v_ingrs_len = data[3]
    else:
        vembedding_instrs = Variable(instrs_emb).unsqueeze(0)
        vembedding_ingrs = Variable(ingrs_emb).unsqueeze(0)
        v_instrs_len = Variable(data[1])
        v_ingrs_len = Variable(data[3])

    return [vembedding_instrs, v_instrs_len, vembedding_ingrs, v_ingrs_len]

def generate_img(model):
    out_dir = os.getcwd() + '/out_img'
    with torch.no_grad():
        data = []
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(1, nz))

         
        noise.data.normal_(0, 1)

        with open(os.path.join('../data', "instrs2num.json")) as f:
            maxInst = 20
            instrs2num = json.load(f)
            instrs_encoding = np.zeros((maxInst))

            instrs_list = [
            "grill",
            "heat",
            "simmer"
        ]
            for i,v in enumerate(instrs_list):
                instrs_encoding[i] = instrs2num[v]

            # instrs_encoding[0] = instrs2num['boil']
            # instrs_encoding[1] = instrs2num['']
            # instrs_encoding[2] = instrs2num['']
            # instrs_encoding[3] = instrs2num['']
            # instrs_encoding[4] = instrs2num['']

            instrs_len = len(instrs_list) if len(instrs_list) < maxInst else maxInst
            instrs_len = torch.LongTensor(np.array([instrs_len]))
            instrs_encoding = torch.LongTensor(instrs_encoding)
            data.append(instrs_encoding)
            data.append(instrs_len)

        with open(os.path.join('../data', "ingrs2num.json")) as f:
            maxIngr = 20
            ingrs2num = json.load(f)
            ingrs_encoding = np.zeros((maxIngr))
            ingrs_list = [
            "pineapple juice",
            "cornstarch",
            "light soy sauce",
            "white vinegar",
            "garlic cloves",
            "fresh ginger",
            "halibut",
            "pineapple",
            "red onions"
        ]
            for i,v in enumerate(ingrs_list):
                ingrs_encoding[i] = ingrs2num[v]
            # ingrs_encoding[0] = ingrs2num['noodles']
            # ingrs_encoding[1] = ingrs2num['']
            # ingrs_encoding[2] = ingrs2num['']
            # ingrs_encoding[3] = ingrs2num['']
            # ingrs_encoding[4] = ingrs2num['']

            ingrs_len = len(ingrs_list) if len(ingrs_list) < maxIngr else maxIngr
            ingrs_len = torch.LongTensor(np.array([ingrs_len]))
            ingrs_encoding = torch.LongTensor(ingrs_encoding)
            data.append(ingrs_encoding)
            data.append(ingrs_len)
        for j in range(10):
            

            txt_embedding = prepare_data_with_text(data)

            fake_imgs, _,_ = model(noise, txt_embedding)

            image_name = ""
            for i in ingrs_list:
                image_name+=i+"_"

            for i in instrs_list:
                image_name+=i+"_"

            img_folder = os.path.join(out_dir,image_name)
            image_name+=str(j)+".jpg"
            output_path = os.path.join(img_folder,image_name)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            # fake_img = (fake_imgs[1] + 1) /2.0
            # g_img = fake_img.numpy()[0].transpose(1,2,0)*255
            img = fake_imgs[1][0].add(1).div(2).mul(255).clamp(0, 255).byte()
            print(img.shape)
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(output_path)
            # cv2.imwrite(output_path, ndarr[:,:,::-1])
            print('*' * 30)
            print('Output Image generated at:', output_path)
            print('*' * 30)

if __name__ == '__main__':
    args = parse_args()

    # Config 1: GPUs
    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    # Config 2: model dir
    if args.model_dir is not None:
        modelpath = args.model_dir
    else:
        raise ValueError('Must specify model to be tested.')

    # Main test
    start_t = time.time()
    model = load_checkpoint(modelpath)
    generate_img(model)
    end_t = time.time()
    print('Total time for testing:', end_t - start_t)
