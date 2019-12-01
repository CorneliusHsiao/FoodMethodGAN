from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from torch.nn import functional as F
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
from miscc.config import cfg, cfg_from_file
from torch.autograd import Variable

from model import G_NET

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import math

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
    s_gpus = cfg.GPU_ID.split(',')
    gpus = [int(ix) for ix in s_gpus]
    torch.cuda.set_device(gpus[0])
    state_dict = torch.load(modelpath, map_location=lambda storage, loc: storage)
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

def prepare_data_with_text(data):
    instrs_emb = data[1]
    ingrs_emb = data[2]

    if cfg.CUDA:
        vembedding_instrs = Variable(instrs_emb).cuda()
        vembedding_ingrs = Variable(ingrs_emb).cuda()
        v_instrs_len = Variable(data[3]).cuda()
        v_ingrs_len = data[5]
    else:
        vembedding_instrs = Variable(instrs_emb)
        vembedding_ingrs = Variable(ingrs_emb)
        v_instrs_len = Variable(data[3])
        v_ingrs_len = Variable(data[5])

    return [vembedding_instrs, v_instrs_len, vembedding_ingrs, v_ingrs_len]

def generate_img(model):
    out_dir = os.getcwd() + '/out_img'
    with torch.no_grad():
        data = []

        with open(os.path.join('../../data', "instrs2num.json")) as f:
            maxInst = 20
            instrs2num = json.load(f)
            instrs_encoding = np.zeros((maxInst))

            instrs_encoding[0] = instrs2num['boil']
            # instrs_encoding[1] = instrs2num['']
            # instrs_encoding[2] = instrs2num['']
            # instrs_encoding[3] = instrs2num['']
            # instrs_encoding[4] = instrs2num['']

            instrs_len = len(instrs) if len(instrs) < maxInst else maxInst
            instrs_encoding = torch.LongTensor(instrs_encoding)

        with open(os.path.join('../../data', "ingrs2num.json")) as f:
            maxIngr = 20
            ingrs2num = json.load(f)
            ingrs_encoding = np.zeros((maxIngr))

            ingrs_encoding[0] = ingrs2num['noodle']
            # ingrs_encoding[1] = ingrs2num['']
            # ingrs_encoding[2] = ingrs2num['']
            # ingrs_encoding[3] = ingrs2num['']
            # ingrs_encoding[4] = ingrs2num['']

            ingrs_len = len(ingrs) if len(ingrs) < maxIngr else maxIngr
            ingrs_encoding = torch.LongTensor(ingrs_encoding)

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(1, nz))
        txt_embedding = prepare_data_with_text(data)
        noise.data.normal_(0, 1)
        fake_imgs, _,_ = model(noise, txt_embedding)

        cv.imwrite(out_dir, fake_imgs)
        print('*' * 30)
        print('Output Image generated at:', out_dir)
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
