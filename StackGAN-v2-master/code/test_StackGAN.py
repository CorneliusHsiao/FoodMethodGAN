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

from model import G_NET

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--b_condition', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--text_used', type=str, default="both", help="both or instruction")
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
    #print(checkpoint.keys())
    #model = checkpoint['model']
    #model.load_state_dict(checkpoint['state_dict'])
    #for parameter in model.parameters():
    #    parameter.requires_grad = False
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

def calculate_fid(real_images, fake_images):
    """
    real_images / fake_images: tensor
    shape: batch * 3 * h * w
    """ 
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    mu1, sigma1 = calculate_statistics(real_iamges)
    mu2, sigma2 = calculate_statistics(fake_images)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    #Product might be almost singular, we have to care numerical issue
    eps = 1e-6
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return  diff.dot(diff) + np.trace(sigma1 + sigma2 + 2 * covmean)

    def calculate_statistics(original_images):
        #images = F.interpolate(original_images, size=(299, 299), mode='bilinear', align_corners=True)
        #images = images.permute(0,2,3,1).numpy()
        images = preprocess_input(original_images)
        act = model.predict(images)
        mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)
        return mu, sigma

def calculate_inception_score(original_images, n_split=10, eps=1E-16):
    """
    original_images: tensor =now=> numpy array
    shape: batch * 3 * h * w =now=> batch (whole test) * 299 (h) * 299 (w) * 3 (channel)
    """ 
    model = InceptionV3()
    #images = F.interpolate(original_images, size=(299, 299), mode='bilinear', align_corners=True)
    #images = images.permute(0,2,3,1).numpy()
    #images = images.numpy()
    images = preprocess_input(original_images)
    yhat = model.predict(images)            # batch_size*299*299*3
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

def prepare_data(self, data):
    imgs = data
    vimgs = []
    for i in range(self.num_Ds):
        if cfg.CUDA:
            vimgs.append(Variable(imgs[i]).cuda())
        else:
            vimgs.append(Variable(imgs[i]))
    return imgs, vimgs

def prepare_data_with_text(self, data):
    imgs = data[0]
    w_imgs = data[1]

    instrs_emb = data[2]
    ingrs_emb = data[4]

    real_vimgs, wrong_vimgs = [], []
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

    for i in range(self.num_Ds):
        if cfg.CUDA:
            real_vimgs.append(Variable(imgs[i]).cuda())
            wrong_vimgs.append(Variable(w_imgs[i]).cuda())
        else:
            real_vimgs.append(Variable(imgs[i]))
            wrong_vimgs.append(Variable(w_imgs[i]))
    return imgs, real_vimgs, wrong_vimgs, [vembedding_instrs,v_instrs_len, vembedding_ingrs,v_ingrs_len]

class Evaluate_ingr():
    def test(model, dataloader):
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                # there are five things in data
                # index 0 contains a 64X64 real image and a 128 *128 real image
                real_imgs = data[0]
                # index 1 contains a 64x64 fake_image and a 128* 128 fake image 
                fake_image = data[1]
                # index 2 contain a set of index of instructions
                instructions = data[2]
                # index 3 contains a number represents the number of instructions
                instr_len = data[3]
                #index 4 contains a set of index of ingredients
                ingredients = data[4]
                #index 5 contian a number represents the number of ingredients
                ingr_len = data[5]

                # if it is img with text model
                nz = cfg.GAN.Z_DIM
                noise = Variable(torch.FloatTensor(2, nz))
                _, _, _, txt_embedding = prepare_data_with_text(data)
                noise.data.normal_(0, 1)
                fake_imgs, _,_ = model(noise, txt_embedding)


class Evaluate_ingr_method():
    def test(model, dataloader):
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                # there are five things in data
                # index 0 contains a 64X64 real image and a 128 *128 real image
                real_imgs = data[0]
                # index 1 contains a 64x64 fake_image and a 128* 128 fake image 
                fake_image = data[1]
                # index 2 contain a set of index of instructions
                instructions = data[2]
                # index 3 contains a number represents the number of instructions
                instr_len = data[3]
                #index 4 contains a set of index of ingredients
                ingredients = data[4]
                #index 5 contian a number represents the number of ingredients
                ingr_len = data[5]

                # if it is img with text model
                nz = cfg.GAN.Z_DIM
                noise = Variable(torch.FloatTensor(2, nz))
                _, _, _, txt_embedding = prepare_data_with_text(data)
                noise.data.normal_(0, 1)
                fake_imgs, _,_ = model(noise, txt_embedding)

class Evaluate_img():
    def test(model, dataloader):
        with torch.no_grad():
            real_imgs_batch = []
            fake_imgs_batch = []
            for i,data in enumerate(dataloader):
                # there are five things in data
                # index 0 contains a 64X64 real image and a 128 *128 real image
                real_imgs = data[0]
                # index 1 contains a 64x64 fake_image and a 128* 128 fake image 
                # fake_image = data[1]
                # index 2 contain a set of index of instructions
                # instructions = data[2]
                # index 3 contains a number represents the number of instructions
                # instr_len = data[3]
                #index 4 contains a set of index of ingredients
                # ingredients = data[4]
                #index 5 contian a number represents the number of ingredients
                # ingr_len = data[5]

                # if is img only model
                # shape of noise
                nz = cfg.GAN.Z_DIM # noise shape
                noise = Variable(torch.FloatTensor(2, nz))
                noise.data.normal_(0, 1)
                fake_imgs, _, _ = model(noise)

                for i in range(real_imgs.shape[0]):
                    real_imgs_batch.append(real_imgs[0, i].resize((299, 299, 3)))
                    fake_imgs_batch.append(fake_imgs[0, i].resize((299, 299, 3)))

            # convert to correct shape (whole test batch * 299 * 299 * 3)
            real_imgs_conv = np.array(real_imgs_batch)
            fake_imgs_conv = np.array(fake_imgs_batch)
            # compute inception score and FID
            test_real_is_avg, test_real_is_std = calculate_inception_score(real_imgs_conv, n_split=1, eps=1E-16)
            test_fake_is_avg, test_fake_is_std = calculate_inception_score(fake_imgs_conv, n_split=1, eps=1E-16)
            test_fid = calculate_fid(real_images_conv, fake_images_conv)

            print('*' * 30)
            print('Image input model test results:')
            print('\tAvg inception score of real images: {}'.format(test_real_is_avg))
            print('\tAvg inception score of fake images: {}'.format(test_fake_is_avg))
            print('\tFID of images: {}'.format(test_fid))
            print('*' * 30)

if __name__ == '__main__':
    args = parse_args()

    # Config 1: load default config
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Config 2: GPUs
    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    # Config 3: img -- 0, ingr -- 1, ingr+method -- 2
    if args.b_condition is not None:
        cfg.GAN.B_CONDITION = int(args.b_condition)

    # Config 4: data and dataloader
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.DATA_DIR.find("recipe") != -1:
        from dataloader import ImageLoader
        dataset = ImageLoader(
            img_path="../../data/img_data",
            transform = image_transform,
            data_path = "../../data",
            partition="test")
        print("using recipe1M dataset")
    else:
        print(cfg.DATA_DIR, " dataset not found")
        raise ValueError
    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, # cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    # Config 5: model dir
    if args.model_dir is not None:
        modelpath = args.model_dir
    else:
        raise ValueError('Must specify model to be tested.')

    # Main test
    start_t = time.time()
    model = load_checkpoint(modelpath)
    if cfg.GAN.B_CONDITION == 0:
        Evaluate_img.test(model, dataloader)
    elif cfg.GAN.B_CONDITION == 1:
        Evaluate_ingr.test(model, dataloader)
    elif cfg.GAN.B_CONDITION == 2:
        Evaluate_ingr_method.test(model, dataloader)
    end_t = time.time()
    print('Total time for testing:', end_t - start_t)
    
