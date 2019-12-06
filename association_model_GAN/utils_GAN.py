import os
import sys
import json
from os.path import join
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image

from args_GAN import args
from utils import load_recipes

def get_imgs(img_path, imsize, levels=2, transform=None, norm=None):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    ret = []
    for i in range(levels):
        if i < levels-1:
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(norm(re_img))
    return ret

class Dataset(data.Dataset):
    def __init__(
        self, data_dir, img_dir, food_type=None, levels=2, part='train', 
        base_size=64, ingrs_dict=None, method_dict=None, transform=None, norm=None):
        self.imsize = [64, 128]     #levels = 2
        self.transform = transform
        self.norm = norm
        self.img_dir = img_dir
        self.levels = levels
        self.ingrs_dict = ingrs_dict
        self.method_dict = method_dict
        self.recipes = load_recipes(join(data_dir, 'recipesV1.json'), part)
        if food_type:
            self.recipes = [x for x in self.recipes if food_type in x['title'].lower() ]

    def _choose_one_image(self, rcp):
        part = rcp['partition']
        local_paths = rcp['images']
        local_path = np.random.choice(local_paths)
        img_path = join(self.img_dir, part, local_path)
        imgs = get_imgs(
            img_path, 
            imsize=self.imsize,
            levels=self.levels, 
            transform=self.transform,
            norm = self.norm)
        return imgs

    def _prepare_recipe_data(self, rcp):
        txt = torch.zeros(len(self.ingrs_dict) + len(self.method_dict), dtype=torch.float)
        for ingr in rcp['ingredients']:
            txt[self.ingrs_dict[ingr]] = 1
        for method in rcp['instructions']:
            txt[self.method_dict[method]] = 1
        img = self._choose_one_image(rcp)
        return txt, img

    def __getitem__(self, index):
        rcp_a = self.recipes[index]       
        rcp_b = self.recipes[np.random.choice(len(self.recipes))]

        txt_a, imgs_a = self._prepare_recipe_data(rcp_a)
        _, imgs_b = self._prepare_recipe_data(rcp_b)
        return imgs_a, imgs_b, txt_a, rcp_a['id']

    def __len__(self):
        return len(self.recipes)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def compute_txt_feat(txt, TxtEnc):
    feat = TxtEnc(txt)
    return feat


def compute_img_feat(img, ImgEnc):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = F.interpolate(img, [224, 224], mode='bilinear', align_corners=True)
    for i in range(img.shape[1]):
        img[:,i] = (img[:,i]-mean[i])/std[i]
    feat = ImgEnc(img)
    return feat


def compute_cycle_loss(feat1, feat2, device, paired=True):
    if paired:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, torch.ones(feat1.shape[0]).to(device))
    else:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, -torch.ones(feat1.shape[0]).to(device))
    return loss


def prepare_data(data, device):
    imgs, w_imgs, txt, _ = data

    real_vimgs, wrong_vimgs = [], []
    for i in range(args.levels):
        real_vimgs.append(imgs[i].to(device))
        wrong_vimgs.append(w_imgs[i].to(device))
    vtxt = txt.to(device)
    return real_vimgs, wrong_vimgs, vtxt


def compute_kl(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)
