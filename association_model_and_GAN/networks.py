import json
import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class TextEncoder(nn.Module):
    def __init__(
        self, data_dir, in_dim, hid_dim, z_dim):
        super(TextEncoder, self).__init__()      
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc2 = nn.Linear(hid_dim, z_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.bn(x)
        x = torch.tanh(self.fc2(x))
        return x


class Resnet(nn.Module):
    def __init__(self, ckpt_path=None):
        super(Resnet, self).__init__()
        resnet = models.resnet50(pretrained=False)
        num_feat = resnet.fc.in_features
        resnet.fc = nn.Linear(num_feat, 101)
        if ckpt_path:
            resnet.load_state_dict(clean_state_dict(torch.load(ckpt_path)))
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, image_list):
        BS = image_list.shape[0]
        return self.encoder(image_list).view(BS, -1)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, ckpt_path=None):
        super(ImageEncoder, self).__init__()
        self.resnet = Resnet(ckpt_path)
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, z_dim),
            nn.Tanh()
        )
    
    def forward(self, image_list):
        feat = self.resnet(image_list)
        feat = self.bottleneck(feat)
        # print('image', feat.shape)
        return feat