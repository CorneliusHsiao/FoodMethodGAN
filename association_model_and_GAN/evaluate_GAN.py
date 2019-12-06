import os
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils


from networks import TextEncoder, ImageEncoder
from args_GAN import args
from networks_GAN import G_NET
from utils_GAN import Dataset, compute_txt_feat


######################################################
#						preprocess
######################################################
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print('device:', device)
if device.__str__() == 'cpu':
	args.batch_size = 16

with open(os.path.join(args.data_dir, 'ingrs2numV2.json'), 'r') as f:
    ingrs_dict = json.load(f)
method_dict = {'baking': 0, 'frying':1, 'roasting':2, 'grilling':3,
                'simmering':4, 'broiling':5, 'poaching':6, 'steaming':7,
                'searing':8, 'stewing':9, 'braising':10, 'blanching':11}
for _ in method_dict.keys():
    method_dict[_] += len(ingrs_dict)
in_dim = len(ingrs_dict) + len(method_dict)

imsize = args.base_size * (2 ** (args.levels-1))
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=mean,
    std=std)
transform = transforms.Compose([
    transforms.Resize(imsize),
    transforms.RandomCrop(imsize)
])
norm = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

noise = torch.FloatTensor(32, args.z_dim).normal_(0, 0.1)

######################################################
#						dataset
######################################################
print(args.food_type, 'in' , 'test')

test_set = Dataset(
    part='test', 
    food_type=args.food_type,
    data_dir=args.data_dir, 
    img_dir=args.img_dir, 
    ingrs_dict=ingrs_dict,
    method_dict=method_dict,
    transform=transform,
    norm=norm)

test_loader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True)
print('test data:', len(test_set), len(test_loader))


######################################################
#						model
######################################################
TxtEnc = TextEncoder(
    data_dir=args.data_dir, in_dim=in_dim, hid_dim=1024, z_dim=1024).to(device)
ImgEnc = ImageEncoder(z_dim=1024, ckpt_path=None).to(device)
TxtEnc.eval()
ImgEnc.eval()
ImgEnc = nn.DataParallel(ImgEnc)

netG = G_NET(levels=args.levels)
netG.to(device)
netG.eval()
netG = torch.nn.DataParallel(netG)

assert args.resume != ''
if args.resume != '':
    print('=> load from checkpoint:', args.resume)
    ckpt = torch.load(args.resume)
    TxtEnc.load_state_dict(ckpt['TxtEnc'])
    ImgEnc.load_state_dict(ckpt['ImgEnc'])
    netG.load_state_dict(ckpt['netG'])


######################################################
#						evaluate
######################################################
num = 32
save_dir = './experiments'

ingre = set(['chocolate_bars', 'chocolate_chips', 'chocolate_curls', 'chocolate_shavings', 'dark_chocolate',
	'dark_chocolate_chips'])

print('Evaluating...')

rcps_pos, rcps_neg =[], []
for rcp in test_set.recipes:
	if ingre & set(rcp['ingredients']):
		rcps_pos.append(rcp)
	else:
		rcps_neg.append(rcp)
print('# w/o ', len(rcps_pos), len(rcps_neg))
IoU = np.zeros((len(rcps_pos), len(rcps_neg)), dtype = float)
for (i, rcp_p) in enumerate(rcps_pos):
	for (j, rcp_n) in enumerate(rcps_neg):
		IoU[i, j] = len(set(rcp_p['ingredients']) & set(rcp_n['ingredients'])) / len(set(rcp_p['ingredients']) | set(rcp_n['ingredients'])) 
txts_p_list, imgs_p_list, txts_n_list, imgs_n_list = [], [], [], []
for _ in range(num):
	i, j = np.where(IoU == IoU.max())
	i, j = i[0], j[0]
	print(IoU[i, j])
	IoU[i, j] = 0
	txt, img = test_set._prepare_recipe_data(rcps_pos[i])
	txts_p_list.append(txt)
	imgs_p_list.append(img[-1])
	txt, img = test_set._prepare_recipe_data(rcps_neg[j])
	txts_n_list.append(txt)
	imgs_n_list.append(img[-1])

print('Images Paired. Processing...')
txts_p = torch.stack(txts_p_list).to(device)
imgs_p = torch.stack(imgs_p_list).to(device)
txts_n = torch.stack(txts_n_list).to(device)
imgs_n = torch.stack(imgs_n_list).to(device)
with torch.no_grad():
	txts_p_embedding = compute_txt_feat(txts_p, TxtEnc)
	txts_n_embedding = compute_txt_feat(txts_n, TxtEnc)
imgs_p_fake, _, _ = netG(noise, txts_p_embedding)
imgs_n_fake, _, _ = netG(noise, txts_n_embedding)

print('All images done. Saving...')
imgs_p_fake = imgs_p_fake[-1]
imgs_n_fake = imgs_n_fake[-1]
images = torch.stack([imgs_p, imgs_p_fake, imgs_n_fake, imgs_n]).permute(1,0,2,3,4).contiguous()
images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
vutils.save_image(
	images,
	'{}/with_out.png'.format(save_dir),
	normalize=True, scale_each=True)
images = vutils.make_grid(images, normalize=True, scale_each=True)

"""
images = imgs_p_fake[-1].contiguous()
vutils.save_image(
	images,
	'./experiments/pos_fake.png',
	normalize=True, scale_each=True)
"""




