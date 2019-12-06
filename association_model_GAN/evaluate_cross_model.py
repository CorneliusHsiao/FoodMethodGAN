import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import args
from networks import TextEncoder, ImageEncoder
from utils import transform, Dataset, rank

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


######################################################
#						dataset
######################################################
test_set = Dataset(
    part='test', 
    data_dir=args.data_dir, 
    img_dir=args.img_dir, 
    ingrs_dict=ingrs_dict,
    method_dict=method_dict,
    transform=transform)

test_loader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
print('test data:', len(test_set), len(test_loader))


######################################################
#						model
######################################################
TxtEnc = TextEncoder(
    data_dir=args.data_dir, in_dim=in_dim, hid_dim=args.hid_dim, z_dim=args.z_dim).to(device)
ImgEnc = ImageEncoder(z_dim=args.z_dim, ckpt_path=args.upmc_model).to(device)
TxtEnc.eval()
ImgEnc.eval()
ImgEnc = nn.DataParallel(ImgEnc)

assert args.resume != ''
print('load from ckpt: ', args.resume)
ckpt = torch.load(args.resume)
TxtEnc.load_state_dict(ckpt['weights_recipe'])
ImgEnc.load_state_dict(ckpt['weights_image'])


######################### evaluate ########################
print('Evaluating...')
imgs = []
rcps = []
for batch in tqdm(test_loader):
    recipe = batch
    recipe[0], recipe[1] = recipe[0].to(device), recipe[1].to(device)
    with torch.no_grad():
        txts_sub = TxtEnc(recipe[0])
        imgs_sub = ImgEnc(recipe[1])
    rcps.append(txts_sub.detach().cpu().numpy())
    imgs.append(imgs_sub.detach().cpu().numpy())
rcps = np.concatenate(rcps, axis=0)
imgs = np.concatenate(imgs, axis=0)

for retrieved_type in ['recipe', 'image']:
    for retrieved_range in [1000, 5000, 10000]:
        print(retrieved_type, retrieved_range)
        print('=> computing ranks...')
        medR, medR_std, recalls = rank(rcps, imgs, retrieved_type, retrieved_range)
        print('=> val MedR: {:.4f}({:.4f})'.format(medR, medR_std))
        for k,v in recalls.items():
            print('Recall@{}'.format(k), v)


"""
recipe 1000
=> computing ranks...
=> val MedR: 4.4000(0.4899)
Recall@1 0.26080000000000003
Recall@5 0.5485000000000001
Recall@10 0.6794
recipe 5000
=> computing ranks...
=> val MedR: 17.9000(0.5385)
Recall@1 0.11638
Recall@5 0.29918
Recall@10 0.40630000000000005
recipe 10000
=> computing ranks...
=> val MedR: 34.9000(0.8307)
Recall@1 0.07719000000000001
Recall@5 0.21172999999999997
Recall@10 0.3009799999999999
image 1000
=> computing ranks...
=> val MedR: 4.2000(0.4000)
Recall@1 0.27019999999999994
Recall@5 0.5561999999999999
Recall@10 0.6819000000000001
image 5000
=> computing ranks...
=> val MedR: 16.7000(0.7810)
Recall@1 0.12888
Recall@5 0.3148
Recall@10 0.4207199999999999
image 10000
=> computing ranks...
=> val MedR: 32.7000(1.1874)
Recall@1 0.08757
Recall@5 0.22885
Recall@10 0.31910000000000005

"""