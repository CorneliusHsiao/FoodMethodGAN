import os
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from args import args
from utils import Dataset, transform, rank, param_counter, make_saveDir, compute_loss
from networks import TextEncoder, ImageEncoder


######################### dataset ########################

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print('device:', device)
if device.__str__() == 'cpu':
    args.batch_size = 16
print(args)

with open(os.path.join(args.data_dir, 'ingrs2numV2.json'), 'r') as f:
    ingrs_dict = json.load(f)
method_dict = {'baking': 0, 'frying':1, 'roasting':2, 'grilling':3,
                'simmering':4, 'broiling':5, 'poaching':6, 'steaming':7,
                'searing':8, 'stewing':9, 'braising':10, 'blanching':11}
for _ in method_dict.keys():
    method_dict[_] += len(ingrs_dict)
in_dim = len(ingrs_dict) + len(method_dict)

######################### preprocess ########################

train_set = Dataset(
    part='train',
    data_dir=args.data_dir,
    img_dir=args.img_dir, 
    ingrs_dict=ingrs_dict,
    method_dict=method_dict,
    transform=transform)

if args.debug:
    print('in debug mode')
    train_set = torch.utils.data.Subset(train_set, range(20000))

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
print('train data:', len(train_set), len(train_loader))

val_set = Dataset(
    part='val', 
    data_dir=args.data_dir, 
    img_dir=args.img_dir, 
    ingrs_dict=ingrs_dict,
    method_dict=method_dict,
    transform=transform)

if args.debug:
    val_set = torch.utils.data.Subset(val_set, range(5000))

val_loader = DataLoader(
    val_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
print('val data:', len(val_set), len(val_loader))


######################### model ########################

TxtEnc = TextEncoder(
    data_dir=args.data_dir, in_dim=in_dim, hid_dim=args.hid_dim, z_dim=args.z_dim).to(device)
ImgEnc = ImageEncoder(z_dim=args.z_dim, ckpt_path=args.upmc_model).to(device)
ImgEnc = nn.DataParallel(ImgEnc)
print('# params_text', param_counter(TxtEnc.parameters()))
print('# params_image', param_counter(ImgEnc.parameters()))


######################### train ########################
optimizer = optim.Adam([
    {'params': TxtEnc.parameters()}, 
    {'params': ImgEnc.parameters()} ],
    lr=args.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

title = 'runs/samples{}'.format(len(train_set))
save_dir = make_saveDir(title, args)
writer = SummaryWriter(save_dir)

epoch_start = 0
epoch_end = args.epochs
niter = 0
if args.resume:
    print('load from ckpt: ', args.resume)
    ckpt = torch.load(args.resume)
    TxtEnc.load_state_dict(ckpt['weights_recipe'])
    ImgEnc.load_state_dict(ckpt['weights_image'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch_start = ckpt['epoch'] + 1
    epoch_end = epoch_start + args.epochs
    niter = ckpt['niter_train'] + 1

def _train(epoch):
    global niter
    print('=> train')
    TxtEnc.train()
    ImgEnc.train()
    loss_epoch = 0.0
    for batch in tqdm(train_loader):
        recipe = batch
        recipe[0], recipe[1] = recipe[0].to(device), recipe[1].to(device)
        txt = TxtEnc(recipe[0])
        img = ImgEnc(recipe[1])
        loss = compute_loss(txt, img, device)
        optimizer.zero_grad()
        loss.backward()
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], args.grad_clip)
        optimizer.step()

        writer.add_scalar('loss_batch_train', loss.item(), niter)
        loss_epoch += loss.item() * recipe[1].shape[0]
        niter += 1
    loss_epoch /= len(train_set)
    writer.add_scalar('loss_epoch_train', loss_epoch, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

def _val(epoch):
    print('=> val')
    TxtEnc.eval()
    ImgEnc.eval()
    loss_epoch = 0.0
    imgs = []
    rcps = []
    for batch in tqdm(val_loader):
        recipe = batch
        recipe[0], recipe[1] = recipe[0].to(device), recipe[1].to(device)
        with torch.no_grad():
            txts_sub = TxtEnc(recipe[0])
            imgs_sub = ImgEnc(recipe[1])
            loss = compute_loss(txts_sub, imgs_sub, device)
            loss_epoch += loss.item() * recipe[1].shape[0]
        rcps.append(txts_sub.detach().cpu().numpy())
        imgs.append(imgs_sub.detach().cpu().numpy())
    rcps = np.concatenate(rcps, axis=0)
    imgs = np.concatenate(imgs, axis=0)
    print('=> computing ranks...')
    medR, medR_std, recalls = rank(rcps, imgs, args.retrieved_type, args.retrieved_range)
    print('=> val MedR: {:.4f}({:.4f})'.format(medR, medR_std))
    writer.add_scalar('medR', medR, epoch)
    writer.add_scalar('medR_std', medR_std, epoch)
    for k,v in recalls.items():
        writer.add_scalar('Recall@{}'.format(k), v, epoch)
    loss_epoch /= len(val_set)
    writer.add_scalar('loss_epoch_val', loss_epoch, epoch)
    scheduler.step(loss_epoch)

def _save(epoch):
    global niter
    print('save checkpoint')
    ckpt = {
        'weights_recipe': TxtEnc.state_dict(),
        'weights_image': ImgEnc.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'niter_train': niter-1,
    }
    print(os.path.join(save_dir, 'e{}.ckpt'.format(epoch)))
    torch.save(
        ckpt, 
        os.path.join(save_dir, 'e{}.ckpt'.format(epoch)))

for epoch in range(epoch_start, epoch_end):
    print('-' * 40)
    print('=> Epoch: {}/{}'.format(epoch, epoch_end-1))
    _train(epoch)
    if epoch % args.val_freq == 0:
        _val(epoch)
    if (epoch+1) % args.save_freq == 0 or epoch+1 == args.epochs:
        _save(epoch)