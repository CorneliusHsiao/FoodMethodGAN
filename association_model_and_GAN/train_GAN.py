import os
import sys
import json
from copy import deepcopy
import pdb
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch import optim

from args_GAN import args
from networks_GAN import INCEPTION_V3, G_NET, D_NET64, D_NET128, D_NET256
from utils_GAN import Dataset, weights_init, compute_txt_feat, compute_cycle_loss, prepare_data, compute_kl, compute_img_feat
from utils_GAN import compute_inception_score, negative_log_posterior_probability
from utils import param_counter, make_saveDir, load_retrieval_model, rank, mean, std
import pprint


######################### preprocess ########################
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(args.__dict__)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' \
    if torch.cuda.is_available() and args.cuda
    else 'cpu')
print('Device:', device)
if device.__str__() == 'cpu':
    args.batch_size = 2

with open(os.path.join(args.data_dir, 'ingrs2numV2.json'), 'r') as f:
    ingrs_dict = json.load(f)
method_dict = {'baking': 0, 'frying':1, 'roasting':2, 'grilling':3,
                'simmering':4, 'broiling':5, 'poaching':6, 'steaming':7,
                'searing':8, 'stewing':9, 'braising':10, 'blanching':11}
for _ in method_dict.keys():
    method_dict[_] += len(ingrs_dict)
in_dim = len(ingrs_dict) + len(method_dict)


######################### model and optimizer ########################
TxtEnc, ImgEnc = load_retrieval_model(args.retrieval_model, in_dim, device)

netG = G_NET(levels=args.levels)
print('# params in G', param_counter(netG.parameters()))
netG.apply(weights_init)
netG.to(device)
netG = torch.nn.DataParallel(netG)

netsD = []
if args.levels > 0:
    netsD.append(D_NET64(bi_condition=args.bi_condition))
if args.levels > 1:
    netsD.append(D_NET128(bi_condition=args.bi_condition))
if args.levels > 2:
    netsD.append(D_NET256(bi_condition=args.bi_condition))
for i in range(len(netsD)):
    print('# params in D_{} = {}'.format(i, param_counter(netsD[i].parameters())))
    netsD[i].apply(weights_init)
    netsD[i].train()
    netsD[i].to(device)
    netsD[i] = torch.nn.DataParallel(netsD[i])

optimizersD = []
num_Ds = len(netsD)
for i in range(num_Ds):
    opt = optim.Adam(netsD[i].parameters(),
                        lr=args.lr_d,
                        betas=(args.beta0, args.beta1))
    optimizersD.append(opt)

optimizer = torch.optim.Adam([{'params': netG.parameters()}], lr=args.lr_g, betas=(args.beta0, args.beta1))


######################### dataset ########################
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


train_set = Dataset(
    args.data_dir, args.img_dir, food_type=args.food_type, levels=args.levels, part='train', 
    base_size=args.base_size, ingrs_dict=ingrs_dict, method_dict=method_dict, transform=transform, norm=norm)
if args.debug:
    print('=> In debug mode')
    train_set = torch.utils.data.Subset(train_set, range(100))
    args.save_interval = 1
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=args.workers)
print('=> train =', len(train_set), len(train_loader))


val_set = Dataset(
    args.data_dir, args.img_dir, food_type=args.food_type, levels=args.levels, part='val', 
    base_size=args.base_size, ingrs_dict=ingrs_dict, method_dict=method_dict, transform=transform, norm=norm)
if args.debug:
    val_set = torch.utils.data.Subset(val_set, range(100))
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=64,
    drop_last=True, shuffle=False, num_workers=args.workers)
print('=> val =', len(val_set), len(val_loader))

fixed_batch = next(iter(val_loader))
fixed_real_imgs, _, fixed_txt = prepare_data(fixed_batch, device)

noise = torch.FloatTensor(args.batch_size, args.z_dim)
fixed_noise_part1 = torch.FloatTensor(1, args.z_dim).normal_(0, 1)
fixed_noise_part1 = fixed_noise_part1.repeat(32, 1)
fixed_noise_part2 = torch.FloatTensor(32, args.z_dim).normal_(0, 1)
fixed_noise = torch.cat([fixed_noise_part1, fixed_noise_part2], dim=0)

######################### train ########################
save_dir = make_saveDir('runs/{}_samples{}'.format(args.food_type, len(train_set)), args)
writer = SummaryWriter(log_dir=save_dir)

e_start = 0
e_end = args.epochs
niter = 0
criterion = nn.BCELoss()
if args.resume != '':
    print('=> load from checkpoint:', args.resume)
    ckpt = torch.load(args.resume)
    TxtEnc.load_state_dict(ckpt['TxtEnc'])
    ImgEnc.load_state_dict(ckpt['ImgEnc'])
    netG.load_state_dict(ckpt['netG'])
    optimizer.load_state_dict(ckpt['optimizer'])
    for i in range(len(netsD)):
        netsD[i].load_state_dict(ckpt['netD_{}'.format(i)])
        optimizersD[i].load_state_dict(ckpt['optimizerD_{}'.format(i)])
    e_start = ckpt['epoch'] + 1
    e_end = e_start + args.epochs
    niter = ckpt['niter'] + 1



def save_model(epoch):
    # load_params(netG, avg_param_G)
    ckpt = {}
    ckpt['epoch'] = epoch
    ckpt['niter'] = niter - 1
    ckpt['TxtEnc'] = TxtEnc.state_dict()
    ckpt['ImgEnc'] = ImgEnc.state_dict()
    ckpt['netG'] = netG.state_dict()
    ckpt['optimizer'] = optimizer.state_dict()
    for i in range(len(netsD)):
        netD = netsD[i]
        optimizerD = optimizersD[i]
        ckpt['netD_{}'.format(i)] = netD.state_dict()
        ckpt['optimizerD_{}'.format(i)] = optimizerD.state_dict()
    filepath = os.path.join(save_dir, 'e{}.ckpt'.format(epoch))
    print('ckpt path:', filepath)
    torch.save(ckpt, filepath)

def save_img_results(real_imgs, fake_imgs, epoch):
    num = 64
    real_img = real_imgs[-1][0:num]
    fake_img = fake_imgs[-1][0:num]
    real_fake = torch.stack([real_img, fake_img]).permute(1,0,2,3,4).contiguous()
    real_fake = real_fake.view(-1, real_fake.shape[-3], real_fake.shape[-2], real_fake.shape[-1])
    vutils.save_image(
            real_fake, 
            '{}/e{}_real_fake.png'.format(save_dir, epoch),  
            normalize=True, scale_each=True)
    real_fake = vutils.make_grid(real_fake, normalize=True, scale_each=True)
    writer.add_image('real_fake', real_fake, epoch)


for epoch in range(e_start, e_end):
    print('-'*40)
    print('Epoch {}/{}'.format(epoch, e_end-1))

    # run val_set
    print('eval')
    TxtEnc.eval()
    ImgEnc.eval()
    netG.eval()
    batch=0
    for data in tqdm(val_loader):
        real_imgs, _, txt = prepare_data(data, device)
        with torch.no_grad():
            txt_embedding = compute_txt_feat(txt, TxtEnc)
            fake_imgs, mu, logvar = netG(fixed_noise, txt_embedding)

        if batch == 0 and (epoch % args.save_interval == 0 or epoch == e_end-1):
            print('saving model after epoch {}'.format(epoch))
            save_model(epoch)
            writer.add_histogram('mu', mu, epoch)
            writer.add_histogram('std', (0.5*logvar).exp(), epoch)
            save_img_results(real_imgs, fake_imgs, epoch)
            # load_params(netG, backup_para)
        batch += 1



    print('train')
    netG.train()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_loss_kl = 0.0
    for data in tqdm(train_loader):

        real_labels = torch.FloatTensor(args.batch_size).fill_(1)  # (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float() #
        fake_labels = torch.FloatTensor(args.batch_size).fill_(0)  # (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() #

        real_labels = real_labels.to(device)
        fake_labels = fake_labels.to(device)
        noise, fixed_noise = noise.to(device), fixed_noise.to(device)

        real_imgs, wrong_imgs, txt = prepare_data(data, device)
        txt_embedding = compute_txt_feat(txt, TxtEnc)
        #######################################################
        # (1) Generate fake images
        ######################################################
        noise.normal_(0, 1)
        fake_imgs, mu, logvar = netG(noise, txt_embedding)

        #######################################################
        # (2) Update D network
        ######################################################
        errD_total = 0
        for level in range(args.levels):
            if args.input_noise:
                sigma = np.clip(1.0 - epoch/200, 0, 1) * 0.1
                real_img_noise = torch.empty_like(real_imgs[level]).normal_(0, sigma)
                wrong_img_noise = torch.empty_like(wrong_imgs[level]).normal_(0, sigma)
                fake_img_noise = torch.empty_like(fake_imgs[level]).normal_(0, sigma)
            else:
                real_img_noise = torch.zeros_like(real_imgs[level])
                wrong_img_noise = torch.zeros_like(wrong_imgs[level])
                fake_img_noise = torch.zeros_like(fake_imgs[level])


            #forward
            netD = netsD[level]
            optD = optimizersD[level]
            
            real_logits = netD(real_imgs[level]+real_img_noise, mu.detach())
            wrong_logits = netD(wrong_imgs[level]+wrong_img_noise, mu.detach())
            fake_logits = netD(fake_imgs[level].detach()+fake_img_noise, mu.detach())

            errD_real = criterion(real_logits[0], real_labels) # cond_real --> 1
            errD_wrong = criterion(wrong_logits[0], fake_labels) # cond_wrong --> 0
            errD_fake = criterion(fake_logits[0], fake_labels) # cond_fake --> 0
            
            if len(real_logits)>1:
                errD_cond = errD_real + errD_wrong + errD_fake
                errD_real_uncond = criterion(real_logits[1], real_labels) # uncond_real --> 1
                errD_wrong_uncond = criterion(wrong_logits[1], real_labels) # uncond_wrong --> 1
                errD_fake_uncond = criterion(fake_logits[1], fake_labels) # uncond_fake --> 0
                errD_uncond = errD_real_uncond + errD_wrong_uncond + errD_fake_uncond
            else: # back to GAN-INT-CLS
                errD_cond = errD_real + 0.5 * (errD_wrong + errD_fake)
                errD_uncond = 0.0
            
            errD = errD_cond + args.weight_uncond * errD_uncond
            
            optD.zero_grad()
            errD.backward()
            optD.step()

            # record
            errD_total += errD
            if level == 2:
                writer.add_scalar('D_loss_cond{}'.format(level), errD_cond, niter)
                writer.add_scalar('D_loss_uncond{}'.format(level), errD_uncond, niter)
                writer.add_scalar('D_loss{}'.format(level), errD, niter)

        #######################################################
        # (3) Update G network: maximize log(D(G(z)))
        ######################################################
        # forward
        errG_total = 0.0
        # txt_embedding = compute_txt_feat(txt, TxtEnc)
        # fake_imgs, mu, logvar = netG(noise, txt_embedding)
        for level in range(args.levels):
            if args.input_noise:
                sigma = np.clip(1.0 - epoch/200, 0, 1) * 0.1
                fake_img_noise = torch.empty_like(fake_imgs[level]).normal_(0, sigma)
            else:
                fake_img_noise = torch.zeros_like(fake_imgs[level])

            outputs = netsD[level](fake_imgs[level] + fake_img_noise, mu)
            errG_cond = criterion(outputs[0], real_labels) # cond_fake --> 1
            
            errG_uncond = criterion(outputs[1], real_labels) # uncond_fake --> 1

            feat_fake = compute_img_feat(fake_imgs[level], ImgEnc)
            errG_cycle_txt = compute_cycle_loss(feat_fake, txt_embedding, device)
            
            feat_real = compute_img_feat(real_imgs[level], ImgEnc)
            errG_cycle_img = compute_cycle_loss(feat_fake, feat_real, device)

            rightRcp_vs_rightImg = compute_cycle_loss(txt_embedding, feat_real, device)
            feat_real_wrong = compute_img_feat(wrong_imgs[level], ImgEnc)
            rightRcp_vs_wrongImg = compute_cycle_loss(txt_embedding, feat_real_wrong, device, paired=False)
            tri_loss = rightRcp_vs_rightImg + rightRcp_vs_wrongImg
            
            errG = errG_cond \
                + args.weight_uncond * errG_uncond \
                    + args.weight_cycle_txt * errG_cycle_txt \
                        + args.weight_cycle_img * errG_cycle_img \
                            + args.weight_tri_loss * tri_loss

            # record
            errG_total += errG
            if level == args.levels:
                writer.add_scalar('G_tri_loss{}'.format(level), tri_loss, niter)
                writer.add_scalar('G_loss_cond{}'.format(level), errG_cond, niter)
                writer.add_scalar('G_loss_uncond{}'.format(level), errG_uncond, niter)
                writer.add_scalar('G_loss_cycle_txt{}'.format(level), errG_cycle_txt, niter)
                writer.add_scalar('G_loss_cycle_img{}'.format(level), errG_cycle_img, niter)
                writer.add_scalar('G_loss{}'.format(level), errG, niter)
        
        errG_kl = compute_kl(mu, logvar)
        writer.add_scalar('G_kl', errG_kl, niter)
        errG_total += args.weight_kl * errG_kl

        optimizer.zero_grad()
        errG_total.backward()
        optimizer.step()
            
        # record
        writer.add_scalar('D_loss', errD_total, niter)
        writer.add_scalar('G_loss', errG_total, niter)
        epoch_loss_D += errD_total
        epoch_loss_G += errG_total
        epoch_loss_kl += errG_kl

        # for p, avg_p in zip(netG.parameters(), avg_param_G):
        #     avg_p.mul_(0.999).add_(0.001, p.data)

        # level = 2
        # feat_real = compute_img_feat(real_imgs[level], ImgEnc)
        # rightRcp_vs_rightImg = compute_cycle_loss(txt_embedding, feat_real)
        # feat_real_wrong = compute_img_feat(wrong_imgs[level], ImgEnc)
        # rightRcp_vs_wrongImg = compute_cycle_loss(txt_embedding, feat_real_wrong, paired=False)
        # tri_loss = rightRcp_vs_rightImg + rightRcp_vs_wrongImg
        # writer.add_scalar('G_tri_loss{}'.format(level), tri_loss, niter)
        # optimizer.zero_grad()
        # tri_loss.backward()
        # optimizer.step()
        niter += 1
    
    # end of the epoch
    epoch_loss_D /= len(train_loader)
    epoch_loss_G /= len(train_loader)
    epoch_loss_kl /= len(train_loader)
    print('Loss_D={:.4f}, Loss_G={:.4f}, Loss_kl={:.4f}'.format(
        epoch_loss_D, epoch_loss_G, epoch_loss_kl
    ))
    # record
    writer.add_scalar('epoch_loss_D', epoch_loss_D, epoch)
    writer.add_scalar('epoch_loss_G', epoch_loss_G, epoch)
    writer.add_scalar('epoch_loss_kl', epoch_loss_kl, epoch)
