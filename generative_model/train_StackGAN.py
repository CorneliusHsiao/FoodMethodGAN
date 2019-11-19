import os
import json
from copy import deepcopy
import pdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from args_StackGAN import args
from dataset_StackGAN import Dataset
from datetime import datetime
from networks_StackGAN import INCEPTION_V3, G_NET, D_NET64, D_NET128, D_NET256
from utils_StackGAN import compute_inception_score, negative_log_posterior_probability
import sys
sys.path.append('../')
from utils import param_counter, make_saveDir, load_retrieval_model, move_recipe, rank, mean, std
import pprint
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(args.__dict__)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' \
    if torch.cuda.is_available() and args.cuda
    else 'cpu')
print('device:', device)
if device.__str__() == 'cpu':
    args.batch_size = 2


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

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

# define models
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir('../')
TxtEnc, ImgEnc = load_retrieval_model(args.retrieval_model, device)
TxtEnc.train()
ImgEnc.train()
os.chdir(dname)

netG = G_NET(levels=args.levels)
print('# params in G', param_counter(netG.parameters()))
netG.apply(weights_init)
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
    netsD[i] = torch.nn.DataParallel(netsD[i])

inception_model = INCEPTION_V3()

# define optimizers
optimizersD = []
num_Ds = len(netsD)
for i in range(num_Ds):
    opt = optim.Adam(netsD[i].parameters(),
                        lr=args.lr_d,
                        betas=(args.beta0, args.beta1))
    optimizersD.append(opt)

optimizer = torch.optim.Adam([
                {'params': TxtEnc.parameters()},
                {'params': ImgEnc.parameters()},
                {'params': netG.parameters()},
            ], lr=args.lr_g, betas=(args.beta0, args.beta1))

def compute_txt_feat(txt, TxtEnc):
    feat = TxtEnc(txt)
    return feat

def compute_img_feat(img, ImgEnc):
    img = img/2 + 0.5
    img = F.interpolate(img, [224, 224], mode='bilinear', align_corners=True)
    for i in range(img.shape[1]):
        img[:,i] = (img[:,i]-mean[i])/std[i]
    feat = ImgEnc(img)
    return feat

def compute_cycle_loss(feat1, feat2, paired=True):
    if paired:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, torch.ones(feat1.shape[0]).to(device))
    else:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, -torch.ones(feat1.shape[0]).to(device))
    return loss


netG.to(device)
for i in range(len(netsD)):
    netsD[i].to(device)
inception_model = inception_model.to(device)
inception_model.eval()

e_start = 0
e_end = args.epochs
niter = 0
# load from ckpt
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

# load dataset
imsize = args.base_size * (2 ** (args.levels-1))
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

train_set = Dataset(
    args.data_dir, args.img_dir, food_type=args.food_type, 
    levels=args.levels, part='train', 
    base_size=args.base_size, transform=image_transform, permute_ingrs=args.permute_ingrs)

if args.debug:
    print('=> in debug mode')
    train_set = torch.utils.data.Subset(train_set, range(100))
    args.save_interval = 1

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=int(args.workers))
print('=> train =', len(train_set), len(train_loader))


def prepare_data(data):
    imgs, w_imgs, txt, _ = data

    real_vimgs, wrong_vimgs = [], []
    for i in range(args.levels):
        real_vimgs.append(imgs[i].to(device))
        wrong_vimgs.append(w_imgs[i].to(device))
    vtxt = [x.to(device) for x in txt]
    return real_vimgs, wrong_vimgs, vtxt

food_type = args.food_type if args.food_type else 'all'
save_dir = make_saveDir('runs/{}_samples{}'.format(food_type, len(train_set)), args)
writer = SummaryWriter(log_dir=save_dir)

criterion = nn.BCELoss()

def compute_kl(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()


noise = torch.FloatTensor(args.batch_size, args.z_dim)
fixed_noise_part1 = torch.FloatTensor(1, args.z_dim).normal_(0, 1)
fixed_noise_part1 = fixed_noise_part1.repeat(32, 1)
fixed_noise_part2 = torch.FloatTensor(32, args.z_dim).normal_(0, 1)
fixed_noise = torch.cat([fixed_noise_part1, fixed_noise_part2], dim=0)
val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imsize)])
val_set = Dataset(
    args.data_dir, args.img_dir, food_type=args.food_type, 
    levels=args.levels, part='val', 
    base_size=args.base_size, transform=val_transform)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=64,
    drop_last=True, shuffle=False, num_workers=int(args.workers))
print('=> val =', len(val_set), len(val_loader))
fixed_batch = next(iter(val_loader))
fixed_real_imgs, _, fixed_txt = prepare_data(fixed_batch)

# avg_param_G = copy_G_params(netG)
# def load_params(model, new_param):
#     for p, new_p in zip(model.parameters(), new_param):
#         p.data.copy_(new_p)

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
    
    # real_img = real_imgs[-1][0:num]
    # vutils.save_image(
    #     real_img, 
    #     '{}/e{}_real_samples.png'.format(save_dir, epoch), 
    #     normalize=True)
    # real_img_set = vutils.make_grid(real_img, normalize=True)
    # writer.add_image('real_img', real_img_set, epoch)

    # for i in range(args.levels):
    #     fake_img = fake_imgs[i][0:num]
    #     vutils.save_image(
    #         fake_img, 
    #         '{}/e{}_fake_samples{}.png'.format(save_dir, epoch, i), 
    #         normalize=True, scale_each=True)
    #     fake_img_set = vutils.make_grid(fake_img, normalize=True, scale_each=True)
    #     writer.add_image('fake_img%d' % i, fake_img_set, epoch)

for epoch in range(e_start, e_end):
    print('-'*40)
    print('Epoch {}/{}'.format(epoch, e_end-1))

    # run val_set
    print('eval')
    txt_feats_real = []
    img_feats_real = []
    img_feats_fake = []
    labels_fake_np = []
    TxtEnc.eval()
    ImgEnc.eval()
    netG.eval()
    batch=0
    for data in tqdm(val_loader):
        real_imgs, _, txt = prepare_data(data)
        with torch.no_grad():
            txt_embedding = compute_txt_feat(txt, TxtEnc)
            fake_imgs, mu, logvar = netG(fixed_noise, txt_embedding)
        
        txt_feats_real.append(txt_embedding.detach().cpu())
        img_fake = fake_imgs[-1]
        img_embedding_fake = compute_img_feat(img_fake, ImgEnc)
        img_feats_fake.append(img_embedding_fake.detach().cpu())
        img_real = real_imgs[-1]
        img_embedding_real = compute_img_feat(img_real, ImgEnc)
        img_feats_real.append(img_embedding_real.detach().cpu())

        label_fake = inception_model(img_fake.detach())
        labels_fake_np.append(label_fake.cpu().numpy())
        if batch == 0 and (epoch % args.save_interval == 0 or epoch == e_end-1):
            print('saving model after epoch {}'.format(epoch))
            save_model(epoch)
            
            # backup_para = copy_G_params(netG)
            # load_params(netG, avg_param_G)
            writer.add_histogram('mu', mu, epoch)
            writer.add_histogram('std', (0.5*logvar).exp(), epoch)
            save_img_results(real_imgs, fake_imgs, epoch)
            # load_params(netG, backup_para)
        
        batch += 1

    txt_feats_real = torch.cat(txt_feats_real, dim=0)
    img_feats_real = torch.cat(img_feats_real, dim=0)
    img_feats_fake = torch.cat(img_feats_fake, dim=0)
    retrieved_range = min(900, len(val_loader)*args.batch_size)
    medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_real.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
    writer.add_scalar('real MedR', medR, epoch)
    print('=> [MedR] real: {:.4f}({:.4f})'.format(medR, medR_std))
    medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_fake.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
    writer.add_scalar('fake MedR', medR, epoch)
    print('=> [MedR] fake: {:.4f}({:.4f})'.format(medR, medR_std))

    labels_fake_np = np.concatenate(labels_fake_np, 0)
    mean_is, std_is = compute_inception_score(labels_fake_np, args.splits)
    writer.add_scalar('inception score', mean_is, epoch)

    print('train')
    TxtEnc.train()
    ImgEnc.train()
    netG.train()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_loss_kl = 0.0
    for data in tqdm(train_loader):

        if args.labels == 'original':
            real_labels = torch.FloatTensor(args.batch_size).fill_(
                1)  # (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float() #
            fake_labels = torch.FloatTensor(args.batch_size).fill_(
                0)  # (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() #
        elif args.labels == 'R-smooth':
            real_labels = torch.FloatTensor(args.batch_size).fill_(1) - (
                        torch.FloatTensor(args.batch_size).uniform_() * 0.1)
            fake_labels = (torch.FloatTensor(args.batch_size).uniform_() * 0.1)
        elif args.labels == 'R-flip':
            real_labels = (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float()  #
            fake_labels = (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float()  #
        elif args.labels == 'R-flip-smooth':
            real_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
                    torch.FloatTensor(args.batch_size).fill_(1) - (
                        torch.FloatTensor(args.batch_size).uniform_() * 0.1)))
            fake_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
                    torch.FloatTensor(args.batch_size).uniform_() * 0.1))

        if args.cuda:
            real_labels = real_labels.to(device)
            fake_labels = fake_labels.to(device)
            noise, fixed_noise = noise.to(device), fixed_noise.to(device)

        real_imgs, wrong_imgs, txt = prepare_data(data)
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
            errG_cycle_txt = compute_cycle_loss(feat_fake, txt_embedding)
            
            feat_real = compute_img_feat(real_imgs[level], ImgEnc)
            errG_cycle_img = compute_cycle_loss(feat_fake, feat_real)

            rightRcp_vs_rightImg = compute_cycle_loss(txt_embedding, feat_real)
            feat_real_wrong = compute_img_feat(wrong_imgs[level], ImgEnc)
            rightRcp_vs_wrongImg = compute_cycle_loss(txt_embedding, feat_real_wrong, paired=False)
            tri_loss = rightRcp_vs_rightImg + rightRcp_vs_wrongImg
            
            errG = errG_cond \
                + args.weight_uncond * errG_uncond \
                    + args.weight_cycle_txt * errG_cycle_txt \
                        + args.weight_cycle_img * errG_cycle_img \
                            + args.weight_tri_loss * tri_loss

            # record
            errG_total += errG
            if level == 2:
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
