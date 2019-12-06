import os
import string
from tqdm import tqdm
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from torchvision import transforms
import torch
import re
import copy
from datetime import datetime
import json
from PIL import Image
from networks import TextEncoder, ImageEncoder
#from generative_model.networks_StackGAN import G_NET
from types import SimpleNamespace
from torch import nn
from torch.utils import data
import math
from torchvision.utils import save_image, make_grid
import argparse
from args import args


class Dataset(data.Dataset):
    def __init__(self, part, data_dir, img_dir, ingrs_dict, method_dict, transform=None):
        assert part in ('train', 'val', 'test'), \
            'part must be one of [train, val, test]'
        self.recipes = load_recipes(os.path.join(data_dir,'recipesV1.json'), part)
        self.ingrs_dict = ingrs_dict
        self.method_dict = method_dict
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transform
    
    def __getitem__(self, index):
        rcp = self.recipes[index]
        txt = torch.zeros(len(self.ingrs_dict) + len(self.method_dict), dtype=torch.float)
        for ingr in rcp['ingredients']:
            txt[self.ingrs_dict[ingr]] = 1
        for method in rcp['instructions']:
            txt[self.method_dict[method]] = 1
        img = choose_one_image(rcp, self.img_dir, self.transform)
        return txt, img
    
    def __len__(self):
        return len(self.recipes)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=mean,
    std=std)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])
param_counter = lambda params: sum(p.numel() for p in params if p.requires_grad)


def make_saveDir(title, args=None):
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = '{}_{}'.format(title, timestamp)
    print('=> save_dir:', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args:
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    return save_dir


def compute_loss(txt, img, device):
    BS = txt.shape[0]
    denom = img.norm(p=2, dim=1, keepdim=True) @ txt.norm(p=2, dim=1, keepdim=True).t()
    numer = img @ txt.t()
    sim = numer / (denom+1e-8)
    cor_sim = (torch.diag(sim) * torch.ones(BS, BS).to(device)).t()
    loss_retrieve_txt = torch.max(
        torch.tensor(0.0).to(device), 
        args.margin + sim - cor_sim)  
    loss_retrieve_img = torch.max(
        torch.tensor(0.0).to(device), 
        args.margin + sim.t() - cor_sim)    
    loss = loss_retrieve_img + loss_retrieve_txt
    loss = loss.sum() / loss.nonzero().shape[0]
    return loss


def rank(rcps, imgs, retrieved_type='recipe', retrieved_range=1000):
    N = retrieved_range
    data_size = imgs.shape[0]
    idxs = range(N)
    glob_rank = []
    glob_recall = {1:0.0, 5:0.0, 10:0.0}
    # average over 10 sets
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        imgs_sub = imgs_sub / np.linalg.norm(imgs_sub, axis=1)[:, None]
        rcps_sub = rcps_sub / np.linalg.norm(rcps_sub, axis=1)[:, None]
        if retrieved_type == 'recipe':
            sims = np.dot(imgs_sub, rcps_sub.T) # [N, N]
        else:
            sims = np.dot(rcps_sub, imgs_sub.T)
        med_rank = []
        recall = {1:0.0, 5:0.0, 10:0.0}
        # loop through the N similarities for images
        for ii in idxs:
            # get a column of similarities for image ii
            sim = sims[ii,:]
            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()
            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)
            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1
            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i] = recall[i]/N
        med = np.median(med_rank)
        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)
    
    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    return np.mean(glob_rank), np.std(glob_rank), glob_recall


def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)


class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        print(len(base))
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base



def _find_args(filepath):
    prefix = filepath.rsplit('.', 1)[0]
    args_file = prefix + '.json'
    assert os.path.exists(args_file), '{} is not found'.format(args_file)
    with open(args_file, 'r') as f:
        args = json.load(f)
    args = SimpleNamespace(**args)
    return args

def load_retrieval_model(filepath, in_dim, device):
    args = _find_args(filepath)
    TxtEnc = TextEncoder(
        data_dir=args.data_dir, in_dim=in_dim, hid_dim=args.hid_dim, z_dim=args.z_dim)
    ImgEnc = ImageEncoder(z_dim=args.z_dim)
    ImgEnc = torch.nn.DataParallel(ImgEnc)
    TxtEnc.eval()
    ImgEnc.eval()
    print('Load from:', filepath)
    ckpt = torch.load(filepath)
    TxtEnc.load_state_dict(ckpt['weights_recipe'])
    ImgEnc.load_state_dict(ckpt['weights_image'])
    return TxtEnc.to(device), ImgEnc.to(device)

def load_generation_model(filepath, device):
    args = _find_args(filepath)
    netG = G_NET(levels=args.levels).eval()
    netG = nn.DataParallel(netG)
    print('load from:', filepath)
    ckpt = torch.load(filepath)
    netG.load_state_dict(ckpt['netG'])
    return netG.to(device)

def compute_img_feature(rcps, img_dir, imgM, transform, device):
    imgs = []
    for rcp in rcps:
        img = choose_one_image(rcp, img_dir, transform=transform)
        imgs.append(img)
    imgs = torch.stack(imgs)
    feats = []
    batch_size = 128
    num_batches = math.ceil(1.0*imgs.shape[0]/batch_size)
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            feat = imgM(imgs[i*batch_size:(i+1)*batch_size].to(device))
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    return imgs, feats

def extract_vector(rcp, vocab_inst, vocab_ingr):
    title = get_title_wordvec(rcp, vocab_inst) # np.int [max_len]
    instructions = get_instructions_wordvec(rcp, vocab_inst) # np.int [max_len, max_len]
    ingredients = get_ingredients_wordvec(rcp, vocab_ingr) # np.int [max_len]
    return title, ingredients, instructions

def compute_txt_feature(rcps, txtM, vocab_inst, vocab_ingr, device):
    title_list = []
    ingredients_list = []
    instructions_list =[]
    for rcp in rcps:
        title, ingredients, instructions = extract_vector(rcp, vocab_inst, vocab_ingr)
        title_list.append(title)
        ingredients_list.append(ingredients)
        instructions_list.append(instructions)
    title_tensor = torch.tensor(np.stack(title_list)).to(device) # [N, 20]
    ingredients_tensor = torch.tensor(np.stack(ingredients_list)).to(device) # [N, 20]
    instructions_tensor = torch.tensor(np.stack(instructions_list)).to(device) # [N, 20, 20]
    feats = []
    batch_size = 128
    num_batches = math.ceil(1.0*title_tensor.shape[0]/batch_size)
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            feat = txtM(
                (title_tensor[i*batch_size:(i+1)*batch_size].to(device), 
                ingredients_tensor[i*batch_size:(i+1)*batch_size].to(device), 
                instructions_tensor[i*batch_size:(i+1)*batch_size].to(device))
            )
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    return (title_tensor, ingredients_tensor, instructions_tensor), feats

def choose_one_image(rcp, img_dir, transform=None):
    part = rcp['partition']
    local_paths = rcp['images']
    local_path = np.random.choice(local_paths)
    img_path = os.path.join(img_dir, part, local_path)
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    return img

def load_recipes(file_path, part=None):
    with open(file_path, 'r') as f:
        info = json.load(f)
    if part:
        info = [x for x in info if x['partition']==part]
    return info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument(
        '--part', default='test', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--index', default=0, type=int)
    args = parser.parse_args()

    import pprint
    pp = pprint.PrettyPrinter()

    recipes = load_recipes(args.part)

    print(args.part, len(recipes))
    
    k = args.index
    recipe = recipes[k]
    pp.pprint(recipe)
    print()

    w2i = load_dict('word2vec_Recipe1M/vocab.txt')
    i2w = {i:w for w,i in w2i.items()}
    print('vacab size', len(w2i))
    print()

    # instructions
    vec = get_instructions_wordvec(recipe, w2i)
    num_sents = vec.nonzero()[0].max() + 1
    for vec_sent in vec[:num_sents]:
        num_words = vec_sent.nonzero()[0].max() + 1
        print(vec_sent[:num_words])
        print(' '.join([i2w[i] for i in vec_sent[:num_words]]))
    print()

    # ingredients
    vec = get_ingredients_wordvec(recipe, w2i)
    num_words = vec.nonzero()[0].max() + 1
    for i in vec[:num_words]:
        print(i, i2w[i])
    print()

    # title
    vec = get_title_wordvec(recipe, w2i)
    num_words = vec.nonzero()[0].max() + 1
    for i in vec[:num_words]:
        print(i, i2w[i])
    print()

    # # compute coverage
    # cvgs = []
    # for recipe in tqdm(recipes):
    #     vec = get_ingredients_wordvec(recipe, ingr2i, replace_dict)
    #     num_words = vec.nonzero()[0].max() + 1
    #     valid_vec = vec[:num_words]
    #     num_words_inDict = valid_vec[valid_vec!=1].shape[0]
    #     cvg = 1.0 * num_words_inDict / num_words
    #     cvgs.append((recipe['title'], int(num_words), float(cvg)))

    # cvgs = sorted(cvgs, key=lambda x: -x[2])
    # with open('v1/coverage_{}.json'.format(args.part), 'w') as f:
    #     json.dump(cvgs, f, indent=2)
    
    # cvg_nums = [x[2] for x in cvgs]
    # print('mean cvg =', sum(cvg_nums)/len(cvg_nums))
    # from matplotlib import pyplot as plt
    # plt.hist(cvg_nums, bins=100)
    # plt.show()


# def prepare_data(data, device):
#     imgs, w_imgs, txt, _ = data
#     real_vimgs, wrong_vimgs = [], []
#     for i in range(args.levels):
#         real_vimgs.append(imgs[i].to(device))
#         wrong_vimgs.append(w_imgs[i].to(device))
#     vtxt = [x.to(device) for x in txt]
#     return real_vimgs, wrong_vimgs, vtxt

# def _get_img_embeddings(img, ImgEnc):
#     img = img/2 + 0.5
#     img = F.interpolate(img, [224, 224], mode='bilinear', align_corners=True)
#     for i in range(img.shape[1]):
#         img[:,i] = (img[:,i]-mean[i])/std[i]
#         img_feats = ImgEnc(img).detach().cpu()
#     return img_feats

# def eval_gan(
#     dataloader, txt_encoder, img_encoder, generator, save_dir, 
#     device=torch.device('cuda'), save_all=False):
    
#     txt_feats_real = []
#     img_feats_real = []
#     img_feats_fake = []

#     fixed_noise = torch.FloatTensor(1, args.z_dim).normal_(0, 1).to(device)
#     fixed_noise = fixed_noise.repeat(args.batch_size, 1)
#     batch = 0
#     for data in tqdm(dataloader):
#         real_imgs, _, txt = prepare_data(data, device)
        
#         with torch.no_grad():
#             txt_embedding = txt_encoder(txt)
#             fake_imgs, _, _ = generator(fixed_noise, txt_embedding)
#             img_fake = fake_imgs[-1]
#             img_embedding_fake = _get_img_embeddings(img_fake, img_encoder)
#             img_real = real_imgs[-1]
#             img_embedding_real = _get_img_embeddings(img_real, img_encoder)
        
#         txt_feats_real.append(txt_embedding.detach().cpu())
#         img_feats_real.append(img_embedding_real.detach().cpu())
#         img_feats_fake.append(img_embedding_fake.detach().cpu())
            

#         if batch == 0:
#             noise = torch.FloatTensor(args.batch_size, args.z_dim).normal_(0, 1).to(device)
#             one_txt_feat = txt_embedding[0:1]
#             one_txt_feat = one_txt_feat.repeat(args.batch_size, 1)
#             fakes, _, _ = generator(noise, one_txt_feat)
#             save_image(
#                 fakes[-1], 
#                 os.path.join(save_dir, 'random_noise_image0.jpg'), 
#                 normalize=True, scale_each=True)
    
#         # save_image(
#         #         real_imgs[-1], 
#         #         os.path.join(save_dir, 'batch{}_real.jpg'.format(batch)), 
#         #         normalize=True)
#         save_image(
#                 fake_imgs[0], 
#                 os.path.join(save_dir, 'batch{}_fake0.jpg'.format(batch)), 
#                 normalize=True, scale_each=True)
#         save_image(
#                 fake_imgs[1], 
#                 os.path.join(save_dir, 'batch{}_fake1.jpg'.format(batch)), 
#                 normalize=True, scale_each=True)
#         # save_image(
#         #         fake_imgs[2], 
#         #         os.path.join(save_dir, 'batch{}_fake2.jpg'.format(batch)), 
#         #         normalize=True, scale_each=True)

#         real_fake = torch.stack([real_imgs[-1], fake_imgs[-1]]).permute(1,0,2,3,4).contiguous()
#         real_fake = real_fake.view(-1, real_fake.shape[-3], real_fake.shape[-2], real_fake.shape[-1])
#         save_image(
#                 real_fake, 
#                 os.path.join(save_dir, 'batch{}_real_fake.jpg'.format(batch)), 
#                 normalize=True, scale_each=True)
#         batch += 1

#     txt_feats_real = torch.cat(txt_feats_real, dim=0)
#     img_feats_real = torch.cat(img_feats_real, dim=0)
#     img_feats_fake = torch.cat(img_feats_fake, dim=0)
#     print('=> computing ranks...')
#     retrieved_range = min(900, len(dataloader)*args.batch_size)
#     medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_real.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
#     print('=> Real MedR: {:.4f}({:.4f})'.format(medR, medR_std))
#     for k, v in recalls.items():
#         print('Real Recall@{} = {:.4f}'.format(k, v))

#     medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_fake.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
#     print('=> Fake MedR: {:.4f}({:.4f})'.format(medR, medR_std))
#     for k, v in recalls.items():
#         print('Fake Recall@{} = {:.4f}'.format(k, v))
