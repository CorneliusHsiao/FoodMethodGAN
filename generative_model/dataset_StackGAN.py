import torch
from torch.utils import data
from torchvision import transforms
import os
from os.path import join
import json
import numpy as np
from glob import glob
from PIL import Image
from inflection import pluralize, singularize

import sys
sys.path.append('../')
from utils import load_recipes, load_dict
from utils import get_title_wordvec, get_instructions_wordvec, get_ingredients_wordvec
from networks import TextEncoder

def get_imgs(img_path, imsize, levels=3,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    ret = []
    for i in range(levels):
        if i < (levels - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    return ret

class Dataset(data.Dataset):
    def __init__(
        self, data_dir, img_dir, food_type=None, levels=3, part='train', 
        base_size=64, transform=None, permute_ingrs=False):

        self.permute_ingrs = permute_ingrs
        self.recipes = load_recipes(os.path.join(data_dir, 'recipesV1.json'), part)
        if food_type:
            single = singularize(pluralize(food_type))
            self.recipes = [x for x in self.recipes \
                if food_type in x['title'] or single in x['title']]
        self.vocab_inst = load_dict(os.path.join(data_dir, 'vocab_inst.txt'))
        self.vocab_ingr = load_dict(os.path.join(data_dir, 'vocab_ingr.txt'))
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for _ in range(levels):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.img_dir = img_dir
        self.levels = levels

    def _choose_one_image(self, rcp):
        part = rcp['partition']
        local_paths = rcp['images']
        local_path = np.random.choice(local_paths)
        img_path = os.path.join(self.img_dir, part, local_path)
        imgs = get_imgs(
            img_path, 
            imsize=self.imsize,
            levels=self.levels, 
            transform=self.transform,
            normalize=self.norm)
        return imgs

    def _prepare_recipe_data(self, rcp):
        title = get_title_wordvec(rcp, self.vocab_inst) # np.int [max_len]
        ingredients = get_ingredients_wordvec(rcp, self.vocab_ingr, self.permute_ingrs) # np.int [max_len]
        instructions = get_instructions_wordvec(rcp, self.vocab_inst) # np.int [max_len, max_len]
        txt = (title, ingredients, instructions)
        imgs = self._choose_one_image(rcp)
        return txt, imgs


    def __getitem__(self, index):
        rcp_a = self.recipes[index]
        
        all_idx = range(len(self.recipes))
        rand_idx = np.random.choice(all_idx)
        while rand_idx == index:
            rand_idx = np.random.choice(all_idx)
        rcp_b = self.recipes[rand_idx]

        txt_a, imgs_a = self._prepare_recipe_data(rcp_a)
        _, imgs_b = self._prepare_recipe_data(rcp_b)
        return imgs_a, imgs_b, txt_a, rcp_a['id']

    def __len__(self):
        return len(self.recipes)


if __name__ == '__main__':
    from args_StackGANv2 import args
    from tqdm import tqdm
    imsize = args.base_size * (2 ** (args.levels-1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = Dataset(
        args.data_dir, args.img_dir, food_type='muffin', 
        levels=args.levels, part='train', base_size=64, transform=image_transform)

    num_gpu = torch.cuda.device_count()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size * num_gpu,
        drop_last=True, shuffle=False, num_workers=int(args.workers))

    print(len(dataset), len(dataloader))
    for imgs_a, imgs_b, txt_a, id_a in tqdm(dataloader):
        print(imgs_a[0].shape)
        print(imgs_a[1].shape)
        print(imgs_a[2].shape)
        print(txt_a[0].shape)
        print(txt_a[1].shape)
        print(txt_a[2].shape)
        print(len(id_a))
        print()
        break