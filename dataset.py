import json
import os
from glob import glob
import numpy as np
import torch
from torch import nn
from torch.utils import data
from utils import load_recipes, load_dict, get_image_paths, choose_one_image
from utils import get_title_wordvec, get_ingredients_wordvec, get_instructions_wordvec
import pdb

class Dataset(data.Dataset):
    def __init__(self, part, data_dir, img_dir, transform=None, permute_ingrs=False):
        assert part in ('train', 'val', 'test'), \
            'part must be one of [train, val, test]'
        # self.recipes = load_recipes(os.path.join(data_dir,'{}.json'.format(part)))
        self.recipes = load_recipes(os.path.join(data_dir,'recipesV1.json'), part)
        self.vocab_inst = load_dict(os.path.join(data_dir, 'vocab_inst.txt'))
        print('vocab_inst size =', len(self.vocab_inst))
        self.vocab_ingr = load_dict(os.path.join(data_dir, 'vocab_ingr.txt'))
        print('vocab_ingr size =', len(self.vocab_ingr))
        self.img_dir = img_dir
        self.transform = transform
        self.permute_ingrs = permute_ingrs
    
    def _prepare_one_recipe(self, index):
        rcp = self.recipes[index]
        title = get_title_wordvec(rcp, self.vocab_inst) # np.int [max_len]
        ingredients = get_ingredients_wordvec(rcp, self.vocab_ingr, self.permute_ingrs) # np.int [max_len]
        instructions = get_instructions_wordvec(rcp, self.vocab_inst) # np.int [max_len, max_len]
        img = choose_one_image(rcp, self.img_dir, self.transform) # tensor [3, 224, 224]
        # img = torch.randn(3, 224, 224)
        return (title, ingredients, instructions), img
    
    def __getitem__(self, index):
        recipe_data = self._prepare_one_recipe(index)
        return recipe_data
    
    def __len__(self):
        return len(self.recipes)
