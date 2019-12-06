import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from dcgan import Generator
from dcgan import Discriminator

from dataloader import ImageLoader 
from textEncoder import textEncoder
from gan_args import get_parser

import time

# =============================================================================
parser = get_parser()
opt = parser.parse_args()
# =============================================================================

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opt.seed)
    device = torch.device(*('cuda',0))
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main():
	# Loss function
	adversarial_loss = torch.nn.BCELoss()

	# Initialize generator and discriminator
	generator = Generator()
	discriminator = Discriminator()

	# Initialize weights
	generator.apply(weights_init_normal)
	discriminator.apply(weights_init_normal)



	# DataParallel
	generator = nn.DataParallel(generator).to(device)
	discriminator = nn.DataParallel(discriminator).to(device)

	# Dataloader
	# data preparation, loaders
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	# cudnn.benchmark = True

    # preparing the training laoder
	train_loader = torch.utils.data.DataLoader(
	    ImageLoader(opt.img_path,
	        transforms.Compose([
	        transforms.Scale(128), # rescale the image keeping the original aspect ratio
	        transforms.CenterCrop(128), # we get only the center of that rescaled
	        transforms.RandomCrop(128), # random crop within the center crop 
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        normalize,
	    ]),data_path=opt.data_path,partition='train'),
	    batch_size=opt.batch_size, shuffle=True,
	    num_workers=opt.workers, pin_memory=True)
	print('Training loader prepared.')


	# preparing validation loader 
	val_loader = torch.utils.data.DataLoader(
	    ImageLoader(opt.img_path,
	        transforms.Compose([
	        transforms.Scale(128), # rescale the image keeping the original aspect ratio
	        transforms.CenterCrop(128), # we get only the center of that rescaled
	        transforms.ToTensor(),
	        normalize,
	    ]),data_path=opt.data_path,partition='val'),
	    batch_size=opt.batch_size, shuffle=False,
	    num_workers=opt.workers, pin_memory=True)
	print('Validation loader prepared.')


	# Optimizers
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

	# ----------
	#  Training
	# ----------
	for epoch in range(opt.n_epochs):
	    pbar = tqdm(total=len(train_loader))

	    start_time = time.time()
	    for i, data in enumerate(train_loader):

	        input_var = list() 
	        for j in range(len(data)):
	            # if j>1:
	            input_var.append(data[j].to(device))


	        imgs = input_var[0]
	        # Adversarial ground truths
	        valid = np.ones((imgs.shape[0],1))
	        valid = torch.FloatTensor(valid).to(device)
	        fake = np.zeros((imgs.shape[0],1))
	        fake = torch.FloatTensor(fake).to(device)
	        # -----------------
	        #  Train Generator
	        # -----------------

	        optimizer_G.zero_grad()
	        # Sample noise as generator input
	        z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
	        z = torch.FloatTensor(z).to(device)
	        # Generate a batch of images
	        gen_imgs = generator(z, input_var[1], input_var[2], input_var[3], input_var[4])

	        # Loss measures generator's ability to fool the discriminator
	        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

	        g_loss.backward()
	        optimizer_G.step()
	        # ---------------------
	        #  Train Discriminator
	        # ---------------------
	        optimizer_D.zero_grad()

	        # Measure discriminator's ability to classify real from generated samples
	        real_loss = adversarial_loss(discriminator(imgs), valid)
	        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
	        d_loss = (real_loss + fake_loss) / 2

	        d_loss.backward()
	        optimizer_D.step()

	        pbar.update(1)
	    
	    pbar.close()
	    print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time Elapsed: %f]"
			% (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item(), time.time()-start_time)
		)

	    if epoch % opt.sample_interval == 0:
		    save_samples(epoch, gen_imgs.data[:25])
		    save_model(epoch,generator.state_dict(), discriminator.state_dict())

def save_samples(epoch, imgs):
	img_folder = opt.save_image
	if not os.path.isdir(img_folder):
		os.makedirs(img_folder)
		print("make dir ", img_folder)

	img_path = os.path.join(img_folder,str(epoch)+".png")
	save_image(imgs, img_path, nrow=5, normalize=True)


def save_model(epoch,g_state,d_state):
	models_forlder = opt.save_model
	if not os.path.isdir(models_forlder):
		os.makedirs(models_forlder)
		print("make dir", models_forlder)

	g_file_name = os.path.join(models_forlder, "g_" + str(epoch)+".pth.tar")
	d_file_name = os.path.join(models_forlder, "d_" + str(epoch)+".pth.tar")
	torch.save(g_state,g_file_name)
	torch.save(d_state,d_file_name)

if __name__ == "__main__":
	main()