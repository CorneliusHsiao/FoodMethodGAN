import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from utils.inception_score import get_inception_score
from itertools import chain
from torchvision import utils
import time
from args import get_parser

parser = get_parser()
opt = parser.parse_args()

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            #16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            #32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1))
       

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32) 128
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16) 64
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8) 32
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8) 16
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            # 8
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class DCGAN_MODEL(object):
    def __init__(self, args, device):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels)
        self.G = nn.DataParallel(self.G).to(device)
        self.D = Discriminator(args.channels)
        self.D = nn.DataParallel(self.D).to(device)
        self.C = args.channels

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss().to(device)

        self.device = device

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size

    def train(self, train_loader):

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()
            
            start_time = time.time()
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 100, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)


                images, z = Variable(images).to(self.device), Variable(z).to(self.device)
                real_labels, fake_labels = Variable(real_labels).to(self.device), Variable(fake_labels).to(self.device)


                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(images)
                print(outputs.size())
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                
                z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs, fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                # Compute loss with fake images
                z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs, real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time Elapsed: %f]"
                % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item(), time.time()-start_time)
            )

        if epoch % 10 == 0:
            self.save_samples(epoch, fake_images.data[:25])
            self.save_model(epoch,self.G.state_dict(), self.D.state_dict())

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




        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.to(self.device)
            z1 = z1.to(self.device)
            z2 = z2.to(self.device)

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))