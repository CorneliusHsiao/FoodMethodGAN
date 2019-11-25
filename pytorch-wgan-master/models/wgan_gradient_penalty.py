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

            #64
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image 128

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
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image 128
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State 64
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State 32
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(2048, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(4096, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=4096, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, args, device):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(args.channels)
        self.G = nn.DataParallel(self.G).to(device)
        self.D = Discriminator(args.channels)
        self.D = nn.DataParallel(self.D).to(device)
        self.C = args.channels


        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        self.device = device

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))


        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.critic_iter = 5
        self.lambda_term = 10



    def train(self, train_loader):

        for epoch in range(self.epochs):
            self.start_time = time.time()
        
            for i, (images, _) in enumerate(train_loader):
                one = torch.FloatTensor([1])
                mone = one * -1

                one = one.to(self.device)
                mone = mone.to(self.device)


                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0
                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):
                    self.D.zero_grad()

                    # Check for batch to have full batch_size
                    if (images.size()[0] != self.batch_size):
                        continue

                    z = torch.rand((self.batch_size, 100, 1, 1))

                    images, z = Variable(images.to(self.device)), Variable(z.to(self.device))


                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(images)
                    d_loss_real = torch.FloatTensor([d_loss_real.mean()])
                    print(d_loss_real.size())
                    print(mone.size())
                    d_loss_real.backward(mone)

                    # Train with fake images
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)

                    fake_images = self.G(z)
                    d_loss_fake = self.D(fake_images)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                    gradient_penalty.backward()


                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    self.d_optimizer.step()

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
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

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

        eta = eta.to(self.device)


        interpolated = eta * real_images + ((1 - eta) * fake_images)

        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

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


    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

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
        print("Saved interpolated images.")
