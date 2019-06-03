# DCGAN-like generator and discriminator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torch.nn.init as weight_init


channels = 4 *3

def normalize_vector(x, eps=.0001):
    # Add epsilon for numerical stability when x == 0
    norm = torch.norm(x, p=2, dim=1) + eps
    return x / norm.expand(1, -1).t()


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.conv0 = (nn.Conv2d(channels, 32, 3, stride=2, padding=(1,1)))
        self.batch0 = nn.BatchNorm2d(32)
         # Input: 40x40x?
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2, padding=(1,1))
        self.batch1 = nn.BatchNorm2d(64)
        # 40 x 40 x 64
        self.conv2 =  nn.Conv2d(64, 128, 4, stride=2, padding=(1,1))
        self.batch2 = nn.BatchNorm2d(128)
        # 20 x 20 x 128
        self.conv3 =  nn.Conv2d(128, 256, 4, stride=2, padding=(1,1))
        self.batch3 = nn.BatchNorm2d(256)
        # 10 x 10 x 256
        self.conv4 =  nn.Conv2d(256, 256, 4, stride=2, padding=(1,1))
        self.batch4 = nn.BatchNorm2d(256)
        # 5 x 5 x 256
        self.conv5 =  nn.Conv2d(256, 256, 3, stride=1, padding=(0,0))
        self.batch5 = nn.BatchNorm2d(256)
        # 3 x 3 x 256

        self.hidden_units = 3 * 3 * 256
        self.fc = nn.Linear(self.hidden_units, latent_size)



    def forward(self, x):
        #(hx, cx) = memory
        x = self.leaky(self.batch0(self.conv0(x)))
        x = self.leaky(self.batch1(self.conv1(x)))
        x = self.leaky(self.batch2(self.conv2(x)))
        x = self.leaky(self.batch3(self.conv3(x)))
        x = self.leaky(self.batch4(self.conv4(x)))
        x = self.leaky(self.batch5(self.conv5(x)))
        x = x.view((-1, self.hidden_units))

        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        
        self.fc = nn.Linear(z_dim , z_dim)
        self.deconv1 = nn.ConvTranspose2d(z_dim , 512, 4, stride=2)
        self.batch1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512 , 256, 4, stride=2, padding=0) # 10
        self.batch2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256 , 128, 4, stride=2, padding=(1,1)) #20
        self.batch3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128 , 128, 4, stride=2, padding=(1,1)) #40
        self.batch4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128 , 64, 4, stride=2, padding=(1,1))
        self.batch5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, channels, 4, stride=2, padding=(1,1))


    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view((-1, self.z_dim, 1, 1))
        x = F.relu(self.batch1(self.deconv1(x)))
        x = F.relu(self.batch2(self.deconv2(x)))
        x = F.relu(self.batch3(self.deconv3(x)))
        x = F.relu(self.batch4(self.deconv4(x)))
        x = F.relu(self.batch5(self.deconv5(x)))
        x = self.deconv6(x)

        return torch.sigmoid(x)


class Agent(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions):
        super(Agent, self).__init__()

        self.latent_size = 256
        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 5 * 5, self.latent_size )
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def get_latent_size(self):
        return self.latent_size

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.linear(x.view(-1, 32 * 5 * 5))
        return x
        #return self.critic_linear(x), self.actor_linear(x)

    def pi(self, x):
        return self.actor_linear(x)

    def value(self, x):
        return self.critic_linear(x)
