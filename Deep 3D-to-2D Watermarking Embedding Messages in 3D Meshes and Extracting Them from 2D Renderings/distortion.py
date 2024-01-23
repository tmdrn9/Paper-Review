import torch.nn as nn
from utils import addNoise,Resize
import torchvision.transforms as T
import random
from Decoder import ConvBNRelu

class distortionLayer(nn.Module):
    def __init__(self, sigma=0.01, pi=30, scale=0.75):
        super(distortionLayer, self).__init__()
        self.no=nn.Identity()
        self.noise=addNoise(sigma)
        self.rotation=T.RandomRotation(pi)
        self.scaling=Resize([scale, 1])

    def forward(self, T_e):
        x = random.randint(0,3)
        if x==0:
            T_e=self.no(T_e)
        elif x==1:
            T_e = self.noise(T_e)
        elif x==2:
            T_e = self.rotation(T_e)
        else:
            T_e = self.scaling(T_e)
        return T_e

class attackNetwork(nn.Module):
    def __init__(self,channels=16):
        super(attackNetwork, self).__init__()
        self.channels = channels
        self.layers = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, 1, padding='same'),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.channels, 3, 3, 1, padding='same')
        )

    def forward(self, x):
        return self.layers(x)