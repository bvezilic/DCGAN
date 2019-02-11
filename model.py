import torch
import torch.nn as nn

import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size

        self.conv1 = nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        # [batch_size, latent_size, 1, 1]
        x = F.relu(self.bn1(self.conv1(x)))
        # [batch_size, 1024, 4, 4]
        x = F.relu(self.bn2(self.conv2(x)))
        # [batch_size, 512, 8, 8]
        x = F.relu(self.bn3(self.conv3(x)))
        # [batch_size, 256, 16, 16]
        x = F.relu(self.bn4(self.conv4(x)))
        # [batch_size, 128, 32, 32]
        x = F.tanh(self.conv5(x))
        # [batch_size, 3, 64, 64]
        return x


class Descriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [batch_size, 3, 64, 64]
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # [batch_size, 128, 32, 32]
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        # [batch_size, 256, 16, 16]
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        # [batch_size, 512, 8, 8]
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        # [batch_size, 1024, 4, 4]
        x = torch.sigmoid(self.conv5(x))
        # [batch_size, 1, 1, 1]
        return x
