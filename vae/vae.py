
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.rep_dim = 512
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 256, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(19968, 3300, bias=False)

        self.fc21 = nn.Linear(3300, self.rep_dim, bias=False)
        self.fc22 = nn.Linear(3300, self.rep_dim, bias=False)
        # self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04,affine=False)

        self.fc3 = nn.Linear(self.rep_dim, 3300, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(3300 / (11 * 25)), 128, 4, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 2, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight)
        self.bn2d8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(32, 3, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fc3(z))

        x = x.view(x.size(0), int(3300 / (11 * 25)), 11, 25)
        x = F.elu(x)
        x = self.deconv1(x)
        # x = F.interpolate(F.elu(self.bn2d5(x)), scale_factor=2)
        # x = self.deconv2(x)
        x = F.interpolate(F.elu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.elu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.elu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x, mu, logvar,z


class GETVAE():
    def __init__(self):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE()
        #self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('vae/model/vae.pt'))


    def get_code(self,rgb):
        input = torch.Tensor([rgb]).float() / 255.0
        #input = input.to(self.device).float()
        x, mu, logvar, z = self.model(input)
        z_v = z.squeeze()
        z__=z_v.detach().numpy()

        return z__


