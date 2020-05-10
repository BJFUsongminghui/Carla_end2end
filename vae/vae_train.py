from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import logging
import h5py
import random
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

logger_path = "./log"
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logging.basicConfig(level=logging.INFO, filename=os.path.join(logger_path, 'vae_epoch_original_Feb_24.log'))
learn_log = SummaryWriter('learn_logs')
def preprocess_rgb_frame(frame):
    frame = frame[:, :, :3]
    frame = frame.astype(np.float32) / 255.0 # [0, 255] -> [0, 1]
    return frame

class CarlaAEBSDataset(Dataset):
    def __init__(self):
        self.rgb = self.get_data()

    def get_data(self):
        images = []
        dir_path = 'rgb'
        for filename in os.listdir(dir_path):
            _, ext = os.path.splitext(filename)
            if ext == ".png":
                filepath = os.path.join(dir_path, filename)
                frame = preprocess_rgb_frame(np.asarray(Image.open(filepath)))
                frame = frame.reshape((3, 88, 200))
                images.append(frame)
        return images

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        return self.rgb[idx]


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


class VAETrainer():
    def __init__(self, epoch):
        self.dataset = CarlaAEBSDataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch

    def fit(self):
        self.model = VAE()
        if os.path.exists('model/vae.pt'):
            self.model.load_state_dict(torch.load('model/vae.pt'))
        self.model = self.model.to(self.device)
        data_loader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=8)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.epoch * 0.7)], gamma=0.1)
        self.model.train()

        for epoch in range(self.epoch):
            loss_epoch = 0.0
            reconstruction_loss_epoch = 0.0
            kl_loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs) in enumerate(data_loader):
                inputs = inputs.to(self.device).float()
                optimizer.zero_grad()

                x, mu, logvar,_ = self.model(inputs)

                #reconstruction_loss = torch.mean((x - inputs)**2, dim=tuple(range(1, x.dim())))
                reconstruction_loss = torch.sum((x-inputs)**2, dim=tuple(range(1, x.dim())))
                kl_loss = 1 + logvar - (mu).pow(2) - logvar.exp()
                #kl_loss = torch.mean(kl_loss, axis=-1) * -0.5
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                #loss = 150*reconstruction_loss + kl_loss
                loss = reconstruction_loss + kl_loss
                reconstruction_loss_mean = torch.mean(reconstruction_loss)
                kl_loss_mean = torch.mean(kl_loss)
                loss = torch.mean(loss)

              #  loss = self.loss_function(x, inputs, mu, logvar)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                reconstruction_loss_epoch += reconstruction_loss_mean.item()
                kl_loss_epoch += kl_loss_mean.item()
                n_batches += 1

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logging.info('Epoch {}/{}\t Time: {:.3f}\t Total Loss: {:.3f}\t\
                        Reconstruction Loss {:.3f}\t KL Loss {:.3f}' \
                .format(
                epoch + 1, self.epoch, epoch_train_time, loss_epoch / n_batches, reconstruction_loss_epoch / n_batches, \
                kl_loss_epoch / n_batches))
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, self.epoch, epoch_train_time,
                                                                     loss_epoch / n_batches))
            learn_log.add_scalar('loss/value_loss', loss_epoch / n_batches, epoch)

            if epoch % 10000 ==0:
                self.save_model(filename=f'{loss_epoch / n_batches:.1f}model.pt')
        return self.model

    def save_model(self, path="./model",filename="vae.pt"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    def loss_fuction(self, recon_x,x,mu,logvar):
        reconatruction_fuction=nn.BCELoss()
        BCE=reconatruction_fuction(recon_x,x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element)*(-0.5)
        return BCE + KLD

if __name__ == '__main__':
    trainer = VAETrainer(100000)
    trainer.fit()
    trainer.save_model()
