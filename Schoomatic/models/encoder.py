import numpy as np
import torch.nn as nn
import os
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
from utils.misc import VAE_PATH, ANCHOR_PATH

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch * 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def recon(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.recon(z), mu, logvar

class ResNet(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_size)


    def forward(self, image):
        out = self.resnet(image)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        self.fpvencoder = ResNet(32).to(device)
        bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)

        # load models
        vae_model_path = VAE_PATH
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        bevencoder.eval()
        for param in bevencoder.parameters():
            param.requires_grad = False

        checkpoint = torch.load(encoder_path, map_location="cpu")
        print(checkpoint['epoch'])
        self.fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
        self.fpvencoder.eval()
        for param in self.fpvencoder.parameters():
            param.requires_grad = False

        # read anchor images and convert to latent representations
        self.anchors_lr = []
        self.anchors = []
        self.label = []
        self.fn = []
        root = ANCHOR_PATH

        for root, subdirs, files in os.walk(root):
            for f in sorted(files):
                if ".jpg" in f:
                    im = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    self.anchors.append(im)
                    self.label.append(int(root[-1]))
                    self.fn.append(f[:-4])
                    with torch.no_grad():
                        im = np.expand_dims(im, axis=(0, 1))
                        im = torch.tensor(im).to(device) / 255.0
                        _, embed_mu, embed_logvar = bevencoder(im)
                        embed_mu = embed_mu.cpu().numpy()[0]
                        embed_logvar = embed_logvar.cpu().numpy()[0]
                        self.anchors_lr.append(embed_mu)

        self.anchors_lr = np.array(self.anchors_lr)
        self.anchors_lr = torch.tensor(self.anchors_lr).to(device)


    def forward(self, img, fpv=True):
        # img - rgb observation, bev - ground truth bev observation
        if fpv:
            img = np.expand_dims(img, axis=0)
            img = np.transpose(img, (0,3,1,2))
            image_val = torch.tensor(img).to(device) / 255.0

            with torch.no_grad():
                # encode rgb image
                image_embed = self.fpvencoder(image_val)
        else:
            image_embed = img

        sims = nn.functional.cosine_similarity(image_embed, self.anchors_lr)

        ys = torch.argmax(sims)

        return ys.cpu().numpy(), float(torch.max(sims).cpu().numpy()), image_embed
