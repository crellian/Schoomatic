import cv2
import torch
import numpy as np
import os
import random
import torch.nn.functional as F
from models import VAEBEV

class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(64, 32)
        self.linear_2 = torch.nn.Linear(32, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        layer_in = self.linear_1(x)
        act = self.relu(layer_in)
        out = self.linear_2(act)
        return out

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(0 if use_cuda else "cpu")
    vae_model_path = "/lab/kiran/ckpts/pretrained/models/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False


    root_dir = "train"
    images = []
    labels = []
    for root, _, files in os.walk(root_dir):
        print(root)
        if len(files) > 10:
            for f in files:
                if "_" not in f:
                    with torch.no_grad():
                        im = cv2.imread(os.path.join(root,f), cv2.IMREAD_GRAYSCALE)
                        im = np.expand_dims(im, axis=(0, 1))
                        im = torch.tensor(im).to(device) / 255.0
                        recon, mu, logvar = vae(im)
                        #cv2.imwrite(os.path.join(root,f[:-4]+"_.jpg"), recon.reshape(64,64).cpu().numpy()*255)
                        fs = torch.concat((mu, logvar), axis=-1).reshape(64, )

                        images.append(fs)
                        labels.append(int(root[-1]))

    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)
    print(images.shape, labels.shape)
    ind_list = [i for i in range(len(labels))]
    random.shuffle(ind_list)
    images_train = images[ind_list, :]
    labels_train = labels[ind_list,]

    net = FFN().to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for i in range(100):
        net.train()
        ind_list = [i for i in range(len(labels_train))]
        random.shuffle(ind_list)
        images_train = images_train[ind_list, :]
        labels_train = labels_train[ind_list,]

        optimizer.zero_grad()
        output = net(images_train)
        loss = loss_func(output, labels_train)

        loss.backward()
        optimizer.step()
        print("train", loss.item())


    torch.save(net.state_dict(), "bev.pt")
