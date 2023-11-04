import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")

class StateLSTM(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, encoder):
        super().__init__()
        self.encoder = encoder
        if encoder is not None:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True).cuda()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_hs(self):
        self.h_0 = Variable(torch.randn((self.num_layers, self.hidden_size))).to(device)
        self.c_0 = Variable(torch.randn((self.num_layers, self.hidden_size))).to(device)

    def forward(self, image):
        # x = torch.reshape(image, (-1,) + image.shape[-3:]).float()
        x = image
        z = self.encoder(x).float()
        z = torch.reshape(z, (1, image.shape[0], -1))
        # z = torch.reshape(z, image.shape[:2] + (-1,))
        outs, (self.h_0, self.c_0) = self.lstm(z.float(), (self.h_0, self.c_0))
        return outs


class StateActionLSTM(StateLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, encoder=None, vae=None):
        super().__init__(latent_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, encoder=encoder)
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, num_layers, batch_first=True)

    def encode(self, image):
        x = torch.reshape(image, (-1,) + image.shape[-3:])
        _, mu, logvar = self.vae(x)
        z = self.vae.reparameterize(mu, logvar)
        z = torch.reshape(z, image.shape[:2] + (-1,))
        return z, mu, logvar

    def decode(self, z):
        z_f = torch.reshape(z, (-1,) + (z.shape[-1],))
        img = self.vae.recon(z_f)
        return torch.reshape(img, z.shape[:2] + img.shape[-3:])

    def forward(self, action, latent):
        in_al = torch.cat([action, latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))
        return outs


class MDLSTM(StateActionLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, gaussian_size, encoder=None, vae=None):
        super().__init__(latent_size, action_size, hidden_size, num_layers, encoder, vae)
        self.gaussian_size = gaussian_size
        self.gmm_linear = nn.Linear(hidden_size, (2 * latent_size + 1) * gaussian_size)

    def forward(self, action, latent):
        seq_len = action.size(0)
        in_al = torch.cat([torch.Tensor(action), latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))

        gmm_outs = self.gmm_linear(outs)
        stride = self.gaussian_size * self.latent_size

        mus = gmm_outs[:, :stride]
        mus = mus.view(seq_len, self.gaussian_size, self.latent_size)

        sigmas = gmm_outs[:, stride:2 * stride]
        sigmas = sigmas.view(seq_len, self.gaussian_size, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, 2 * stride: 2 * stride + self.gaussian_size]
        pi = pi.view(seq_len, self.gaussian_size)
        logpi = f.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi