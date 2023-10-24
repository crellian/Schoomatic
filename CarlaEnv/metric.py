import numpy as np
#import tensorflow as tf
#from tensorflow.keras.utils import Sequence
import os
import cv2
import matplotlib.pyplot as plt

import bisect
import random
import torch
import matplotlib.pyplot as plot

from models import VAEBEV, MDLSTM
from classify import FFN
use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")
'''
class ResNet:
    def __init__(self):
        model = tf.keras.models.load_model('/lab/kiran/img2cmd_data/model_')

    def predict(self, rgb):
        if len(rgb.shape) < 4:
            rgb = np.expand_dims(rgb, axis=0)
        rgb = rgb / 255.0
        probs = model.predict(rgb, verbose=0)

        return (np.argmax(probs, axis=1), probs)
'''

def normalize_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame


def readReal():
    for i in range(10):
        for root, subdirs, files in os.walk("/home2/USC_GStView/" + str(i) + "/"):
            t = 0
            cnt = np.zeros(10)
            for f in files:
                if '.jpg' in f:
                    # print(os.path.join(root, f))
                    rgb = cv2.imread(os.path.join(root, f))
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)

                    label, probs = resnet.predict(rgb)

                    # print(probs)
                    if np.argsort(-probs)[0][0] == i:
                        t += 1
                    cnt[np.argsort(-probs)[0][0]] += 1
                    print(np.argsort(-probs)[0][0], f)
            if files:
                print(root, len(files), "accuracy:", t / len(files))
                print(cnt / len(files))


def readReal2():
    for root, subdirs, files in os.walk("/home/tmp/kiran/USC_GStView/5/"):
        for f in files:
            print(f)
            rgb = cv2.imread(os.path.join(root, f))
            rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)

            label, probs = resnet.predict(rgb)

            print(np.argsort(-probs))


def readSim():
    loss = {"mse": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
            "entropy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
            "accuracy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
            }
    for root, subdirs, files in os.walk("/home2/random_bev_carla/new"):
        for f in files:
            if "observation_rgb" in f:
                rgb = np.load(os.path.join(root, f), mmap_mode='r')
                bev = np.load(os.path.join(root, "observation.npy"), mmap_mode='r')
                ter = np.load(os.path.join(root, "terminal.npy"))
                action = np.load(os.path.join(root, "action.npy"))

                bev_lstm.init_hs()
                window = []
                correction = False
                z_in = None
                step_count = 0
                with torch.no_grad():
                    cos = torch.nn.CosineSimilarity(dim=0)
                    for i in range(len(rgb)):
                        step_count += 1
                        RGB_img = cv2.resize(rgb[i], (84, 84), interpolation=cv2.INTER_LINEAR)

                        BEV_img = bev[i][:,:,0]  # ground truth
                        BEV_img = np.expand_dims(BEV_img, axis=(0, 1))
                        BEV_img = torch.tensor(BEV_img).to(device) / 255.0
                        _, mu, logvar = vae(BEV_img.float())
                        fs = torch.concat((mu, logvar), axis=-1).reshape(64, )
                        output = net(fs).reshape(8, )
                        prob = torch.exp(output) / torch.sum(torch.exp(output))
                        #plot.bar(range(8), prob.cpu().numpy())
                        #plot.pause(0.01)
                        #plot.clf()
                        id = torch.argmax(prob).cpu().numpy()
                        if id == 6 or id == 7:
                            id = 4

                        window.append(int(id))
                        if step_count < 10:
                            z_obs = bev_lstm.vae.reparameterize(anchors_lr[id][:32], anchors_lr[id][32:])
                            z_in = torch.unsqueeze(z_obs, dim=0)
                        else:
                            window = window[1:]
                            mid = max(set(window), key=window.count)
                            z_obs = bev_lstm.vae.reparameterize(anchors_lr[mid][:32],
                                                                     anchors_lr[mid][32:])
                            if torch.max(prob) > 0.95:
                                correction = True
                            if correction:
                                dis = cos(z_in[0], z_obs)
                                if dis > 0.5:
                                    correction = False
                                z_in = torch.unsqueeze(z_obs, dim=0)

                        # raw prediction
                        # r = torch.reshape(bev_lstm.vae.recon(torch.unsqueeze(z_obs, dim=0)) * 255, (64, 64))
                        #cv2.imshow("0", r.cpu().numpy())
                        # robust prediction
                        r_ = torch.reshape(bev_lstm.vae.recon(z_in), (1, 1, 64, 64))
                        #cv2.imshow("1", r_.cpu().numpy())
                        #cv2.imshow("2", BEV_img[0][0].cpu().numpy())
                        #cv2.waitKey(10)
                        mse = torch.nn.functional.mse_loss(BEV_img, r_)
                        entropy = torch.nn.functional.cross_entropy(BEV_img, r_)
                        loss["mse"][gt].append(mse)
                        loss["entropy"][gt].append(entropy)

                        # next prediction
                        a = torch.Tensor([action[i]]).to(device)
                        out = bev_lstm(a, z_in)
                        mus = out[0][0]
                        pi = torch.exp(out[2][0])
                        z_in = (mus[0] * pi[0] + mus[1] * pi[1] + mus[2] * pi[2] + mus[3] * pi[3] + mus[4] * pi[
                            4]).unsqueeze(0)

    
                        if ter[i]:
                            bev_lstm.init_hs()
                            window = []
                            correction = False
                            z_in = None
                            step_count = 0
                    total.append(loss_acc / len(action))
    return total





'''
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, transform_params=None):
        x, y = x_set, y_set
        batch_size = batch_size
        each_len = cal_len()
        transform_params = transform_params

    def __len__(self):
        return int(np.ceil(each_len[-1] / float(batch_size)))

    def __getitem__(self, idx):
        idx = idx * batch_size
        file_ind = bisect.bisect_right(each_len, idx)
        if file_ind == 0:
            im_ind = idx
        else:
            im_ind = idx - each_len[file_ind - 1]
        batch_x = x[file_ind][im_ind:im_ind + batch_size]
        batch_y = y[file_ind][im_ind:im_ind + batch_size]

        batch_x = normalize_frame(batch_x)
        # if transform_params is not None:
        # bx = []
        # for i in range(len(batch_x)):
        #    bx.append(transform_frame(batch_x[i], transform_params))
        # batch_x = np.array(bx)
        # transform_frame_v = np.vectorize(transform_frame, excluded=['transform_params'], signature="(n, m, c), (a) -> (n, m, c)")
        # batch_x = transform_frame(batch_x, transform_params)
        return batch_x, batch_y

    def cal_len(self):
        each_len = []
        for i in range(len(y)):
            if len(each_len) == 0:
                each_len.append(y[i].shape[0])
            else:
                each_len.append(y[i].shape[0] + each_len[-1])
        return each_len

    def on_epoch_end(self):
        l = list(zip(x, y))
        random.shuffle(l)
        x, y = zip(*l)
        each_len = cal_len()

'''
if __name__ == "__main__":
    vae_model_path = "/lab/kiran/ckpts/pretrained/models/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
    lstm_path = "/lab/kiran/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    bev_lstm = MDLSTM(latent_size=32, action_size=2, hidden_size=256, num_layers=1, gaussian_size=5,
                           vae=vae).to(device)
    bev_lstm.init_hs()
    checkpoint = torch.load(lstm_path, map_location="cpu")
    bev_lstm.load_state_dict(checkpoint['model_state_dict'])
    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])
    bev_lstm.eval()
    for param in bev_lstm.parameters():
        param.requires_grad = False

    net = FFN().to(device)
    net.load_state_dict(torch.load("/home/carla/img2cmd/bev.pt"))
    net.eval()

    anchors_lr = []
    for i in range(6):
        im = cv2.imread(os.path.join("/home/carla/img2cmd/train", str(i) + ".jpg"), cv2.IMREAD_GRAYSCALE)
        with torch.no_grad():
            im = np.expand_dims(im, axis=(0, 1))
            im = torch.tensor(im).to(device) / 255.0
            _, mu, logvar = vae(im.float())
            fs = torch.concat((mu, logvar), axis=-1).reshape(64, )
            anchors_lr.append(fs.cpu().numpy())
    anchors_lr = np.array(anchors_lr)
    anchors_lr = torch.tensor(anchors_lr).to(device)

    #resnet = ResNet()
    total = readSim()
    print(total)
    print(torch.tensor(total).sum() / len(total))
    # readReal2()
    # readSim()
    # cnn prediction -- obtained from rgb
    # val_gen = DataGenerator(images, labels, 1)
    # loss = resnet.model.evaluate(val_gen)
    # print(loss)
    # plt.bar(range(len(probs[0])), probs[0])  # density=False would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('Class')

    # plt.show()
