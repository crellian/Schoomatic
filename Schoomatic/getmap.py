import rospy
from sklearn.metrics import mean_squared_error
import cv2
from models.encoder import VAEBEV, Encoder
from models.lstm import MDLSTM
import torch
from std_msgs.msg import Int64, Float64
from geometry_msgs.msg import Twist
from utils.misc import VAE_PATH, LSTM_PATH, ENCODER_PATH

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

class Map:
    def __init__(self):
        rospy.init_node('map', anonymous=True)

        vae_model_path = VAE_PATH
        naive_lstm_path = LSTM_PATH

        vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.bev_lstm = MDLSTM(latent_size=32, action_size=2, hidden_size=256, num_layers=1, gaussian_size=5,
                          vae=vae).to(device)
        self.bev_lstm.eval()
        self.bev_lstm.init_hs()
        checkpoint = torch.load(naive_lstm_path, map_location="cpu")

        self.bev_lstm.load_state_dict(checkpoint['model_state_dict'])
        for param in self.bev_lstm.parameters():
            param.requires_grad = False

        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        self.bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])

        self.encoder = Encoder(ENCODER_PATH)

        self.i = 0
        self.window = []
        self.z = None

        self.map_pub = rospy.Publisher(
            '/occ_map', Int64, queue_size=1)

        self.aux_pub = rospy.Publisher(
            '/aux', Float64, queue_size=1)

        self.cmd_sub = rospy.Subscriber(
            '/cmd', Twist, self.cmd_cb)

        self.throttle = 0
        self.steer = 0

    # our state check method
    def method(self, info, aux):
        self.i += 1
        RGB_img = cv2.resize(info['rgb_obs'], (84, 84), interpolation=cv2.INTER_LINEAR)

        rid, score, image_embed = self.encoder(RGB_img)  # FPV embedding and approx

        if self.i < 11:
            id = rid
            s = torch.unsqueeze(self.encoder.anchors_lr[rid], dim=0)  # input to lstm
            l_t = self.encoder.label[rid]  # label at t
            self.window.append(0)

        else:
            nid = self.encoder(self.z, False)[0]
            l_t = self.encoder.label[nid]

            r_ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                               (64, 64)).cpu().numpy() * 255
            r__ = torch.reshape(self.bev_lstm.vae.recon(self.z),  # LSTM raw output
                                (64, 64)).cpu().numpy() * 255

            mse = mean_squared_error(r_.reshape((1, 64 * 64)),
                                     r__.reshape((1, 64 * 64)))

            if score > 0.85:
                self.window = self.window[1:]
                self.window.append(1 if mse > 20000 else 0)
                w = 1
            else:
                w = 0

            self.z = image_embed * w + self.z * (1 - w)
            id, _, _ = self.encoder(self.z, False)
            s = torch.unsqueeze(self.encoder.anchors_lr[id], dim=0)


        self.z = s

        ID = Int64()
        ID.data = id
        AUX = Float64()
        AUX.data = aux
        self.aux_pub.publish(AUX)    # publish direction vector
        self.map_pub.publish(ID)     # publish obstacles to TEB planner

        return l_t, self.encoder.anchors[id]

    def cmd_cb(self, twist):
        self.throttle = twist.linear.x  # receive velocities from TEB planner
        self.steer = twist.angular.z


