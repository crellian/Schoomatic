import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from fpvbev import Encoder
from models import VAEBEV, MDLSTM
import torch
import numpy as np
from std_msgs.msg import Int64, Float64
from geometry_msgs.msg import Twist

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

class Map:
    def __init__(self):
        rospy.init_node('map', anonymous=True)

        vae_model_path = "/home/administrator/ckpts/pretrained/models/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        naive_lstm_path = "/home/administrator/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"

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

        self.encoder = Encoder("/home/administrator/ckpts/pretrained/models/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt",
                          False)

        self.aux = 0
        self.i = 0
        self.window = []
        self.z = None

        self.map_pub = rospy.Publisher(
            '/occ_map', Int64, queue_size=1)

        self.aux_pub = rospy.Publisher(
            '/aux', Float64, queue_size=1)

        self.img_sub = rospy.Subscriber(
            '/cam1/color/image_raw/compressed', CompressedImage, self.method)


        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def method(self, rgb):
        self.i += 1
        np_arr = np.frombuffer(rgb.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        RGB_img = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_LINEAR)

        rid, score, image_embed = self.encoder(RGB_img)  # FPV embedding and approx

        if self.i < 11:
            id = rid
            s = torch.unsqueeze(self.encoder.anchors_lr[rid], dim=0)  # input to lstm
            self.window.append(0)
        else:
            r_ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                               (64, 64)).cpu().numpy() * 255
            r__ = torch.reshape(self.bev_lstm.vae.recon(self.z),
                                (64, 64)).cpu().numpy() * 255



            if score > 0.85:
                self.window = self.window[1:]
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
        AUX.data = self.aux
        self.aux_pub.publish(AUX)
        self.map_pub.publish(ID)
        print(id)

        return self.encoder.anchors[id]


if __name__ == '__main__':
    map_ = Map()