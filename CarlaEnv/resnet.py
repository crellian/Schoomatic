import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import bisect
import random

from models import VAEBEV, MDLSTM
from classify import FFN

class ResNet:
    def __init__(self):
        self.model = tf.keras.models.load_model('/lab/kiran/img2cmd_data/model/cnn')

    def predict(self, rgb):
        if len(rgb.shape) < 4:
            rgb = np.expand_dims(rgb, axis=0)
        rgb = rgb / 255.0
        probs = self.model.predict(rgb, verbose=0)

        return (np.argmax(probs, axis=1), probs)


def normalize_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame


def readReal():
    for i in range(10):
        for root, subdirs, files in os.walk("/home2/USC_GStView/" + str(i) + "/"):
            t = 0
            cnt = np.zeros(10)
            for f in files:
                if '00.jpg' in f:
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
                print(root, len(files)/3, "accuracy:", t*3 / len(files))
                print(cnt*3 / len(files))


def readReal2():
    for root, subdirs, files in os.walk("/home/tmp/kiran/USC_GStView/5/"):
        for f in files:
            print(f)
            rgb = cv2.imread(os.path.join(root, f))
            rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)

            label, probs = resnet.predict(rgb)

            print(np.argsort(-probs))





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


if __name__ == "__main__":
    resnet = ResNet()
    readReal()
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
