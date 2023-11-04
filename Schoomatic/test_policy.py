import threading
import numpy as np
import yaml
import pygame
from pygame.locals import *
from getmap import Map
from envs.env import Env
from ray.rllib.policy.policy import Policy
import cv2
from utils.misc import POLICY_PATH

actions = [[0, -1],
                       [0.4, -0.2],
                       [0.75, 0],
                       [0.2, 0],
                       [0.0, 0],
                       [0.4, 0.2],
                       [0, 1]
                       ]

policy_path = POLICY_PATH
if __name__ == '__main__':
    with open("config/carla_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    env = Env(env_config)
    action = np.zeros(2)
    my_restored_policy = Policy.from_checkpoint(policy_path)
    # init_state = state = [np.zeros([256], np.float32) for _ in range(2)]
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_idx = 0
    avg_reward = 0

    for i in range(1, env_config["task_config"]["num_rollouts"]+1):
        while True:
            try:
                obs = env.reset()
            except RuntimeError as e:  # disconnected from the server
                print(e)
                env.init_world()  # internal loop until reconnected
            else:
                break
        reward = 0
        a = 0
        obs = env.observation
        init_state = state = my_restored_policy.get_initial_state()
        getmap = Map()

        if env_config["recording"]:
            out = cv2.VideoWriter(str(vid_idx) + "_rllib.mp4", fourcc, 10.0, (1280, 720))
            vid_idx += 1
        else:
            out = None

        while True:
            '''
            # manual control for debugging
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[1] = -0.8
            elif keys[K_RIGHT] or keys[K_d]:
                action[1] = 0.8
            elif keys[K_s]:
                im = cv2.cvtColor(info["rgb_obs"], cv2.COLOR_BGR2RGB)
                cv2.imwrite("1_.jpg", im)
                cv2.imwrite("1.jpg", obs[:, :, 0])
            else:
                action[1] = 0.0
            action[1] = np.clip(action[1], -1, 1)
            action[0] = 0.8 if keys[K_UP] or keys[K_w] else 0.0
            '''

            a, state, _ = my_restored_policy.compute_single_action(obs, state, prev_action=a, prev_reward=reward)
            # Take action
            t = threading.Thread(target=env.step, args=(actions[a],))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break

            obs, reward, done, info = env.observation, env.last_reward, env.terminal_state, env.info
            _, r = getmap.method(actions[a], info)

            env.render()
            tmp = obs["obs"]
            obs["obs"] = r

            if out is not None:
                out_img = env.viewer_image['rgb']
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                r = cv2.resize(r, (200, 200), interpolation=cv2.INTER_LINEAR)
                r = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
                r[0, :] = (150, 146, 135)
                r[-1, :] = (150, 146, 135)
                r[:, 0] = (150, 146, 135)
                r[:, -1] = (150, 146, 135)
                gt = cv2.resize(tmp, (200, 200), interpolation=cv2.INTER_LINEAR)
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
                gt[0, :] = (150, 146, 135)
                gt[-1, :] = (150, 146, 135)
                gt[:, 0] = (150, 146, 135)
                gt[:, -1] = (150, 146, 135)
                '''
                RGB_img = cv2.resize(r_, (200, 200), interpolation=cv2.INTER_LINEAR)
                RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_GRAY2RGB)
                RGB_img[0, :] = (150, 146, 135)
                RGB_img[-1, :] = (150, 146, 135)
                RGB_img[:, 0] = (150, 146, 135)
                RGB_img[:, -1] = (150, 146, 135)
                '''
                out_img[50:50 + 200, 100:100 + 200] = gt
                #out_img[270:270 + 200, 100:100 + 200] = RGB_img
                out_img[490:490 + 200, 100:100 + 200] = r
                out.write(out_img)
            avg_reward = env.last_reward

            if done:
                break

        if out is not None:
            out.release()

        print("average reward:", avg_reward / i)

    cap.release()

