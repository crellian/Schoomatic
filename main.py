import threading
import numpy as np

import pygame
from pygame.locals import *
from CarlaEnv.getmap_ import Map
from CarlaEnv.carla_lap_env import CarlaLapEnv
import cv2


env_config ={
"addresses": [["192.168.0.21", 2066]
        ],
        "timeout": 10,  # this is in seconds
        "synchronous": False,
        "delta_seconds": -1,
        "fps": -1,
        "server_display": False,
        "debug_mod": False,

        "render_hud": True,
        "rgb_display": True, #if we want rgb_display
        "rgb_viewer_res": (1280, 720), #only if rgb_display is activated if rgb_display is True
        "bev_display": False, #if we want bev display
        "bev_viewer_res": (128, 128), #only if bev_display is activated
        "horizontal_fov": 80.0, #upto you
        "rgb_obs": True, #if we want the fpv observation
        "rgb_obs_res": (84, 84), #only is activated if rgb_obs is true
        "bev_obs": True, #if we want the bev observation
        "bev_obs_res": (64, 64), #only is activated if bev_res is true

        "task_config":
            {
                "max_timesteps": 2500,  # max timesteps for the episode
                "town": "Town05",
                "src_loc": (-67, -91),  # if its None generate randomly
                "dst_loc": (27, -72),  # if its None generate randomly
                "pedestrian_fq": 30.0, #0.0 -> 100.0, how many moving pedestrians on the curb and the crosswalk
                "vehicle_fq": 23.0, #0.0 -> 100.0, how many moving vehicles on the road
                "pedestrian_obstacle_fq": 0.0, #0.0 -> 100.0 how many static pedestrian obstacles in the scene
                "vehicle_obstacle_fq": 8.0, #0.0 -> 100.0 how many static vehicle obstacles in the scene
                "sparse_reward_fn": False, #if its false then implement the reward fn we talked about
                "goal_reward": "propdist", #goal reward proportional to the distance
                "goal_tolerance": 10,
                "terminate_reward": -1000.0, #add an assertion that this reward always has to be negative
                "resolution": 5,
                "sparse_config": #only is activated if sparse_reward_fn is True
                    {
                        "sparse_reward_thresh": 200.0,
                        "sparse_reward_fn": (lambda i: 200.0 + i*200.0),
                    },
                "curriculum": False,
                "cirriculum_config": #only is activated if curriculum is True
                    {
                        "num_goals": 1, #-1 if there is no limit
                        "start_dist": 300,
                        "end_dist": -1,
                        "cirriculum_fn": (lambda i: "start_dist" + 100*i), #where i is the index of the generation
                    }
            },
        "action_smoothing": 0,  # dont worry about this
    }




import random
#encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/carla/checkpoint/')
if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaLapEnv(env_config)

    action = np.zeros(env.action_space.shape[0])
    action_m = np.zeros(env.action_space.shape[0])
    anchors = []
    for i in range(6):
        im = cv2.imread("/home/carla/img2cmd/train/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE).astype(float)
        anchors.append(im)

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    cnt = 0

    actions = [[0, 1],
              [1, 0],
              [0.5, -0.5],
              [0.5, 0.5],
              [0.2, -0.6],
              [0.2, 0.6],
              ]

    manual = False
    while True:
        while True:
            try:
                obs = env.reset()
            except RuntimeError as e:  # disconnected from the server
                print(e)
                env.init_world()  # internal loop until reconnected
            else:
                break
        getmap = Map()
        cnt += 1
        out = cv2.VideoWriter(str(cnt) + "_rllib.mp4", fourcc, 10.0, (1280, 720))
        while True:
            # Take action
            t = threading.Thread(target=env.step, args=(action,))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break

            obs, reward, done, info, aux = env.observation, env.last_reward, env.terminal_state, env.info, env.aux
            l_t, r = getmap.method(info, aux)

            env.render(r, anchors[l_t])
            if np.abs(getmap.steer) < 0.5:
                getmap.steer = 0
            else:
                getmap.steer = np.sign(getmap.steer)
            action = [getmap.throttle, getmap.steer] #actions[l_t]

            out_img = env.viewer_image['rgb']
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            r = cv2.resize(r, (200, 200), interpolation=cv2.INTER_LINEAR)
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
            r[0, :] = (150, 146, 135)
            r[-1, :] = (150, 146, 135)
            r[:, 0] = (150, 146, 135)
            r[:, -1] = (150, 146, 135)
            out_img[100:100 + 200, 100:100 + 200] = r
            RGB_img = cv2.resize(info['rgb_obs'], (200, 200), interpolation=cv2.INTER_LINEAR)
            RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2BGR)
            RGB_img[0, :] = (150, 146, 135)
            RGB_img[-1, :] = (150, 146, 135)
            RGB_img[:, 0] = (150, 146, 135)
            RGB_img[:, -1] = (150, 146, 135)
            out_img[350:350 + 200, 100:100 + 200] = RGB_img
            out.write(out_img)
            print(env.step_count)

            if done:
                break
        out.release()

    cap.release()
