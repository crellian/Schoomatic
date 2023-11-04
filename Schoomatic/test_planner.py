import threading
import numpy as np
import yaml
from pygame.locals import *
from getmap import Map
from envs.env import Env
import cv2

#encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/carla/checkpoint/')
if __name__ == "__main__":
    with open("config/carla_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    env = Env(env_config)

    action = np.zeros(env.action_space.shape[0])

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
        getmap = Map()

        if env_config["recording"]:
            out = cv2.VideoWriter(str(vid_idx) + "_rllib.mp4", fourcc, 10.0, (1280, 720))
            vid_idx += 1
        else:
            out = None

        while True:
            # Take action
            t = threading.Thread(target=env.step, args=(action,))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break

            obs, reward, done, info, aux = env.observation, env.last_reward, env.terminal_state, env.info, env.aux
            _, r = getmap.method(info, aux)

            env.render()
            if np.abs(getmap.steer) < 0.5:
                getmap.steer = 0
            else:
                getmap.steer = np.sign(getmap.steer)
            action = [getmap.throttle, getmap.steer]

            if out is not None:
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

            avg_reward = env.last_reward

            if done:
                break
        if out is not None:
            out.release()

        print("Average reached waypoints:", avg_reward / i, avg_reward, i)

    cap.release()
