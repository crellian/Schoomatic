import threading
import sys
import os

from wrappers import *
from pathlib import Path
from env import CarlaLapEnv

root_dir = "/home2/carla/dataset/"

tasks = [
    {"host": "192.168.0.192", "port": 2117, "task_name": "Town01_0", "town": "Town01", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.192", "port": 2132, "task_name": "Town01_1", "town": "Town01", "src_loc": None,
     "dst_loc": None}, #iGpu2
    {"host": "192.168.0.192", "port": 2159, "task_name": "Town02_0", "town": "Town02", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.192", "port": 2240, "task_name": "Town02_1", "town": "Town02", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.192", "port": 2066, "task_name": "Town03_0", "town": "Town03", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.192", "port": 2069, "task_name": "Town03_1", "town": "Town03", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.184", "port": 2036, "task_name": "Town04_0", "town": "Town04", "src_loc": None,
     "dst_loc": None},#iGpu3
    {"host": "192.168.0.184", "port": 2042, "task_name": "Town04_1", "town": "Town04", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.184", "port": 2291, "task_name": "Town05_0", "town": "Town05", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.184", "port": 2339, "task_name": "Town05_1", "town": "Town05", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.183", "port": 2342, "task_name": "Town06_0", "town": "Town06", "src_loc": None,
     "dst_loc": None}, #iGpu4
    {"host": "192.168.0.21", "port": 2003, "task_name": "Town06_1", "town": "Town06", "src_loc": None,
     "dst_loc": None}, #iGpu25
    {"host": "192.168.0.183", "port": 2345, "task_name": "Town07_0", "town": "Town07", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.182", "port": 2153, "task_name": "Town07_1", "town": "Town07", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.183", "port": 2342, "task_name": "Town10_0", "town": "Town10HD", "src_loc": None,
     "dst_loc": None},
    {"host": "192.168.0.21", "port": 2000, "task_name": "Town10_1", "town": "Town10HD", "src_loc": None,
     "dst_loc": None},
]

env_config = {
    "addresses": [["127.0.0.1", 2000]
                  ],
    "timeout": 10,  # this is in seconds
    "synchronous": False,
    "delta_seconds": -1,
    "fps": 30,
    "server_display": False,

    "render_hud": True,
    "rgb_display": True,  # if we want rgb_display
    "rgb_viewer_res": (1280, 720),  # only if rgb_display is activated if rgb_display is True
    "bev_display": False,  # if we want bev display
    "bev_viewer_res": (128, 128),  # only if bev_display is activated
    "horizontal_fov": 80.0,  # upto you
    "rgb_obs": True,  # if we want the fpv observation
    "rgb_obs_res": (256, 256),  # only is activated if rgb_obs is true
    "bev_obs": True,  # if we want the bev observation
    "bev_obs_res": (84, 84),  # only is activated if bev_res is true

    "task_config":
        {
            "max_timesteps": 1000,  # max timesteps for the episode
            "town": "Town06",
            "src_loc": (-27, -4),  # if its None generate randomly
            "dst_loc": (-44, -15),  # if its None generate randomly
            "pedestrian_fq": 30.0,  # 0.0 -> 100.0, how many moving pedestrians on the curb and the crosswalk
            "vehicle_fq": 23.0,  # 0.0 -> 100.0, how many moving vehicles on the road
            "pedestrian_obstacle_fq": 0.0,  # 0.0 -> 100.0 how many static pedestrian obstacles in the scene
            "vehicle_obstacle_fq": 8.0,  # 0.0 -> 100.0 how many static vehicle obstacles in the scene
            "sparse_reward_fn": False,  # if its false then implement the reward fn we talked about
            "goal_reward": "propdist",  # goal reward proportional to the distance
            "goal_tolerance": 1,
            "terminate_reward": -1000.0,  # add an assertion that this reward always has to be negative
            "sparse_config":  # only is activated if sparse_reward_fn is True
                {
                    "sparse_reward_thresh": 200.0,
                    "sparse_reward_fn": (lambda i: 200.0 + i * 200.0),
                },
            "curriculum": False,
            "cirriculum_config":  # only is activated if curriculum is True
                {
                    "num_goals": 1,  # -1 if there is no limit
                    "start_dist": 300,
                    "end_dist": -1,
                    "cirriculum_fn": (lambda i: "start_dist" + 100 * i),  # where i is the index of the generation
                }
        },
    "action_smoothing": 0,  # dont worry about this
}

if __name__ == "__main__":
    i = int(sys.argv[1])
    task_name = tasks[i]["task_name"]
    env_config["addresses"][0][0] = tasks[i]["host"]
    env_config["addresses"][0][1] = tasks[i]["port"]
    env_config["task_config"]["town"] = tasks[i]["town"]
    env_config["task_config"]["src_loc"] = tasks[i]["src_loc"]
    env_config["task_config"]["dst_loc"] = tasks[i]["dst_loc"]

    if not os.path.exists(root_dir + task_name + "/5/0"):
        Path(root_dir + task_name + "/5/0").mkdir(parents=True, exist_ok=True)

    observations = []
    observations_rgb = []
    actions = []
    rewards = []
    terminals = []

    env = CarlaLapEnv(env_config)
    action = np.zeros(env.action_space.shape[0])

    total = 0
    while total < 1000 * 1000:
        while True:
            try:
                env.reset()
            except RuntimeError as e:  # disconnected from the server
                print(e)
                env.init_world()  # internal loop until reconnected
            else:
                break

        obs_per_episode = []
        obs_rgb_per_episode = []
        ac_per_episode = []
        rwd_per_episode = []
        term_per_episode = []
        while True:
            action = [random.uniform(0, 1), random.uniform(-1, 1)]  # random actions
            t = threading.Thread(target=env.step, args=(action,))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break

            env.render()

            obs, reward, done, info = env.observation, env.last_reward, env.terminal_state, env.info
            obs_per_episode.append(obs)
            obs_rgb_per_episode.append(info["rgb_obs"])
            ac_per_episode.append(action)
            rwd_per_episode.append(reward)
            if done:
                term_per_episode.append(True)
                break
            else:
                term_per_episode.append(False)

        if term_per_episode[-1] == True:  # episode terminated normally
            observations += obs_per_episode[100:]
            observations_rgb += obs_rgb_per_episode[100:]
            actions += ac_per_episode[100:]
            rewards += rwd_per_episode[100:]
            terminals += term_per_episode[100:]
            print("step: ", len(terminals))


        if len(terminals) > 1000:
            total += len(terminals)

            np.save(root_dir + task_name + "/5/0/observation_" + str(total), np.array(observations))
            np.save(root_dir + task_name + "/5/0/observation_rgb_" + str(total), np.array(observations_rgb))
            np.save(root_dir + task_name + "/5/0/action_" + str(total), np.array(actions))
            np.save(root_dir + task_name + "/5/0/reward_" + str(total), np.array(rewards))
            np.save(root_dir + task_name + "/5/0/terminal_" + str(total), np.array(terminals))

            observations = []
            observations_rgb = []
            actions = []
            rewards = []
            terminals = []

# merge records
for root, subdirs, files in os.walk(root_dir + task_name):
    if files:
        bevs = None
        rgbs = None
        actions = None
        terminals = None
        rewards = None
        for f in files:
            if 'action' in f:
                if bevs is None:
                    bevs = np.load(os.path.join(root, "observation"+f[6:]))
                    os.remove(os.path.join(root, "observation"+f[6:]))
                else:
                    bevs = np.concatenate((bevs, np.load(os.path.join(root, "observation"+f[6:]))))
                    os.remove(os.path.join(root, "observation"+f[6:]))

                if rgbs is None:
                    rgbs = np.load(os.path.join(root, "observation_rgb"+f[6:]))
                    os.remove(os.path.join(root, "observation_rgb"+f[6:]))
                else:
                    rgbs = np.concatenate((rgbs, np.load(os.path.join(root, "observation_rgb"+f[6:]))))
                    os.remove(os.path.join(root, "observation_rgb"+f[6:]))

                if actions is None:
                    actions = np.load(os.path.join(root, "action"+f[6:]))
                    os.remove(os.path.join(root, "action"+f[6:]))
                else:
                    actions = np.concatenate((actions, np.load(os.path.join(root, "action"+f[6:]))))
                    os.remove(os.path.join(root, "action"+f[6:]))

                if terminals is None:
                    terminals = np.load(os.path.join(root, "terminal"+f[6:]))
                    os.remove(os.path.join(root, "terminal"+f[6:]))
                else:
                    terminals = np.concatenate((terminals, np.load(os.path.join(root, "terminal"+f[6:]))))
                    os.remove(os.path.join(root, "terminal"+f[6:]))

                if rewards is None:
                    rewards = np.load(os.path.join(root, "reward"+f[6:]))
                    os.remove(os.path.join(root, "reward"+f[6:]))
                else:
                    rewards = np.concatenate((rewards, np.load(os.path.join(root, "reward"+f[6:]))))
                    os.remove(os.path.join(root, "reward"+f[6:]))

        np.save(os.path.join(root, 'observation.npy'), bevs)
        np.save(os.path.join(root, 'observation_rgb.npy'), rgbs)
        np.save(os.path.join(root, 'reward.npy'), rewards)
        np.save(os.path.join(root, 'terminal.npy'), terminals)
        np.save(os.path.join(root, 'action.npy'), actions)