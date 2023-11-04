import os
import numpy as np

root_dir = "/home2/kiran/rgb_bev/"

# merge records
for root, subdirs, files in os.walk(root_dir):
    if 'observation_0.npy' in files:
        bevs = None
        rgbs = None
        actions = None
        terminals = None
        rewards = None

        for f in files:
            print(f)
            if 'observation' in f and 'rgb' not in f:
                if bevs is None:
                    bevs = np.load(os.path.join(root, f))
                else:
                    bevs = np.concatenate((bevs, np.load(os.path.join(root, f))))
            elif 'action' in f:
                if actions is None:
                    actions = np.load(os.path.join(root, f))
                else:
                    actions = np.concatenate((actions, np.load(os.path.join(root, f))))
            elif 'terminal' in f:
                if bevs is None:
                    terminals = np.load(os.path.join(root, f))
                else:
                    terminals = np.concatenate((terminals, np.load(os.path.join(root, f))))
            elif 'reward' in f:
                if rewards is None:
                    rewards = np.load(os.path.join(root, f))
                else:
                    rewards = np.concatenate((rewards, np.load(os.path.join(root, f))))


        np.save(os.path.join(root, 'observation.npy'), bevs)
        np.save(os.path.join(root, 'reward.npy'), rewards)
        np.save(os.path.join(root, 'terminal.npy'), terminals)
        np.save(os.path.join(root, 'action.npy'), actions)
        del bevs
        del actions
        del rewards
        del terminals
        for f in files:
            print(f)
            if 'rgb' in f:
                if rgbs is None:
                    rgbs = np.load(os.path.join(root, f))
                else:
                    rgbs = np.concatenate((rgbs, np.load(os.path.join(root, f))))
        np.save(os.path.join(root, 'observation_rgb.npy'), rgbs)