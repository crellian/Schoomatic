import os
import sys
import time
import argparse

os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.experimental.set_visible_devices(devices[1], 'GPU')

import tensorflow_probability as tfp

from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv
from CarlaEnv.carla_lap_env import carla_weather_presets
from AirSimEnv.airsim_lap_env import AirSimLapEnv as AirSimEnv

from vae_common import load_vae, make_encode_state_fn
from ppo import PPO
from reward_functions import reward_functions
from utils import compute_gae, VideoRecorder

from PIL import Image, ImageEnhance


class DRParameters:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        with tf.variable_scope('brightness'):
            self.brightness = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['brightness']['mu'], name='b_mu'),
                tf.Variable(self.hyper_params['brightness']['sigma'], name='b_sigma')
            )
        with tf.variable_scope('contrast'):
            self.contrast = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['contrast']['mu'], name='c_mu'),
                tf.Variable(self.hyper_params['contrast']['sigma'], name='c_sigma'),
            )
        with tf.variable_scope('hue'):
            self.hue = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['hue']['mu'], name='h_mu'),
                tf.Variable(self.hyper_params['hue']['sigma'], name='h_sigma'),
            )

    def sample(self, session):
        return session.run([self.brightness.sample(), self.contrast.sample(), self.hue.sample()])

    def get_trainable_params(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class DomainRandomizer:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.init_vae()
        self.init_cnn()
        self.measurements = set(['steer', 'throttle', 'speed'])
        self.encode_state_fn = make_encode_state_fn(self.measurements)
        self.init_source_env()
        self.init_target_env()
        self.action_space = self.source_env.action_space
        self.num_actions = self.action_space.shape[0]
        self.params = DRParameters(self.hyper_params['dr'])
        self.trainable_params = {
            'b': self.params.get_trainable_params('brightness'),
            'c': self.params.get_trainable_params('contrast'),
            'h': self.params.get_trainable_params('hue'),
        }
        self.best_eval_reward = -np.inf
        self.epochs = self.hyper_params['dr']['epochs']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hyper_params['dr']['learning_rate'],
                                                name='dr_optimizer')
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['b']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['c']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['h']))
        self.init_ppo()
        self.presets = carla_weather_presets[:1]
        if self.hyper_params['env']['source']['weather']:
            self.presets = carla_weather_presets

    def init_source_env(self):
        self.source_env = CarlaEnv(
            host=self.hyper_params['env']['common']['host'],
            obs_res=self.hyper_params['env']['common']['obs_res'],
            encode_state_fn=self.encode_state_fn,
            reward_fn=reward_functions[self.hyper_params['env']['common']['reward_fn']],
            action_smoothing=self.hyper_params['env']['common']['action_smoothing'],
            fps=self.hyper_params['env']['common']['fps'],
            synchronous=self.hyper_params['env']['source']['synchronous'],
            start_carla=self.hyper_params['env']['source']['start_carla']
        )

    def init_target_env(self):
        self.target_env = AirSimEnv(
            host=self.hyper_params['env']['common']['host'],
            obs_res=self.hyper_params['env']['common']['obs_res'],
            encode_state_fn=self.encode_state_fn,
            reward_fn=reward_functions[f"{self.hyper_params['env']['common']['reward_fn']}_airsim"],
            action_smoothing=self.hyper_params['env']['common']['action_smoothing'],
            fps=self.hyper_params['env']['common']['fps'],
            route_file=self.hyper_params['env']['target']['route_file']
        )

    def init_vae(self):
        if self.hyper_params['model']['use_vae']:
            self.vae = load_vae(
                os.path.join('./vae/models', self.hyper_params['model']['vae']['model_name']),
                self.hyper_params['model']['vae']['z_dim'],
                self.hyper_params['model']['vae']['model_type']
            )
        else:
            self.vae = None

    def init_cnn(self):
        if self.hyper_params['model']['use_vae']:
            self.cnn = None
        else:
            self.cnn = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(*self.hyper_params['env']['common']['obs_res'], 3),
                include_top=False,
                pooling='avg'
            )

    def init_ppo(self):
        self.input_shape = np.array([self.vae.z_dim + len(self.measurements)]) if self.vae else np.array(
            [1280 + len(self.measurements)])
        self.model = PPO(
            self.input_shape, self.action_space,
            learning_rate=self.hyper_params['model']['ppo']['learning_rate'],
            lr_decay=self.hyper_params['model']['ppo']['lr_decay'],
            epsilon=self.hyper_params['model']['ppo']['epsilon'],
            initial_std=self.hyper_params['model']['ppo']['initial_std'],
            value_scale=self.hyper_params['model']['ppo']['value_scale'],
            entropy_scale=self.hyper_params['model']['ppo']['entropy_scale'],
            model_dir=os.path.join('./models', self.hyper_params['model']['ppo']['model_name'])
        )

        self.model.init_session()
        self.model.load_latest_checkpoint()
        self.model.write_dict_to_summary('hyperparams/ppo', self.hyper_params['model']['ppo'], 0)
        if self.vae:
            self.model.write_dict_to_summary('hyperparams/vae', self.hyper_params['model']['vae'], 0)
        else:
            self.model.write_dict_to_summary('hyperparams/cnn', {
                'model_name': 'mobilenet_v2',
                'model_type': 'cnn',
                'z_dim': 1280
            }, 0)
        self.model.write_dict_to_summary('hyperparams/general',
                                         {k: self.hyper_params['model'][k] for k in self.hyper_params['model'] if
                                          k != 'ppo' and k != 'vae'}, 0)

    def transform_frame(self, frame, transform_params):
        frame = Image.fromarray(frame)
        brightness = ImageEnhance.Brightness(frame)
        contrast = ImageEnhance.Contrast(frame)
        hue = ImageEnhance.Color(frame)
        frame = brightness.enhance(transform_params[0])
        frame = contrast.enhance(transform_params[1])
        frame = hue.enhance(transform_params[2])
        frame = np.asarray(frame)
        return frame

    def normalize_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        return frame

    def make_state(self, state, for_source_env=True, transform_params=(0.0, 0.0, 0.0), save_frame_idx=-1):
        frame, measurements = state['frame'], state['measurements']
        if for_source_env:
            frame = self.transform_frame(frame, transform_params)
            if save_frame_idx > -1:
                Image.fromarray(frame).save(os.path.join(self.model.image_dir, f'epoch-{save_frame_idx}.png'))
        frame = self.normalize_frame(frame)
        encoded_state = self.vae.encode([frame])[0] if self.vae else \
        self.cnn.predict(np.reshape(frame, (1, *frame.shape)))[0]
        encoded_state = np.append(encoded_state, measurements)
        return encoded_state

    def train(self, idx, transform_params):
        # self.model.reset_episode_idx()
        episodes = self.hyper_params['model']['episodes']
        epochs = self.hyper_params['model']['epochs']
        batch_size = self.hyper_params['model']['batch_size']
        horizon = self.hyper_params['model']['horizon']

        gae_lambda = self.hyper_params['model']['gae_lambda']
        discount_factor = self.hyper_params['model']['discount_factor']

        while episodes <= 0 or (self.model.get_episode_idx() % (episodes + 2)) < episodes:
            episode_idx = (self.model.get_episode_idx() % (episodes + 2))

            for preset in self.presets:
                state, terminal_state, total_reward = self.source_env.reset(), False, 0
                self.source_env.change_weather(preset)
                state = self.make_state(state, transform_params=transform_params)

                print(f"Episode {episode_idx} (Step {self.model.get_train_step_idx()})")
                while not terminal_state:
                    states, taken_actions, values, rewards, dones = [], [], [], [], []
                    for _ in range(horizon):
                        action, value = self.model.predict(state, write_to_summary=True)
                        next_state, reward, terminal_state, info = self.source_env.step(action)
                        next_state = self.make_state(next_state, transform_params=transform_params)
                        if info['closed']:
                            sys.exit(0)
                        self.source_env.extra_info.extend([
                            "Episode {}".format(episode_idx),
                            "Training...",
                            "",
                            "Value:  % 20.2f" % value
                        ])

                        self.source_env.render()
                        total_reward += reward

                        states.append(state)  # [T, *input_shape]
                        taken_actions.append(action)  # [T,  num_actions]
                        values.append(value)  # [T]
                        rewards.append(reward)  # [T]
                        dones.append(terminal_state)  # [T]
                        state = next_state

                        if terminal_state:
                            break

                    _, last_values = self.model.predict(state)

                    advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
                    returns = advantages + values
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    states = np.array(states)
                    taken_actions = np.array(taken_actions)
                    returns = np.array(returns)
                    advantages = np.array(advantages)

                    T = len(rewards)
                    assert states.shape == (T, *self.input_shape)
                    assert taken_actions.shape == (T, self.num_actions)
                    assert returns.shape == (T,)
                    assert advantages.shape == (T,)

                    self.model.update_old_policy()
                    for _ in range(epochs):
                        num_samples = len(states)
                        indices = np.arange(num_samples)
                        np.random.shuffle(indices)
                        for i in range(int(np.ceil(num_samples / batch_size))):
                            begin = i * batch_size
                            end = begin + batch_size
                            if end > num_samples:
                                end = None
                            mb_idx = indices[begin:end]

                            self.model.train(states[mb_idx], taken_actions[mb_idx],
                                             returns[mb_idx], advantages[mb_idx])

            self.model.write_value_to_summary(f"train/reward", total_reward, idx * episodes + episode_idx)
            self.model.write_value_to_summary(f"train/distance_traveled", self.source_env.distance_traveled,
                                              idx * episodes + episode_idx)
            self.model.write_value_to_summary(f"train/average_speed",
                                              3.6 * self.source_env.speed_accum / self.source_env.step_count,
                                              idx * episodes + episode_idx)
            self.model.write_value_to_summary(f"train/center_lane_deviation", self.source_env.center_lane_deviation,
                                              idx * episodes + episode_idx)
            self.model.write_value_to_summary(f"train/average_center_lane_deviation",
                                              self.source_env.center_lane_deviation / self.source_env.step_count,
                                              idx * episodes + episode_idx)
            self.model.write_value_to_summary(f"train/distance_over_deviation",
                                              self.source_env.distance_traveled / self.source_env.center_lane_deviation,
                                              idx * episodes + episode_idx)

            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/reward", total_reward, episode_idx)
            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/distance_traveled", self.source_env.distance_traveled, episode_idx)
            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/average_speed", 3.6 * self.source_env.speed_accum / self.source_env.step_count, episode_idx)
            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/center_lane_deviation", self.source_env.center_lane_deviation, episode_idx)
            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/average_center_lane_deviation", self.source_env.center_lane_deviation / self.source_env.step_count, episode_idx)
            # self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/distance_over_deviation", self.source_env.distance_traveled / self.source_env.center_lane_deviation, episode_idx)

            self.model.write_episodic_summaries()

    def eval(self, idx, in_source_env=True, transform_params=(0.0, 0.0, 0.0)):
        env = self.source_env if in_source_env else self.target_env
        if in_source_env:
            state, terminal, total_reward = env.reset(is_training=False), False, 0
            state = self.make_state(state, transform_params=transform_params, save_frame_idx=idx)
        else:
            state, terminal, total_reward = env.reset(), False, 0
            state = self.make_state(state, for_source_env=False)

        rendered_frame = env.render(mode='rgb_array')

        if not in_source_env:
            filename = os.path.join(self.model.video_dir,
                                    f"epoch-{idx}-drparams-{'-'.join(map(str, transform_params))}.avi")
            video_recorder = VideoRecorder(filename, frame_size=rendered_frame.shape, fps=env.fps)
            video_recorder.add_frame(rendered_frame)

        episode_idx = self.model.get_episode_idx()

        while not terminal:
            if in_source_env:
                env.extra_info.append("Episode {}".format(episode_idx))
                env.extra_info.append("Running eval...".format(episode_idx))
                env.extra_info.append("")

            action, _ = self.model.predict(state, greedy=True)
            state, reward, terminal, info = env.step(action)

            if in_source_env:
                state = self.make_state(state, transform_params=transform_params)
            else:
                state = self.make_state(state, for_source_env=False)

            if info['closed']:
                break

            rendered_frame = env.render(mode='rgb_array')
            if not in_source_env:
                video_recorder.add_frame(rendered_frame)
            total_reward += reward

        if not in_source_env:
            video_recorder.release()

        if info['closed']:
            sys.exit(0)

        if in_source_env:
            print('Source Domain Reward:', total_reward)
            self.model.write_value_to_summary("dr_params/brightness", transform_params[0], idx)
            self.model.write_value_to_summary("dr_params/contrast", transform_params[1], idx)
            self.model.write_value_to_summary("dr_params/hue", transform_params[2], idx)
        else:
            print('Target Domain Reward:', total_reward)
            self.model.write_value_to_summary("dr_params/brightness_mean",
                                              self.model.sess.run(self.params.brightness.loc), idx)
            self.model.write_value_to_summary("dr_params/brightness_std",
                                              self.model.sess.run(self.params.brightness.scale), idx)
            self.model.write_value_to_summary("dr_params/contrast_mean", self.model.sess.run(self.params.contrast.loc),
                                              idx)
            self.model.write_value_to_summary("dr_params/contrast_std", self.model.sess.run(self.params.contrast.scale),
                                              idx)
            self.model.write_value_to_summary("dr_params/hue_mean", self.model.sess.run(self.params.hue.loc), idx)
            self.model.write_value_to_summary("dr_params/hue_std", self.model.sess.run(self.params.hue.scale), idx)

        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/reward", total_reward, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/distance_traveled",
                                          env.distance_traveled, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/average_speed",
                                          3.6 * env.speed_accum / env.step_count, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/center_lane_deviation",
                                          env.center_lane_deviation, idx)
        self.model.write_value_to_summary(
            f"eval/{'source' if in_source_env else 'target'}/average_center_lane_deviation",
            env.center_lane_deviation / env.step_count, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/distance_over_deviation",
                                          env.distance_traveled / env.center_lane_deviation, idx)

        self.model.sess.run([self.model.episode_counter.inc_op])

        return total_reward

    def close(self):
        self.source_env.close()
        self.target_env.close()

    def run(self):
        # self.model.sess.run(tf.variables_initializer(self.optimizer.variables()))
        for idx in range(self.epochs):
            print(f'DR Epoch {idx}')
            transform_params = self.params.sample(self.model.sess)
            print(transform_params)
            self.train(idx, transform_params)
            source_reward = self.eval(idx, transform_params=transform_params)
            target_reward = self.eval(idx, in_source_env=False, transform_params=transform_params)

            if target_reward > self.best_eval_reward:
                self.model.save()
                self.best_eval_reward = target_reward

            self.transfer_loss = (source_reward - target_reward)
            bl = -tf.math.log(self.params.brightness.prob(transform_params[0]) * self.transfer_loss)
            cl = -tf.math.log(self.params.contrast.prob(transform_params[1]) * self.transfer_loss)
            hl = -tf.math.log(self.params.hue.prob(transform_params[2]) * self.transfer_loss)

            bg = tf.gradients(bl, self.trainable_params['b'])
            cg = tf.gradients(cl, self.trainable_params['c'])
            hg = tf.gradients(hl, self.trainable_params['h'])

            self.model.sess.run([
                self.optimizer.apply_gradients(zip(bg, self.trainable_params['b'])),
                self.optimizer.apply_gradients(zip(cg, self.trainable_params['c'])),
                self.optimizer.apply_gradients(zip(hg, self.trainable_params['h'])),
            ])

        with open(os.path.join(self.model.model_dir, 'converged_params.txt'), 'w') as outfile:
            outfile.write(
                f'Brightness ~ Normal(mu={self.model.sess.run(self.params.brightness.loc)}, sigma={self.model.sess.run(self.params.brightness.scale)})\n')
            outfile.write(
                f'Contrast ~ Normal(mu={self.model.sess.run(self.params.contrast.loc)}, sigma={self.model.sess.run(self.params.contrast.scale)})\n')
            outfile.write(
                f'Hue ~ Normal(mu={self.model.sess.run(self.params.hue.loc)}, sigma={self.model.sess.run(self.params.hue.scale)})\n')

        print('Done...')
        self.close()


def init_hyper_params():
    parser = argparse.ArgumentParser(description="Domain Randomization (sim2sim)")

    # DR hyper parameters
    parser.add_argument("--dr_learning_rate", type=float, default=1e-2, help="DR learning rate")
    parser.add_argument("--dr_num_epochs", type=int, default=500, help="DR number of epochs")
    parser.add_argument("--brightness_mean", type=float, default=1.5, help="Initial Distribution Mean (Brightness)")
    parser.add_argument("--brightness_std", type=float, default=0.5, help="Initial Distribution Std Dev (Brightness)")
    parser.add_argument("--contrast_mean", type=float, default=1.5, help="Initial Distribution Mean (Contrast)")
    parser.add_argument("--contrast_std", type=float, default=0.5, help="Initial Distribution Std Dev (Contrast)")
    parser.add_argument("--hue_mean", type=float, default=1.5, help="Initial Distribution Mean (Hue)")
    parser.add_argument("--hue_std", type=float, default=0.5, help="Initial Distribution Std Dev (Hue)")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0,
                        help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")

    # VAE parameters
    parser.add_argument("--vae_model", type=str,
                        default="seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data",
                        help="Trained VAE model to load")
    parser.add_argument("--vae_model_type", type=str, default='cnn', help="VAE model type (\"cnn\" or \"mlp\")")
    parser.add_argument("--vae_z_dim", type=int, default=64, help="Size of VAE bottleneck")
    parser.add_argument("-use_vae", action="store_true", help="If True, use vae, else mobilenet v2")

    # General hyper parameters
    parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=2500,
                        help="Number of episodes to train for (0 or less trains forever)")

    # Common Environment settings
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to connect to")
    parser.add_argument("--fps", type=int, default=15, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.3, help="Action smoothing factor")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See reward_functions.py for more info.")

    # Carla Settings
    parser.add_argument("--synchronous", type=int, default=True,
                        help="Set this to True when running in a synchronous environment")
    parser.add_argument("-weather", action="store_true", help="If True, use all weather presets to train every episode")

    # AirSim Settings
    parser.add_argument("--route_file", type=str, default="./AirSimEnv/routes/dr-test-02.txt",
                        help="Route to use in AirSim")

    parser.add_argument("--model_name", type=str, default=f"dr-model-{int(time.time())}",
                        help="Name of the model to train. Output written to models/model_name")

    params = vars(parser.parse_args())

    return {
        'model': {
            'ppo': {
                'learning_rate': params['learning_rate'],
                'lr_decay': params['lr_decay'],
                'epsilon': params['ppo_epsilon'],
                'initial_std': params['initial_std'],
                'value_scale': params['value_scale'],
                'entropy_scale': params['entropy_scale'],
                'model_name': params['model_name']
            },
            'vae': {
                'model_name': params['vae_model'],
                'model_type': params['vae_model_type'],
                'z_dim': params['vae_z_dim']
            },
            'horizon': params['horizon'],
            'epochs': params['num_epochs'],
            'episodes': params['num_episodes'],
            'batch_size': params['batch_size'],
            'gae_lambda': params['gae_lambda'],
            'discount_factor': params['discount_factor'],
            'use_vae': params['use_vae']
        },
        'env': {
            'common': {
                'host': params['host'],
                'fps': params['fps'],
                'action_smoothing': params['action_smoothing'],
                'reward_fn': params['reward_fn'],
                'obs_res': (160, 80) if params['use_vae'] else (160, 160)
            },
            'source': {
                'synchronous': params['synchronous'],
                'start_carla': False,
                'weather': params['weather']
            },
            'target': {
                'route_file': params['route_file']
            }
        },
        'dr': {
            'brightness': {
                'mu': params['brightness_mean'],
                'sigma': params['brightness_std']
            },
            'contrast': {
                'mu': params['contrast_mean'],
                'sigma': params['contrast_std']
            },
            'hue': {
                'mu': params['hue_mean'],
                'sigma': params['hue_std']
            },
            'epochs': params['dr_num_epochs'],
            'learning_rate': params['dr_learning_rate']
        }
    }


if __name__ == '__main__':
    hyper_params = init_hyper_params()
    dr = DomainRandomizer(hyper_params)
    try:
        dr.run()
    except Exception as e:
        dr.source_env.close()
        raise e
