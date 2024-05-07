import abc
import gym
from gym import spaces
from pygame import locals
import numpy as np
import deepmind_lab
import cv2


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DeepmindLabEnvironment(gym.Env):

    def __init__(self,
                 level,
                 configs,
                 observation_keys,
                 height=84,
                 width=84,
                 frame_skip=4,
                 fps=60,
                 ):
        super(DeepmindLabEnvironment, self).__init__()
        self.level = level
        self.viewer = None
        self.frame_skip = frame_skip
        config = {
            'fps': str(fps),
            'width': str(width),
            'height': str(height),
        }
        config.update(configs)
        self.lab = deepmind_lab.Lab(level, observation_keys, config=config)

    @abc.abstractmethod
    def reset(self, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    def close(self):
        self.lab.close()

    @abc.abstractmethod
    def render(self, mode='human'):
        pass

    def observation_space(self):
        return self.lab.observation_spec()

    def action_space(self):
        return self.lab.action_spec()


class DeepmindLabMazeNavigationEnvironment(DeepmindLabEnvironment):

    NAV_ACTION_MEANING = {
        0: "look_left",
        1: "look_right",
        2: "strafe_left",
        3: "strafe_right",
        4: "forward",
        5: "backward"
    }

    NAV_KEY_TO_ACTION = {
        (locals.K_4,): 0,
        (locals.K_6,): 1,
        (locals.K_LEFT,): 2,
        (locals.K_RIGHT,): 3,
        (locals.K_UP,): 4,
        (locals.K_DOWN,): 5
    }

    ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        _action(0, 0, 0, 0, 1, 0, 0),  # fire
        _action(0, 0, 0, 0, 0, 1, 0),  # jump
        _action(0, 0, 0, 0, 0, 0, 1),  # crouch
    ]

    DEFAULT_ACTION = _action(0, 0, 0, 0, 0, 0, 0)

    def __init__(self, level, width=84, height=84, frame_skip=4, fps=60, enable_depth=False, other_configs=None, other_obs=None):
        configs = {}
        if other_configs:
            configs.update(other_configs)
        obs = ['RGBD_INTERLEAVED'] if enable_depth else ['RGB_INTERLEAVED']
        if other_obs:
            obs.extend(other_obs)
        self._obs_key = obs[-1]
        self._enable_depth = enable_depth
        super(DeepmindLabMazeNavigationEnvironment, self).__init__(level, configs, obs, height, width, frame_skip, fps)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Seven action dimensions
            dtype=np.float32,
        )

        obs_shape = [height, width, 4 if enable_depth else 3]
        self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)

        self.last_state = None

    def get_action_meanings(self):
        return [self.NAV_ACTION_MEANING[i] for i in range(self.action_space.shape[0])]

    def _prepare_for_rgb(self, img):
        if self._enable_depth:
            img = img[:, :, :3]
        return img

    def render(self, mode='human'):
        rgb_or_rgbd = self.lab.observations()[self._obs_key]
        if mode == 'rgb_array':
            return self._prepare_for_rgb(rgb_or_rgbd)
        elif mode == 'rgbd_array':
            assert self._enable_depth, 'Please enable depth output when initializing the object'
            return rgb_or_rgbd
        elif mode == 'human':
            img = self._prepare_for_rgb(rgb_or_rgbd)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Deepmind Lab', img_bgr)
            cv2.waitKey(1)
        else:
            raise ValueError("Unsupported render mode")

    def step(self, action):
        if np.any(np.isnan(action)):
            action = self.DEFAULT_ACTION
        else:
            real_action = np.clip(action, -1, 1) * np.array([20, 10, 1, 1, 1, 1, 1])
        
        reward = self.lab.step(real_action, num_steps=self.frame_skip)
        terminated = not self.lab.is_running()

        if terminated:
            state = np.copy(self.last_state) if self.last_state is not None else None
            info = {}
        else:
            obs = self.lab.observations()
            state = obs[self._obs_key]
            self.last_state = state
            info = {key: val for key, val in obs.items() if key != self._obs_key}

        return state, reward, terminated, info

    def reset(self, **kwargs):
        self.lab.reset(seed=None)
        obs = self.lab.observations()
        state = np.copy(obs[self._obs_key])
        self.last_state = state
        return state

    @staticmethod
    def get_keys_to_action():
        return DeepmindLabMazeNavigationEnvironment.NAV_KEY_TO_ACTION


    def close(self):
        self.lab.close()



class ContinuousDeepmindLabEnvironment(gym.Env):
    def __init__(self, level, configs, observation_keys, height=84, width=84, frame_skip=1, fps=60):
        super().__init__()
        self.frame_skip = frame_skip
        self.lab = deepmind_lab.Lab(level, observation_keys, config={
            'fps': str(fps),
            'width': str(width),
            'height': str(height),
        })
        
        # Define continuous action space with seven dimensions
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Seven action dimensions
            dtype=np.float32,
        )

        # Observation space setup
        self.observation_space = gym.spaces.Box(
            0, 255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

    def reset(self):
        self.lab.reset(seed=None)
        obs = self.lab.observations()
        state = obs['RGB_INTERLEAVED']
        return state

    def step(self, action):
        # Assuming the environment accepts continuous actions directly
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.lab.step(action)  # Using the continuous action as provided

        terminated = not self.lab.is_running()

        obs = self.lab.observations()
        state = obs['RGB_INTERLEAVED'] if not terminated else None

        return state, reward, terminated, {}

    def render(self, mode='human'):
        if mode == 'human':
            img = self.lab.observations().get('RGB_INTERLEAVED', None)
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Deepmind Lab", img_bgr)
                cv2.waitKey(1)

    def close(self):
        self.lab.close()
