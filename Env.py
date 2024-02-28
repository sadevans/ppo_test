import numpy as np

# import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

import matplotlib.pyplot as plt
from pathlib import Path

class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        #self.render_mode == "rgb_array"
        MujocoEnv.__init__(
            self,
            model_path="model.xml",
            frame_skip=2,
            observation_space=observation_space,
            **kwargs
        )
        self.last_ob = None
        self.env_name = 'InvertedPendulumEnv'

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[1] = 3.14 # Set the pole to be facing down
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


if __name__ == '__main__':
    env = InvertedPendulumEnv(render_mode = "single_rgb_array")
    env.reset_model()
    plt.imshow(env.render())
    plt.show()