import numpy as np
import os
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import matplotlib.pyplot as plt


def get_reward(ob, a):
    reward = 0
    theta = np.mod(ob[1], 2*np.pi) # [0; 2*pi]
    theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]
    if abs(ob[0]) > 0.8:
      out_of_bound = 1
    else:
      out_of_bound = 0
    x = abs(ob[0]) # ob[0] in [-1.1, 1.1]
    x_change_reward = -x ** 2
    reward += 0.3 * x_change_reward
    if abs(theta) < 0.2:
      reward += 50
    # else:
      # reward = 2.5 * np.cos(theta) - (ob[3])**2 - 0.5*a[0]**2 - 10*out_of_bound
    reward = 5 * np.cos(theta) - 0.3*ob[0]**2 - 0.1*a[0]**2 - 10*out_of_bound

    return reward

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
        MujocoEnv.__init__(
            self,
            model_path=os.path.abspath('config/model.xml'),
            frame_skip=2,
            observation_space=observation_space,
            **kwargs
        )
        self.last_ob = None
        self.env_name = 'InvertedPendulumEnv'

    def step(self, a):
        # reward = 1.0
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())

        if self.render_mode == "human":
            self.render()

        # added for reward
        # reward, terminated = get_reward(ob, a)
        reward = get_reward(ob, a)
        # reward = 0
        # reward_bounds = 0
        # stability_reward = -1
        # reward_swing_up = 0
        # bound_reward = 0

        # bound_coef = 200
        # stability_coef = 10

        # theta = ob[1]
        # # print(theta)
        # x = abs(ob[0])
        # theta = np.mod(theta, 2*np.pi) # [0; 2*pi]
        # theta = (theta - 2*np.pi) if theta > np.pi else theta # [-pi; pi]

        # # pole stability
        # if abs(theta) <= 0.4:
        #     stability_reward = 1
        # reward += stability_coef * stability_reward

        # reward += 1 - np.abs(theta)/np.pi


        # # change cart position reward
        # reward += 0.5 * (-x**2)

        # # if abs(theta) <= np.pi:
        
        #     # reward_swing_up = 1 - theta/np.pi

        # # reward += 3 * reward_swing_up


        # # force
        # # force_reward = abs(a[0])
        # # reward += 0.1 * force_reward

        # if x > 0.8:
        #     bound_reward = -1
        # reward += bound_coef * bound_reward



        # # print('NEW THETA: ', theta)
        # # if theta < 1:
        # # reward += (1 - np.cos(theta))
        # # if theta == -np.pi:
        # #     reward += 100
        # # elif theta == np.pi:
        # #     reward -= 400
        # # elif theta <= np.pi/2:
        # #     reward += 2
        # # print(reward)
        # # if abs(x) > 0.8:
        # #     reward -= 400
        # # if abs(theta) <= 0.2:
        # #     reward += 200
            
        # # if x > 0.8:
        # #     reward_bounds = -1
        # #     terminated = True

        # # reward += bound_reward_coef * reward_bounds
            
        # # terminated
        # # if abs(theta) > 0.2:
        # #     terminated = True

        # if x > 0.8:
        #   terminated = True

        # # if abs(ob[3]) > 
        # #   reward -= 200
        # #   reward = reward - 400

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