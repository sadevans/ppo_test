import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from environment.environment import InvertedPendulumEnv
from tqdm import tqdm
import os
import imageio
from utils.runner import Runner
from utils.wrappers import *
from models.net import Net
from models.policy import Policy
from agents.PPO import PPO


if __name__=='__main__':
    env_test = Normalize(Summaries(InvertedPendulumEnv(render_mode="single_rgb_array")))
    seed = 1
    env_test.seed(seed)
    env_test.action_space.seed(seed)
    env_test.observation_space.seed(seed)
    obs = env_test.reset(seed=seed)
    plt.imshow(env_test.render())

    model_name = 'model_10000 (4).pth'
    model_path = f'./model_zoo/{model_name}'
    #model_path = f'/content/{model_name}'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_test = Net(env_test.observation_space.shape[0], env_test.action_space.shape[0])
    model_test.load_state_dict(torch.load(model_path, map_location=device))
    policy_test = Policy(model_test)
    writer = SummaryWriter()

    images = []

    for i in range(3000):
        action = policy_test.act(obs)["actions"]
        obs, r, d, _, _ = env_test.step(action)
        writer.add_scalar("x/step", obs[0], i)
        writer.add_scalar("theta/step", obs[1], i)
        writer.add_scalar("force/step", action[0], i)
        writer.add_scalar("reward/step", r, i)
        images.append(env_test.render())

    writer.flush()
    writer.close()

    imageio.mimsave(f'videos/video_test_{model_name[:-4]}.gif', [np.array(img) for i, img in enumerate(images)], fps=30)