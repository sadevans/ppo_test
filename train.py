import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from environment.environment import InvertedPendulumEnv
from tqdm import tqdm
import os
from utils.runner import Runner
from utils.wrappers import *
from models.net import Net
from models.policy import Policy
from agents.PPO import PPO


if __name__ == '__main__':
    env = Normalize(Summaries(InvertedPendulumEnv(render_mode="single_rgb_array")))
    env.reset()
    seed = 10
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


    model = Net(env.observation_space.shape[0], env.action_space.shape[0])
    # model = PolicyModel(env.observation_space.shape[0], env.action_space.shape[0])

    policy = Policy(model)

    # optimizer = torch.optim.Adam(policy.model.parameters(), lr = 3e-4, eps=1e-5) # lr=1e-3
    optimizer = torch.optim.Adam(policy.model.parameters(), lr = 1e-3, eps=1e-5) # lr=1e-3

    # epochs = 250000
    # epochs = 20000
    epochs = 10000

    print("observation space: ", env.observation_space,
        "\nobservations:", env.reset())
    print("action space: ", env.action_space,
        "\naction_sample: ", env.action_space.sample())

    writer = SummaryWriter()

    runner = Runner(env, policy, writer=writer, render=True)

    lr_mult = lambda epoch: (1 - (epoch/epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)

    ppo = PPO(policy, optimizer)

    num_steps = []
rewards = []
images = []

for epoch in tqdm(range(epochs)):
    trajectory = runner.get_minibatch()

    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f"./checkpoints/model_{epoch + 1}.pth")
    ppo.step(trajectory)
    # print(env.render())
    # img = env.render()
    # images.append(img)
    # plt.imshow(env.render())
    # plt.show()
    sched.step()


# imageio.mimsave('video_new.mp4', [np.array(img) for i, img in enumerate(images)], fps=30)
num = os.listdir('./model_zoo/')
torch.save(model.state_dict(), f"./model_zoo/model_{num}.pth")