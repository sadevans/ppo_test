import numpy as np
import torch
from torch import nn


class PPO:
    def __init__(self, policy, optimizer, scheduler = None, cliprange=0.2, value_loss_coef=0.25, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def get_policy_loss(self, rollout, act):
        """ Computes and returns policy loss on a given rollout. """

        logprobs_current = act['distribution'].log_prob(torch.tensor(rollout['actions']))

        logprobs_old = torch.tensor(rollout['log_probs'])
        r = (logprobs_current - logprobs_old).exp()
        self.advs = torch.FloatTensor(rollout['advantages'])
        # print(type(r), type(rollout['advantages']))
        loss_cpi = r*self.advs
        # self.advs = rollout['advantages']
        loss_clip = torch.clamp(r, 1 - self.cliprange, 1 + self.cliprange)*self.advs

        return -torch.mean(torch.min(loss_cpi, loss_clip))

    def get_value_loss(self, rollout, act):
        """ Computes and returns value loss on a given rollout. """
        new_vals = act['values']
        target_vals = torch.FloatTensor(rollout['value_targets'])
        vals = rollout['values']

        clipped = torch.clamp(new_vals - vals, -self.cliprange, self.cliprange)
        # print(type(new_vals), type(target_vals))
        surr_1 = (target_vals - new_vals)**2
        surr_2 = (target_vals - vals - clipped)**2

        self.vals = rollout['values']
        return torch.mean(torch.max(surr_1, surr_2))


    def loss(self, rollout):
        act = self.policy.act(rollout["observations"], training=True)

        policy_loss = self.get_policy_loss(rollout, act)
        value_loss = self.get_value_loss(rollout, act)
        self.policy_loss = policy_loss.detach().numpy()
        self.value_loss = value_loss.detach().numpy()
        self.ppo_loss = self.policy_loss + self.value_loss_coef * self.value_loss
        return policy_loss + self.value_loss_coef * value_loss


    def step(self, rollout):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        self.loss(rollout).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        # torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        # self.total_norm = 0