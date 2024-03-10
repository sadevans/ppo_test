import numpy as np
import torch
from collections import defaultdict


class Runner():
  def __init__(self, env, policy, writer=None, num_runner_steps=512, num_epochs=10, gamma=0.99, lambda_=0.95, num_minibatches=32, step_var=None, render=False):
    self.env = env
    self.policy = policy
    self.num_runner_steps = num_runner_steps # N from algo
    self.gamma = gamma
    self.lambda_ = lambda_
    self.num_minibatches = num_minibatches # B from algo
    self.num_epochs = num_epochs
    self.step_var = step_var if step_var is not None else 0

    self.render = render
    self.images = []

    res = self.env.reset()
    self.len_reset = len(res)
    if len(res) == 2:
      self.latest_observation = res[0]
    elif len(res) == 4:
      self.latest_observation = res

    self.minibatch_count = 0
    self.epoch_count = 0
    self.score = 0
    self.rollout = None

    self.writer = writer

  @property
  def nenvs(self):
      """ Returns number of batched envs or `None` if env is not batched """
      return getattr(self.env.unwrapped, "nenvs", None)

  def normalize_advantages(self, advantages):
    return (advantages- advantages.mean()) / (advantages.std() + 1e-10)

  def toarray(self, rollout):
    for key, value in filter(lambda kv: kv[0] != 'latest_observation' and kv[0] != 'env_steps', rollout.items()):
        rollout[key] = np.asarray(value)

  def gae(self, rollout):
    """GAE оценка"""

    value_target = torch.tensor(self.policy.act(rollout['latest_observation'])['values'], dtype=torch.float32)

    env_steps = rollout['env_steps']

    rewards = torch.tensor(rollout['rewards'], dtype=torch.float32)
    dones = torch.tensor(rollout['dones'], dtype=torch.float32)
    undones = 1 - dones
    rollout['values'] = torch.tensor(rollout['values'],dtype=torch.float32)
    rollout['advantages'] = []
    rollout['value_targets'] = []

    advantages = np.zeros_like(rollout['rewards'], dtype=np.float32)
    value_targets = np.zeros_like(rollout['rewards'], dtype=np.float32)

    gae = 0
    for step in reversed(range(env_steps)):
        if step==env_steps - 1:
          # print(type(value_target), type(undones[step]))
          delta = rewards[step] + self.gamma*value_target*undones[step] - rollout['values'][step]
        else:
          # print(type(rollout['values'][step + 1]), type(undones[step]))
          delta = rewards[step] + self.gamma*rollout['values'][step + 1]*undones[step] - rollout['values'][step]

        gae = delta + self.gamma*self.lambda_*undones[step]*gae

        value_target = rollout['values'][step] # added line

        # rollout['advantages'].insert(0, gae)
        # rollout['value_targets'].insert(0, gae + rollout['values'][step])
        advantages[step] = gae
        value_targets[step] = gae + rollout['values'][step]
    # rollout['advantages'] = torch.tensor(rollout['advantages'], dtype=torch.float32)
    # rollout['value_targets'] = torch.tensor(rollout['value_targets'], dtype=torch.float32)

    rollout['advantages'] = advantages
    rollout['value_targets'] = value_targets


  def shuffle_rollout(self):
    """ Shuffles all elements in trajectory.

    Should be called at the beginning of each epoch.
    """
    rollout_len = self.rollout["observations"].shape[0]
    # print('ROLLOUT LEN: ', rollout_len)

    permutation = np.random.permutation(rollout_len)
    # print('PERMUTATION: ', permutation)
    # print('ROLLOUT ITEMS: ', self.rollout.items())
    for key, value in self.rollout.items():
        if key != 'latest_observation' and key != 'env_steps':
            self.rollout[key] = value[permutation]


  def get_minibatch(self):
    """Сэмплируем минибатч заданного размера из траектории"""
    if not self.rollout:
      self.images = []
      self.rollout = self.take_step()
      # self.rollout['advantages'] = self.normalize_advantages(self.rollout['advantages'])


    if self.minibatch_count == self.num_minibatches:
      self.shuffle_rollout()
      self.minibatch_count = 0
      self.epoch_count += 1

    if self.epoch_count == self.num_epochs:
      self.images = []
      self.rollout = self.take_step()
      self.shuffle_rollout()
      self.minibatch_count = 0
      self.epoch_count = 0

    rollout_len = self.rollout["observations"].shape[0]


    batch_size = rollout_len//self.num_minibatches
    minibatch = {}

    for key, value in self.rollout.items():
      if key != 'latest_observation' and key != 'env_steps':
        minibatch[key] = value[self.minibatch_count*batch_size: (self.minibatch_count + 1)*batch_size]
    self.minibatch_count += 1
    self.normalize_advantages(minibatch['advantages'])

    return minibatch


  def take_step(self):
    """Запускает агента в среде, собирая при этом роллаут длины N"""
    rollout = defaultdict(list, {"actions": []}) # пустая траектория
    observations = []
    rewards = []
    dones = []
    self.env_steps = self.num_runner_steps

    for step in range(self.num_runner_steps):
      observations.append(self.latest_observation) # не забываем добавить последнее наблюдение в массив
      act = self.policy.act(self.latest_observation) # действие от последнего наблюдения

      # получаем все штуки и засовываем в траекторию
      for key, val in act.items():
        rollout[key].append(val)

      obs, reward, terminated, truncated, _ = self.env.step(rollout['actions'][-1]) # шаг на основе последнего действия, чтобы получить все штуки из среды
      if self.render:
        self.images.append(self.env.render())

      if not self.self.writer:
        self.self.writer.add_scalar("trajectory/step_reward", reward, step)
        self.writer.add_scalar("trajectory/position", obs[0], step)
        self.writer.add_scalar("trajectory/theta", obs[1], step)
        self.writer.add_scalar("trajectory/velocity", obs[2], step)
        self.writer.add_scalar("trajectory/angular_velocity", obs[3], step)

      self.latest_observation = obs
      self.score += reward # added
      rewards.append(reward)
      done = np.logical_or(terminated, truncated)
      dones.append(done)

      self.step_var += self.nenvs or 1

      if not self.nenvs and np.all(done): # не дошли ли до конца
        self.env_steps = step + 1
        if self.len_reset == 2: self.latest_observation = self.env.reset()[0]
        elif self.len_reset == 4: self.latest_observation = self.env.reset()

      rollout.update(observations=observations, rewards=rewards, dones=dones)
      rollout['latest_observation'] = self.latest_observation
      rollout['env_steps'] = self.env_steps

    self.writer.flush()
    self.toarray(rollout)
    self.gae(rollout)
    # self.normalize_advantages(rollout['advantages'])

    return rollout