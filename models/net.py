import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn import functional as F

class Net(nn. Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.h = 64

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h,  output_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 1)
        )

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)

    def get_policy(self, inputs):
      if isinstance(inputs, np.ndarray):
        inputs = torch.FloatTensor(inputs)
      means = self.policy_net(inputs)
      var = F.softplus(means)
      return means, var

    def get_value(self, inputs):
      if isinstance(inputs, np.ndarray):
        inputs = torch.FloatTensor(inputs)
      out = self.value_net(inputs)
      return out

    def forward(self, inputs):
      if isinstance(inputs, np.ndarray):
        inputs = torch.FloatTensor(inputs)
      policy = self.get_policy(inputs)
      value = self.get_value(inputs)

      return policy, value