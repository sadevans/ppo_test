import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.normal import Normal


class Policy:
  def __init__(self, model):
      self.model = model

  def act(self, inputs, training=False):
      inputs = torch.FloatTensor(inputs)
      if inputs.ndim < 2:
          inputs = inputs.unsqueeze(0)

      means, var = self.model.get_policy(inputs)
      cov_mat = torch.diag_embed(var)
      distr = MultivariateNormal(means, cov_mat)

      actions = distr.sample()
      log_probs = distr.log_prob(actions)

      # print('ACTION, LOG PROBS: ', actions, log_probs)
      values = self.model.get_value(inputs)

      if training: return {'distribution': distr, 'values': values}
      else: return {'actions': actions.cpu().numpy().tolist()[0],
                  'log_probs': log_probs[0].detach().cpu().numpy(),
                  'values': values[0].detach().cpu().numpy()}