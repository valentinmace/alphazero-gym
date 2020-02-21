import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import scaling_func_inv, compute_value_from_support_torch


class AlphaZeroModel(nn.Module):
    def __init__(self, obs_space, action_space, model_config):
        self.num_actions = action_space.n
        self.obs_dim = obs_space.shape[0]
        self.value_min_val = model_config['value_support_min_val']
        self.value_max_val = model_config['value_support_max_val']
        self.value_support_size = self.value_max_val - self.value_min_val + 1

        nn.Module.__init__(self)

        num_hidden = model_config['num_hidden']
        # Dense and simple neural network as base for both heads
        self.decision_function_shared = nn.Sequential(
            nn.Linear(self.obs_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(num_hidden, self.num_actions)
        self.value_head = nn.Linear(num_hidden, self.value_support_size)

    def forward(self, obs):
        x = self.decision_function_shared(obs)
        logits = self.policy_head(x)
        values_support_logits = self.value_head(x)
        return logits, values_support_logits

    def compute_priors_and_value(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float()
            # TODO: complete here

            # this function takes a batch of numpy observations,
            # computes a batch of priors (as probabilities, not raw logits)
            # and a batch of values (scalar and unscaled with h^{-1})

            prior, value = self.forward(obs)
            prior = F.softmax(prior, dim=1)
            # As describe in MuZero, values must be a distribution of probabilities to later compute value from support
            value = F.softmax(value, dim=1)
            value = compute_value_from_support_torch(value, self.value_min_val, self.value_max_val)
            # We train on downscaled values for env-independancy but the actual values for the tree needs to be upscaled
            value = scaling_func_inv(value, mode='torch')

            # both priors and values returned are numpy as well
            return prior.cpu().numpy(), value.cpu().numpy()