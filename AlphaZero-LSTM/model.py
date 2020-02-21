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
        # LSTM neural network as base for both heads
        self.decision_function_shared = nn.LSTMCell(input_size=self.obs_dim, hidden_size=num_hidden)
        self.policy_head = nn.Linear(num_hidden, self.num_actions)
        self.value_head = nn.Linear(num_hidden, self.value_support_size)

    def forward(self, obs, hidden_state, cell_state):
        h, c = self.decision_function_shared(obs, (hidden_state, cell_state))
        logits = self.policy_head(h)
        values_support_logits = self.value_head(h)
        return logits, values_support_logits, h, c

    def compute_priors_and_value(self, obs, hidden_state, cell_state):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float()
            # TODO: complete here

            # this function takes a batch of numpy observations,
            # computes a batch of priors (as probabilities, not raw logits)
            # and a batch of values (scalar and unscaled with h^{-1})

            prior, value, h, c = self.forward(obs, hidden_state, cell_state)
            prior = F.softmax(prior, dim=1)
            # As describe in MuZero, values must be a distribution of probabilities to later compute value from support
            value = F.softmax(value, dim=1)
            value = compute_value_from_support_torch(value, self.value_min_val, self.value_max_val)
            # We train on downscaled values for env-independancy but the actual values for the tree needs to be upscaled
            value = scaling_func_inv(value, mode='torch')

            # both priors and values returned are numpy as well
            return prior.cpu().numpy(), value.cpu().numpy(), h, c
