from numpy.polynomial.polynomial import polyval
from scipy.linalg import hankel
import numpy as np
import torch


def compute_support_torch(value, min_val, max_val, mesh=1.0):
    # rescale according to mesh
    value = value / mesh
    min_val = int(min_val / mesh)
    max_val = int(max_val / mesh)

    # value is assumed to be a batch of values of shape (bs,)
    minv = min_val * torch.ones_like(value)
    value = torch.clamp(value, min_val, max_val)

    support_size = max_val - min_val + 1
    support = torch.zeros((value.size()[0], support_size))

    ind = torch.unsqueeze(torch.floor(value) - minv, dim=1)
    src = torch.unsqueeze(torch.ceil(value) - value, dim=1)
    support.scatter_(dim=1, index=ind.long(), src=src)

    ind = torch.unsqueeze(torch.ceil(value) - minv, dim=1)
    src = torch.unsqueeze(value - torch.floor(value), dim=1)
    src.apply_(lambda x: 1 if x == 0 else x)
    support.scatter_(dim=1, index=ind.long(), src=src)

    return support


def compute_cross_entropy(labels, predictions):
    # assume both of shape (bs, num_probs) and to be distributions along last axis
    # assume predictions are raw logits
    predictions = torch.nn.LogSoftmax(dim=-1)(predictions)
    return torch.mean(-torch.sum(labels * predictions, dim=-1))


def compute_value_from_support_torch(support, min_val, max_val, mesh=1.0):
    min_val = int(min_val / mesh)
    max_val = int(max_val / mesh)
    return torch.sum(support * torch.arange(min_val, max_val + 1), dim=1) * mesh


def scaling_func(x, mode='numpy', epsilon=0.001):
    assert mode in ['numpy', 'torch'], 'mode {} not implemented'.format(mode)
    if mode == 'numpy':
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + epsilon * x)
    else:
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)


def scaling_func_inv(x, mode='numpy', epsilon=0.001):
    assert mode in ['numpy', 'torch'], 'mode {} not implemented'.format(mode)
    if mode == 'numpy':
        f_ = (np.power((np.sqrt(1 + 4 * epsilon * (np.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon), 2) - 1)
        return np.sign(x) * f_
    else:
        f_ = (torch.pow((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon), 2) - 1)
        return torch.sign(x) * f_


def compute_discounted_n_steps(gamma, rewards, n_steps):
    return polyval(gamma, hankel(rewards)[:n_steps, :])


def compute_td_target(gamma, rewards, values, n_steps):
    assert rewards.shape == values.shape, 'rewards and values of different shape'
    if n_steps < values.shape[0]:
        values = np.concatenate([values[n_steps:], np.zeros((n_steps,))])
        values *= gamma**n_steps
        return compute_discounted_n_steps(gamma, rewards, n_steps) + values
    else:
        return compute_discounted_n_steps(gamma, rewards, n_steps)


def get_temperature_from_schedule(schedule, timesteps):
    t_limits, temps = zip(*schedule)
    t_limits = np.array(t_limits)
    # assume schedule sorted by increasing timesteps
    largers = np.extract(t_limits > timesteps, t_limits)
    if len(largers) > 0:
        idx = np.argmin(np.abs(timesteps - largers)) + t_limits.shape[0] - largers.shape[0]
    else:
        idx = len(t_limits) - 1
    return temps[idx]


def softmax(x, temperature=1.0):
    """Softmax function with a temperature feature that allows one to reshape the output probability distribution.
    High temperature implies a distribution tending towards uniform distribution
    Low temperature implies a distribution tending toward a one hot vector, or Dirac distribution
    A temperature of 1.0 yields a classical softmax function

    :param x: (ndarray of floats) input vector
    :param temperature: (float) allows to smooth or sharpen
    :return: (ndarray of floats) A probability distribution
    """
    if temperature < 0.1:
        raise ValueError('Temperature parameter should not be less than 0.1')
    t = 1/temperature
    return np.exp(x)**t/sum(np.exp(x)**t)


def softmax_normalized(x, temperature=1.0):
    """A wrapper of the softmax function, allowing to normalize input and avoid overflow due to exponentiation

    :param x: (ndarray of floats) input vector
    :param temperature: (float) allows to smooth or sharpen
    :return: (ndarray of floats) A probability distribution
    """
    x = x - np.max(x)
    return softmax(x, temperature=temperature)


def rollout(env):
    """Performs a rollout (random actions until done) for the given environment

    :return: (float) Rollout total reward
    """
    done = False
    ep_reward = 0.0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        ep_reward += reward
    return ep_reward


def multiple_rollouts(env, initial_state, n_rollouts):
    """Performs multiple rollouts

    :return: (float) Mean of rollouts rewards
    """
    scores = np.zeros(n_rollouts)
    for i in range(n_rollouts):
        env.set_state(initial_state)
        scores[i] = rollout(env)
    return np.mean(scores)
