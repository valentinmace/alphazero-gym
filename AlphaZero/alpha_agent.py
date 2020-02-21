import numpy as np
import torch

from alpha_mcts import AlphaZeroMCTS
from model import AlphaZeroModel
from node import Node, RootParentNode
from replay_buffer import ReplayBuffer
from utils import scaling_func, compute_td_target, compute_support_torch, compute_cross_entropy


class AlphaZero:
    def __init__(self, env_creator, config):
        self.env_creator = env_creator
        self.env = env_creator()
        self.config = config
        self.mcts_config = config['mcts_config']
        self.mcts_config.update(config)
        self.model = AlphaZeroModel(self.env.observation_space, self.env.action_space, config['model_config'])
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.total_num_steps = 0

    def play_episode(self):
        transitions = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "tree_policies": [],
            "tree_values": [],
        }
        # TODO: complete this method

        # play one episode with mcts and store the observations, actions, rewards,
        # tree policies and tree values at each timestep in the dictionnary transitions

        obs = self.env.reset()
        env_state = self.env.get_state()
        done = False

        mcts = AlphaZeroMCTS(self.mcts_config, self.model)

        root_node = Node(
            state=env_state,
            done=False,
            obs=obs,
            reward=0,
            action=None,
            parent=RootParentNode(env=self.env_creator()),
            mcts=mcts,
            depth=0
        )

        while not done:
            # compute action choice
            tree_policy, action, tree_value, root_node = mcts.compute_action(root_node)
            # remove old part of the tree that we wont use anymore
            root_node.parent = RootParentNode(env=self.env_creator())

            # store observation, action, tree_policy, tree_value
            transitions['observations'].append(obs)
            transitions['actions'].append(action)
            transitions['tree_policies'].append(tree_policy)
            transitions['tree_values'].append(tree_value)
            # obtain reward for the chosen action and store it
            obs, reward, done, info = self.env.step(action)
            transitions['rewards'].append(reward)

        return transitions

    def postprocess_transitions(self, transitions):

        # TODO: complete this method

        # transitions dict flows directly into this function when an episode has been played
        # compute the value targets from the rewards and tree values
        # the parameter gamma is in self.config['gamma'] and the parameter n is in
        # self.config['n_steps']
        value_targets = compute_td_target(self.config['gamma'], np.asarray(transitions['rewards']), np.asarray(transitions['tree_values']), self.config['n_steps'])

        # we scale the value targets using function h
        value_targets = scaling_func(value_targets, mode='numpy')

        # we transform the np array into a list of numpy arrays, one per transition
        transitions['value_targets'] = np.split(value_targets, len(value_targets))

        # we dont store useless arrays in the buffer
        del transitions['rewards']
        del transitions['tree_values']

        return transitions

    def compute_loss(self, batch):

        # TODO: complete this method

        # compute AlphaZero loss in this function
        # batch is a dict of transitions with keys: 'observations', 'tree_policies', 'value_targets'
        # each key is associated to a numpy which first dim equals batch size

        # first we get supports parameters
        v_support_minv, v_support_maxv = self.model.value_min_val, self.model.value_max_val

        # transform numpy vectors to torch tensors
        observations = torch.from_numpy(batch['observations']).float()
        mcts_policies = torch.from_numpy(batch["tree_policies"]).float()
        value_targets = torch.from_numpy(batch["value_targets"]).float()[:, 0]
        # compute support for the loss function
        value_targets = compute_support_torch(value_targets, v_support_minv, v_support_maxv)

        # compute losses
        model_output = self.model(observations)
        # policy loss computed with actual mcts policies and the model's logits
        policy_loss = compute_cross_entropy(mcts_policies, model_output[0])
        # value loss computed with actual value targets and model's values estimated
        value_loss = compute_cross_entropy(value_targets, model_output[1])

        # compute total loss
        # we rescale the value loss with a coefficient given as an hyperparameter
        value_loss = self.config['value_loss_coefficient'] * value_loss
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss

    def train(self):
        # we train the agent for several epochs. In this notebook we define an epoch as the succession
        # of data generation (we play episodes with the MCTS) and training (we sample
        # batches of data in the replay buffer and train on them)
        for _ in range(self.config['num_epochs']):
            episode_rewards = []
            num_steps = 0
            for _ in range(self.config['num_episodes_per_epoch']):
                # play an episode
                transitions = self.play_episode()
                episode_rewards.append(np.sum(transitions['rewards']))
                num_steps += len(transitions['rewards'])
                # process the transitions
                transitions = self.postprocess_transitions(transitions)
                # store them in the replay buffer
                self.replay_buffer.add(transitions)

            avg_rewards = np.mean(episode_rewards)
            max_rewards = np.max(episode_rewards)
            min_rewards = np.min(episode_rewards)
            self.total_num_steps += num_steps

            s = 'Num timesteps sampled so far {}'.format(self.total_num_steps)
            s += ', mean accumulated reward: {}'.format(avg_rewards)
            s += ', min accumulated reward: {}'.format(min_rewards)
            s += ', max accumulated reward: {}'.format(max_rewards)
            print(s)

            # we want for the buffer to contain a minimum numer of transitions
            # if enough timesteps collected, then start training
            if self.total_num_steps >= self.config['learning_starts']:

                # perform one SGD per transition sampled
                for _ in range(num_steps):
                    # sample transitions in the replay buffer
                    batch = self.replay_buffer.sample(self.config['batch_size'])
                    # compute loss
                    total_loss, policy_loss, value_loss = self.compute_loss(batch)
                    # do backprop
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()