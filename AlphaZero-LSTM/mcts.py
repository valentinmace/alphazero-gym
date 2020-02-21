import numpy as np

from utils import softmax_normalized, multiple_rollouts


class MCTS:
    def __init__(self, mcts_param):
        self.params = mcts_param
        self.max_q_value = 0
        self.min_q_value = 0

    def compute_priors_and_value(self, node):
        env = node.env
        env.set_state(node.state)

        # todo: write the value estimation, we may estimate it with a single random rollout now
        # Doing a certain amount of rollouts to estimate a node value
        value = multiple_rollouts(env, node.state, self.params["num_rollouts"])

        # todo: compute the priors, the resulting tensors must have a (1, num_actions) shape
        # Priors are computed as 1/action_space_size (biggest difference with AlphaZero)
        priors = np.full((1,node.action_space_size), 1/node.action_space_size)
        return priors, value

    def compute_action(self, node):
        # Run simluations
        for _ in range(self.params['num_simulations']):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Compute Tree policy target (TPT): todo: complete the tree policy computation
        # Tree policy is a probability distribution over potential actions
        tree_policy = softmax_normalized(node.child_number_visits, self.params['temperature'])

        # Choose action according to tree policy
        action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return action, node.children[action]

    def update_q_value_stats(self, q_value):
        self.max_q_value = max(self.max_q_value, q_value)
        self.min_q_value = min(self.min_q_value, q_value)

    def normalize_q_value(self, q_value):
        if self.max_q_value > self.min_q_value:
            return (q_value - self.min_q_value) / (self.max_q_value - self.min_q_value)
        else:
            return q_value
