import collections
import numpy as np


class RootParentNode(object):
    def __init__(self, env):
        self.parent = None
        self.child_q_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.depth = 0
        self.env = env


class Node:
    def __init__(self, action, reward, obs, state, mcts, depth, done, parent=None):

        self.env = parent.env
        self.action = action  # Action used to go to this state
        self.done = done

        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.depth = depth

        self.action_space_size = self.env.action_space.n

        self.child_q_value = np.zeros([self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros([self.action_space_size], dtype=np.float32)  # N

        self.reward = reward
        self.obs = obs
        self.state = state

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def q_value(self):
        return self.parent.child_q_value[self.action]

    @q_value.setter
    def q_value(self, value):
        self.parent.child_q_value[self.action] = value
        self.mcts.update_q_value_stats(value)

    # todo: complete this method
    def best_action(self):
        # compute a score based on the children priors and values
        child_score = self.score_child_PUCT()
        return np.argmax(child_score)

    def score_child_UCT(self):
        """Estimate a score to chose a node among available children according to UCB1 method

        :return: (np.ndarray) The estimated score over children
        """
        exploration_incentive = np.log(np.full((self.action_space_size), self.number_visits)) / self.child_number_visits
        exploration_incentive = 2 * self.mcts.params['cp_coefficient'] * np.sqrt(exploration_incentive)
        return self.child_q_value + exploration_incentive

    def score_child_PUCT(self):
        """Estimate a score to chose a node among available children according to PUCT method
        It is similar to UCT but differs from the fact that priors are involved in the score (usefull for Alphazero)
        and an adaptative coefficient is applied to the score

        :return: (np.ndarray) The estimated score over children
        """
        coeff = np.log((1 + self.mcts.params['c2_coefficient'] + self.number_visits) / self.mcts.params['c2_coefficient']) + self.mcts.params['c1_coefficient']
        exploration_incentive = np.sqrt(np.full((self.action_space_size), self.number_visits)) / (1 + self.child_number_visits)
        exploration_incentive = coeff * self.child_priors * exploration_incentive
        return self.mcts.normalize_q_value(self.child_q_value) + exploration_incentive

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        # create a node for a selected action and attach it to the current node.
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, info = self.env.step(action)
            next_state = self.env.get_state()

            self.children[action] = Node(
                obs=obs,
                done=done,
                state=next_state,
                action=action,
                depth=self.depth + 1,
                parent=self,
                reward=reward,
                mcts=self.mcts,
            )
        return self.children[action]

    # todo: write a backup function
    def backup(self, value):
        # update current node
        self.q_value = value
        self.number_visits += 1
        # update all other nodes up to root node
        current = self.parent
        while current.parent is not None:
            current.q_value = self.q_value_estimation(current, value)
            current.number_visits += 1
            current = current.parent

    def q_value_estimation(self, node, value):
        """Estimate a Q value for a given node
        The Q value is computed according to the MuZero paper's method
        """
        g = self.cumulative_discounted_reward(node, value)
        return (node.number_visits * node.q_value + g) / (node.number_visits + 1)

    def cumulative_discounted_reward(self, node, value):
        """Estimate a cumulative discounted reward for the current node
        It takes into account every reward obtained by its children on the current path and the value obtained by
        the leaf node
        """
        step = self.depth - node.depth
        discounted_value = (self.mcts.params['gamma'] ** step) * value
        rewards = 0
        # Starting from the leaf, going up the path to the desired parent
        current = self
        for i in range(step-1,-1,-1):
            rewards += (self.mcts.params['gamma'] ** i) * current.reward
            current = current.parent
        return rewards + discounted_value
