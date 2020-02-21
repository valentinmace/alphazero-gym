from mcts import MCTS
from node import Node, RootParentNode

class MCTSAgent:
    def __init__(self, env_creator, config):
        self.env = env_creator()
        self.env_creator = env_creator
        self.config = config

    def play_episode(self):
        obs = self.env.reset()
        env_state = self.env.get_state()

        done = False
        t = 0
        total_reward = 0.0

        mcts = MCTS(self.config)

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
            t += 1
            # compute action choice
            action, root_node = mcts.compute_action(root_node)
            # remove old part of the tree that we wont use anymore
            root_node.parent = RootParentNode(env=self.env_creator())

            # take action
            obs, reward, done, info = self.env.step(action)
            if self.config["render"]:
                self.env.render()
            total_reward += reward
        self.env.close()
        return t, total_reward
