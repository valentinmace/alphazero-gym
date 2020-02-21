import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from agent import MCTSAgent
from cartpole import CartPole

env_creator = lambda: CartPole()

cp = np.sqrt(2)/2
mcts_config = {
    "num_simulations": 15,   # must be > 1
    "num_rollouts": 3,
    "gamma": 0.997,
    "temperature": 1.0,
    "c1_coefficient": 1.25,
    "c2_coefficient": 19652,
    "render": True,
    "cp_coefficient": cp
}

agent = MCTSAgent(env_creator, mcts_config)

#todo: assess the agent performance and study the hyperparameters importances

number_of_episodes = 3
scores = np.zeros(number_of_episodes)
for n in range(number_of_episodes):
    t, total_reward = agent.play_episode()
    scores[n] = total_reward

# compute and print episodes informations: mean and std over scores and times
mean_score = np.mean(scores)
std_score = np.std(scores)

print("Mean score over ", number_of_episodes, " episodes :", mean_score)
print("SD of scores : ", std_score)

