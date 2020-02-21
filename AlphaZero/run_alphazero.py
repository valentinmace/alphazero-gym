from alpha_agent import AlphaZero
from cartpole import CartPole

# create an env_creator function
env_creator = lambda: CartPole()

# define the config with the hyper-parameters
config = {
    'buffer_size': 1000,
    'batch_size': 256,
    'lr': 1e-3,

    'gamma': 0.997,
    'n_steps': 10,

    'num_epochs': 100,
    'num_episodes_per_epoch': 5,
    'learning_starts': 500,  # number of timesteps to sample before SGD

    'value_loss_coefficient': 0.2,

    'model_config': {
        'value_support_min_val': 0,
        'value_support_max_val': 30,
        'num_hidden': 32,
    },

    'mcts_config': {
        'num_simulations': 20,
        "temperature": 1.0,
        "c1_coefficient": 1.25,
        "c2_coefficient": 19652,
        'add_dirichlet_noise': True,
        'dir_noise': 0.5,
        'dir_epsilon': 0.2,
    }
}

# instanciate the agent
agent = AlphaZero(env_creator, config)

# train it
agent.train()
