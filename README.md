# AlphaZero-Gym
>An implementation of AlphaZero for Gym environments

> Made for my interview at InstaDeep


This repository contains:
- A folder implementing AlphaZero with a dense neural network that can be tested on CartPole environment
- An experimental folder implementing AlphaZero with an LSTM neural net

## Installation

Libraries you'll need to run the project:

{``torch, gym, numpy, scipy``}

Clone the repo using

```sh
git clone https://github.com/valentinmace/alphazero-gym.git
```

## Usage

Testing the MCTS algorithm only
```sh
cd AlphaZero
python3 run_mcts.py 
```

Testing the AlphaZero algorithm with dense neural net
```sh
cd AlphaZero
python3 run_alphazero.py
```

Testing the AlphaZero algorithm with LSTM neural net (Experimental)
```sh
cd AlphaZero-LSTM
python3 run_alphazero.py
```

## Meta

Valentin Macé – [LinkedIn](https://www.linkedin.com/in/valentin-mac%C3%A9-310683165/) – [YouTube](https://www.youtube.com/channel/UCMIW0JKxoxBDM5yiiF17SrA) – [Twitter](https://twitter.com/ValentinMace) - valentin.mace@kedgebs.com

Distributed under the MIT license. See ``LICENSE`` for more information.
