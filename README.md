# AlphaZero in Connect 4 

An asynchronous implementation of the AlphaZero algorithm based on the [AlphaZero paper](https://arxiv.org/pdf/1712.01815.pdf). 

AlphaZero is an algorithm that trains a reinforcement learning agent through self-play. The training examples are states of games, while the 'ground truth' labels are value of a state and policy (probability distribution of actions) of a state. 

AlphaZero uses a [modified version](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/AlphaZero/AlphaZeroMCTS.py)
of the Monte Carlo Tree Search (MCTS) which uses the trained network to predict values of states rather than performing rollouts upon traversing to a leaf node. 

## Training
Training was done with a multiprocessing, asynchronous approach demonstrated [here](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/AlphaZeroTrainMultiprocessing.py).

The agent was trained for 1 week, and was able to defeat the one-step-look-ahead agent consistently very quickly (at around 3000 epochs). 

I then tested the agent against myself. While it was difficult to beat, it is not unbeatable, and as Connect4 is a solved game, this agent should theoretically be able to converge to an optimal policy. I then increased the memory buffer size and started training on it again. Future updates will be reported.


## Codebase
The AlphaZero [folder](https://github.com/timvvvht/AlphaZero-Connect4/tree/main/AlphaZero) contains all of the backend code for this implementation. 

The training configuration, ResNet built using tensorflow 2, memory object and game object can be found [here](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/AlphaZero/AlphaZero_backend.py).

MCTS related functions can be found [here](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/AlphaZero/AlphaZeroMCTS.py).

The Pit object for evaluating the agent against a one-step-look-ahead agent can be found [here](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/AlphaZero/Pit.py).



![Gif](https://github.com/timvvvht/AlphaZero-Connect4/blob/main/media/c4ai.gif)
