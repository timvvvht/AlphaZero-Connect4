from AlphaZero.AlphaZero_backend import *
from AlphaZero.AlphaZeroMCTS import *
from AlphaZero.Pit import *
from tqdm import tqdm
import os
import tensorflow as tf

MODEL_NAME = "AlphaZeroResNet"
subdir = f'{MODEL_NAME}'
if not os.path.isdir(f'models/{subdir}'):
    os.makedirs(f'models/{subdir}')

class AlphaZero:
    def __init__(self, config, replay_buffer: ReplayBuffer, net):
        self.config = config
        self.replay_buffer = replay_buffer
        self.net = net
        self.learning_rate = self.config.learning_rate
        self.total_epochs = 0

    def run_selfplay(self, verbose=False):
        game = Game()
        while not game.terminal:
            action, root = run_mcts(self.config, game, self.net)
            game.apply(action)
            game.store_search_statistics(root)
        if verbose:
            print(game)
        self.replay_buffer.save_game(game)
        return game


    def train_network(self):

        for i in range(self.config.batches_per_iter):
            # get batch of data from replay_buffer
            batch = self.replay_buffer.sample_batch()
            self.update_weights(self.net.model, batch)

    def update_weights(self, network, batch):
        # compile network with most recent learning rate
        network.compile(loss=[tf.nn.softmax_cross_entropy_with_logits, 'mean_squared_error'],
                        optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        images, target_v, target_p = batch

        network.fit(x=images, y=[target_v, target_p], epochs=self.config.epochs_per_batch)

        return

    def train_ep(self, num_self_play_games=None):
        if num_self_play_games is None:
            num_self_play_games = self.config.self_play_games
        for i in range(num_self_play_games):
            print(f'SelfPlay Game: {i + 1}')
            if i % 10 == 0:
                self.run_selfplay(verbose=True)
            else:
                self.run_selfplay()  # self play games added to replay buffer

        self.train_network()


game = Game()
config = AlphaZeroConfig()
net = ResNet()
agentZero = AlphaZero(config, replay_buffer=ReplayBuffer(config), net=net)

episodes = 500
winrates_v_smart = []
winrates_v_dumb = []
SmartRandAgent = SmartRandomAgent()
RandAgent = SmartRandomAgent(dumb=True)

'''
for e in tqdm(range(episodes)):
    # 1 episode = self_play_games x 25
    # 5 x Gets 2048 sets of data from replay buffer as a batch
    # train 2 epochs on each batch
    agentZero.train_ep()
    if e % 10 == 0:
        agentZero_C = AgentZeroCompetitive(AlphaZeroConfig(), net=agentZero.net)

        winrate = 0

        # Evaluation against Smart Random Agent
        pit = Pit(agentZero_C, SmartRandAgent, num_sims=20)
        pit.simulate()
        winrate += pit.winrate * 0.5
        winrates_v_smart.append(pit.winrate)

        # Evaluation against Random Agent
        pit = Pit(agentZero_C, RandAgent, num_sims=20)
        pit.simulate()
        winrate += pit.winrate * 0.5
        winrates_v_dumb.append(pit.winrate)

        print('\n')
        print(f' {e}/ {episodes}')
        print(f'Winrate v Smart: {winrates_v_smart}')
        print(f'Winrate v Dumb: {winrates_v_dumb}')
        print('\n')

        # Save latest model to disk
        agentZero.net.model.save(
            f'models/{subdir}/episode_{e}__avg_winrate_{str(winrate)}.model')

    if e == 100:
        agentZero.learning_rate = 1e-4

    if e == 300:
        agentZero.learning_rate = 1e-5'''
