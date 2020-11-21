from AlphaZero.AlphaZero_backend import AlphaZeroConfig, ReplayBuffer, ResNet, Game
from multiprocessing import cpu_count, Manager, Pool, active_children
from AlphaZero.AlphaZeroMCTS import *
import tensorflow as tf
from AlphaZero.Pit import AgentZeroCompetitive, Pit, SmartRandomAgent
from tqdm import tqdm
import os
import time
from ctypes import c_int, c_bool
import pickle

def selfplay(games, weights):
    network = ResNet()
    #if weights[0] is not None:
    if type(weights[0]) == str:
        network.model.load_weights(weights[0])
    try:
        network.model.set_weights(weights[0])
    except:
        pass

    game = Game()
    config = AlphaZeroConfig()
    while not game.terminal:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    games.append(game)

    return



def selfplay_worker(games, weights, switch):
    while switch.value is True:
        selfplay(games, weights)
    return

def pit(network):
    pit = Pit(AgentZeroCompetitive(config=AlphaZeroConfig(), net=network, mcts=True),
              SmartRandomAgent(),
              num_sims=40)

    print(f'AI Winrate: {pit.p1_winrate}')
    print(f'Opp Winrate: {pit.p2_winrate}')
    return pit.p1_winrate

def train_network(buffer, weights, games, episode, winrate_list, switch):
    #while episode.value < 2002:
    switch.value = True
    for _ in tqdm(range(3430)):
        print(len(games))
        # Save games to buffer only when more than 10 new games are generated from selfplay
        if len(games) > 10:

            for idx in range(len(games)):
                try:
                    game = games.pop(0)
                    buffer.save_game(game)
                except IndexError:
                    pass
            num_games = len(buffer.buffer)


            print(f'Epoch : {episode.value}, generating batch on {num_games} games')
            print(time.ctime(time.time()))
            del num_games

            # Instantiates network and load weights
            network = ResNet()

            # Loading weights from weights file
            if type(weights[0]) == str:
                # load weights from path to weights
                network.model.load_weights(weights[0])
            try:
                # set weights that are stored in shared memory
                network.model.set_weights(weights[0])
            except:
                pass

            # Set learning rate of model
            learning_rate = 2e-3
            if episode.value > 1200:
                learning_rate = 1e-5
            if episode.value > 500:
                learning_rate = 1e-4
            if episode.value > 100:
                learning_rate = 1e-3

            # Compile model
            losses = {'value_head': 'mse', 'policy_head': tf.nn.softmax_cross_entropy_with_logits}

            network.model.compile(loss=losses,
                                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9))

            # Average loss over 2 batches
            loss = 0

            # Training Loop
            # Sample 2 batches per episode
            for _ in range(2):

                # Sample from ReplayBuffer
                batch = buffer.sample_batch()

                images, target_v, target_p = batch
                del batch

                h = network.model.fit(x=images, y=[target_v, target_p])
                for i in h.history['loss']:
                    loss += i

            print(f'{loss=}')
            w = network.model.get_weights()
            weights[0] = w

            if episode.value % 20 == 0:
                print('Saving buffer to disk')
                with open('models/AlphaZeroResNet/buffer.pkl', 'wb') as f:
                    pickle.dump(buffer, f)

                print('Saving weights to disk')
                network.model.save_weights(
                    f'models/AlphaZeroResNet/{episode.value}__loss_{loss}.h5'
                )

                print('Entering the Pit')
                wins = pit(network)
                try:
                    # save weights to disk if winrate against pit opponent is better than all previous pit evaluations
                    if float(wins) > float(max(winrate_list)):
                        network.model.save_weights(
                            f'models/AlphaZeroResNet/episode_{episode.value}__winrate_{wins}_loss_{loss}.h5'
                        )

                except ValueError:
                    pass

                winrate_list.append(wins)
                print(winrate_list)
                del wins

            episode.value += 1

            del network
            del h
            del images
            del target_p
            del target_v
            del loss
            del losses

        else:
            selfplay(games, weights)

    switch.value = False

def main():

    with Pool(processes=cpu_count()-1, maxtasksperchild=5) as pool:
        jobs = []
        jobs.append(pool.apply_async(train_network, [buffer, weights, games, episode, winrate_list, switch]))
        for i in range(6):
            jobs.append(pool.apply_async(selfplay_worker, [games, weights, switch]))

        for job in jobs:
            job.get()
        jobs = None
        del jobs
        pool.close()
        pool.join()

    pool = None
    active_children()
    del pool

    return

if __name__ == '__main__':
    # Current number of total epochs trained : 4990
    buffer = ReplayBuffer()

    MODEL_NAME = "AlphaZeroResNet"
    subdir = f'{MODEL_NAME}'
    if not os.path.isdir(f'models/{subdir}'):
        os.makedirs(f'models/{subdir}')
    manager = Manager()
    episode = manager.Value(c_int, 1)
    switch = manager.Value(c_bool, True)

    winrate_list = manager.list()

    games = manager.list()
    weights = manager.list()
    weights.append(None)
    #w = r'/Users/timwu/models/AlphaZeroResNet/440__loss_4.144873142242432.h5'
    #weights.append(w)

    main()






