from AlphaZero.AlphaZero_backend import AlphaZeroConfig, ReplayBuffer, ResNet, Game
from multiprocessing import cpu_count, Manager, Pool, active_children
from AlphaZero.AlphaZeroMCTS import *
import tensorflow as tf
from AlphaZero.Pit import AgentZeroCompetitive, Pit, SmartRandomAgent
from tqdm import tqdm
import os
import time
from ctypes import c_int, c_bool, c_double
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
              num_sims=50)

    print(f'AI Winrate: {pit.p1_winrate}')
    print(f'Opp Winrate: {pit.p2_winrate}')
    return pit.p1_winrate

def train_network(buffer, weights, games, episode, winrate_list, switch):
    #while episode.value < 2002:
    switch.value = True

    

    for _ in tqdm(range(3000)):
        print(f'New Games: {len(games)}')
        # Save games to buffer only when more than 5 new games are generated from selfplay
        if len(games) > 5:

            for _ in range(len(games)):
                try:
                    game = games.pop(0)
                    buffer.save_game(game)
                except IndexError:
                    pass
            max_len = 5000
            if episode.value > 1000:
                max_len = 10000
            if episode.value > 2000:
                max_len = 20000

            while len(buffer.buffer) > max_len:
                buffer.buffer.pop(0)

            print(f'Epoch : {episode.value}, generating batch on {len(buffer.buffer)} games')
            print(time.ctime(time.time()))

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

            # Set learning rate of model - trying a cyclical learning rate

            learning_rate = 1e-3
            learning_rate = learning_rate * 0.99 ** episode.value

            if episode.value > 250:
                learning_rate = 1e-4
                learning_rate = learning_rate * 0.995 ** (episode.value - 250)

            if episode.value > 500:
                learning_rate = 1e-4
                learning_rate = learning_rate * 0.995 ** (episode.value - 500)
            
            if episode.value > 800:
                learning_rate = 1e-4
                learning_rate = learning_rate * 0.999 ** (episode.value - 800)
                
            if episode.value > 1500:
                learning_rate = 1e-4
                learning_rate = learning_rate * 0.999 ** (episode.value - 800)

            print(f'{learning_rate=}')

            # Compile model
            losses = {'value_head': 'mse', 'policy_head': tf.nn.softmax_cross_entropy_with_logits}

            network.model.compile(loss=losses,
                                    # optimizer=tf.keras.optimizers.Nadam(lr=learning_rate))
                                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate, 
                                                                    momentum=0.9, 
                                                                    nesterov=True))

            # Average loss over 2 batches
            loss = 0
            if episode.value < 200:
                bs = 32
            elif episode.value < 500:
                bs = 64
            elif episode.value < 1000:
                bs = 128
            else:
                bs = 256
            

            # Training Loop
            # Sample 2 batches per episode
            for _ in range(2):

                # Sample from ReplayBuffer
                batch = buffer.sample_batch()

                images, target_v, target_p = batch
                del batch
                

                h = network.model.fit(x=images, y=[target_v, target_p], batch_size=bs)
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

    with Pool(processes=cpu_count()-2, maxtasksperchild=5) as pool:
        jobs = []
        jobs.append(pool.apply_async(train_network, [buffer, weights, games, episode, winrate_list, switch]))
        for i in range(7):
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
    # Current number of total epochs trained : 5160
    buffer_path = '/Users/timwu/models/AlphaZeroResNet/buffer.pkl'
    with open(buffer_path, 'rb') as f:
        buffer = pickle.load(f)

    # buffer = ReplayBuffer()
    #
    # for game in disk_buffer.buffer:
    #     buffer.save_game(game)

    print('Loaded')

    MODEL_NAME = "AlphaZeroResNet"
    subdir = f'{MODEL_NAME}'
    if not os.path.isdir(f'models/{subdir}'):
        os.makedirs(f'models/{subdir}')
    manager = Manager()
    episode = manager.Value(c_int, 1)
    switch = manager.Value(c_bool, True)
    # lr = manager.Value(c_double, 5e-4)
    winrate_list = manager.list()

    games = manager.list()
    weights = manager.list()
    # weights.append(None)
    w = r'/Users/timwu/models/AlphaZeroResNet/240__loss_4.0471885204315186.h5'
    weights.append(w)

    main()






