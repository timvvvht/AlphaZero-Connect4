from collections import deque
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf


class AlphaZeroConfig:

    def __init__(self):
        # Self-Play
        self.num_sampling_moves = 4  # moves until agent stops sampling moves; 0 when competitive; orig trained on 8
        self.num_simulations = 30  # trained on 30 sims / move ; AlphaZero used 800 simulations

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.2
        self.root_exploration_fraction = 0.25  # controls exploration-exploitation

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.window_size = 50000  # num training examples to keep in rolling memory
        self.batch_size = 4096  # takes in 4096 training examples of X, Y : state_image, (state_val, state_policy)


class ReplayBuffer:
    def __init__(self, config=None):
        self.batch_size = 4096
        self.buffer = list()
        if config is not None:
            self.batch_size = config.batch_size
            self.buffer = deque(maxlen=config.window_size)

    def save_game(self, game):
        self.buffer.append(game)

    def sample_batch(self, long_game_multiplier):
        num_examples = len(self.buffer)

        # makes it more likely that longer games will be selected, with game length as a proxy for skill
        game_len = np.array([len(g.history) for g in self.buffer])
        mean_game_len = np.mean(game_len)

        print(f'Mean Game Len: {mean_game_len}')

        move_sum = float(np.sum(game_len))

        prob = game_len / move_sum

        for idx in np.argwhere(game_len > mean_game_len):
            prob[idx] *= long_game_multiplier  # more likely to be selected if game len more than mean

        # make it more likely that newer examples will be selected
        prob[-int(num_examples*0.2):] *= 5
        prob /= sum(prob)

        games = np.random.choice(self.buffer, size=self.batch_size, p=prob)
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        # batch = []
        images = []
        target_vs = []
        target_ps = []
        for g, i in game_pos:
            image = g.make_image(i)
            target_v, target_p = g.make_target(i)
            target_v = np.array(target_v, dtype=np.float32)  # .reshape(-1, 1)
            target_p = np.array(target_p, dtype=np.float32)

            # Augment Data randomly to take advantage of the vertical symmetry of Connect4
            if np.random.random() < 0.5:
                image = np.fliplr(image)
                target_p = np.flip(target_p)

            images.append(image)
            target_vs.append(target_v)
            target_ps.append(target_p)
            # batch.append((image, target_v, target_p))

        # Process duplicate images
        images = np.array(images)
        target_vs = np.array(target_vs)
        target_ps = np.array(target_ps)
        images_str = [str(i) for i in images]
        unique = set(images_str)
        for i in unique:
            idx = np.where(np.array(images_str) == i)[0]

            mean_v = np.mean(target_vs[idx])
            target_vs[idx] = mean_v

            mean_p = np.mean(target_ps[idx], axis=0)
            target_ps[idx] = mean_p

        batch = [images, target_vs, target_ps]
        del unique
        del images_str
        del game_pos
        del games
        del prob

        return batch

class ResNet:
    def __init__(self, weights=None):
        self.rows = 6
        self.columns = 7
        self.model = self.create_model()
        if weights:
            self.model.load_weights(weights)


    @staticmethod
    def res_block(inputs, filters, reg=0.01, bn_eps=2e-5):
        '''Builds blocks with skip connections'''

        x = Conv2D(filters=int(filters), kernel_size=3, padding="same", kernel_regularizer=l2(reg))(inputs)
        x = BatchNormalization(epsilon=bn_eps)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=int(filters), kernel_size=3, padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(epsilon=bn_eps)(x)
        x = add([x, inputs])
        x = Activation('relu')(x)
        return x

    def policy_head(self, x, bn_eps=2e-5):
        '''Policy head that outputs move probabilities-logits over each column'''
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization(epsilon=bn_eps)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.columns, activation='linear', name='policy_head')(x)
        return x

    def value_head(self, x, bn_eps=2e-5):
        '''Value head that outputs a prediction of the state's value in range {-1,1}'''
        x = Conv2D(32, kernel_size=1, padding='same')(x)
        x = BatchNormalization(epsilon=bn_eps)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='tanh', name='value_head')(x)
        return x

    def create_model(self, num_residual_blocks=2, reg=0.01, bn_eps=2e-5):
        inputs = Input(shape=(self.rows, self.columns, 3))
        x = Conv2D(256, kernel_size=3, padding='same', kernel_regularizer=l2(reg))(inputs)
        x = BatchNormalization(epsilon=bn_eps)(x)
        x = Activation('relu')(x)

        '''AlphaZero used 20 residual blocks. Here we'll use 2.'''
        for _ in range(num_residual_blocks):
            x = ResNet.res_block(x, 256)

        p = self.policy_head(x)
        v = self.value_head(x)

        model = Model(inputs, [v, p])
        return model

    def inference(self, x):
        if len(x.shape) != 4:
            x = np.expand_dims(x, axis=0)
        value, policy_logits = self.model.predict(x)
        prob = tf.nn.softmax(policy_logits)
        return np.squeeze(value), np.squeeze(prob)


class Ensemble(ResNet):
    def __init__(self, *weights):
        super().__init__()
        self.weights = weights
        self.models = [self.create_model() for _ in self.weights]
        for m, w in zip(self.models, self.weights):
            m.load_weights(w)

    def inference(self, x):
        if len(x.shape) != 4:
            x = np.expand_dims(x, axis=0)

        value = 0.
        policy = np.zeros(self.columns)
        for model in self.models:
            v, p = model.predict(x)
            value += v
            policy += np.squeeze(p)
        value /= len(self.models)
        prob = tf.nn.softmax(policy)
        return np.squeeze(value), prob

class Game:
    '''
    Handles Connect4 Game object,
    creating training data in the form of

    state representation : [value, move probabilities], and

    handling MCTS search tree statistics'''

    def __init__(self, history=None):
        self.history = history or []
        self.child_visits = []
        self.rows = 6
        self.columns = 7
        self.num_actions = self.columns
        self.initial_state = np.zeros((self.rows, self.columns))

    def __repr__(self):
        return str(self.state())

    def state(self, state_index=None):
        board = self.initial_state.copy()
        history = self.history
        if state_index is not None:
            history = history[:state_index]
        for move in history:
            token = -1 if len(np.argwhere(board != 0)) % 2 == 1 else 1
            move_idx = self.lowest_position(board, move)
            board[move_idx] = token
        return board

    @staticmethod
    def lowest_position(board, col):
        try:
            row = np.argwhere(board[:, col] == 0)[-1]
            return int(row), col
        except IndexError:
            return False

    @property
    def terminal(self):
        to_play = self.to_play
        if self.terminal_value(to_play) == 0:
            return False
        return True

    def terminal_value(self, to_play, board_state=None):
        reward = 1
        state = self.state()

        if board_state is not None:
            state = board_state

        if len(self.legal_actions) == 0:
            return 1e-10

        consec = self.check_win(state)
        if consec == 0:
            return 0

        winner = 0 if consec == 1 else 1

        return reward if winner != int(to_play) else -reward


    @property
    def legal_actions(self):
        # returns np array of legal columns
        try:
            return np.array(list(filter(None, [self.lowest_position(self.state(), c) for c in range(self.columns)])))[:, -1]
        except IndexError:
            return []

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index=None):
        '''Creates the input data for the resnet to train on
            Output data will be in the shape of (rows, columns, 3)
            Top layer is a canonical game board so to the agent, it is always player 1
            Middle layer is a binary matrix with 1s where positions are occupied by own tokens, else 0
            Bottom layer is a binary matrix with 1s where positions are occupied by opponents, else 0 '''

        state = self.state(state_index)
        to_play = len(self.history[:state_index]) % 2
        to_play_matrix = np.ones((self.rows, self.columns))
        if to_play != 0:
            state = state*-1
            state[state == -0] = 0
        curr_player_binary = np.array(state == to_play_matrix, dtype='float32')
        opp_player_binary = np.array(state == -to_play_matrix, dtype='float32')
        input_image = np.dstack([state, curr_player_binary, opp_player_binary])
        return input_image

    def make_target(self, state_index: int):
        discount_rate = 0.999
        # discount based on distance from terminal state
        move_dist = (len(self.history) - state_index)/2
        value = self.terminal_value(state_index % 2) * discount_rate ** move_dist
        return value, self.child_visits[state_index]

    @property
    def to_play(self):
        # returns 0 if player 1 else returns 1
        return len(self.history) % 2

    @staticmethod
    def check_win(state):
        window_len = 4
        rows, columns = [], []

        for row in state:
            for idx in range(len(row)):
                rows.append(row[idx:idx+window_len])

        for col in state.T:
            for idx in range(len(col)):
                columns.append(col[idx:idx+window_len])

        diags_lr = [state[::-1, :].diagonal(i) for i in range(-state.shape[0] + 1, state.shape[1])]
        board_flip = np.fliplr(state)
        diags_rl = [board_flip[::-1, :].diagonal(i) for i in range(-state.shape[0] + 1, state.shape[1])]

        diags = diags_lr + diags_rl
        diags = [i for i in diags if len(i) >= window_len]

        _all = rows + columns + diags

        for i in _all:
            winner = Game.check_consec(i)
            if winner != 0:
                return winner
        return 0

    @staticmethod
    def check_consec(array):
        consec = []
        for i in array:
            if len(consec) == 0 or i == consec[-1]:
                consec.append(i)
            else:
                consec = [i]
            if len(consec) == 4:
                return consec[-1]
        return 0

