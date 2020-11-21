from collections import deque
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf


class AlphaZeroConfig:

    def __init__(self):
        # Self-Play
        self.num_sampling_moves = 8  # moves until agent stops softmax-sampling moves
        self.num_simulations = 150  # trained on 30 sims / move ; AlphaZero used 800 simulations

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.2  # for training; 0 for competitive mode
        self.root_exploration_fraction = 0.25  # controls exploration-exploitation

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.window_size = 500  # num training examples to keep in rolling memory
        self.batch_size = 4096  # takes in 4096 training examples of X, Y : state_image, (state_val, state_policy)


class ReplayBuffer:
    def __init__(self, config=None):
        self.batch_size = 4096
        self.buffer = deque(maxlen=500)
        if config is not None:
            self.batch_size = config.batch_size
            self.buffer = deque(maxlen=config.window_size)

    def save_game(self, game):
        self.buffer.append(game)

    def sample_batch(self):
        num_examples = len(self.buffer)

        # make it more likely that newer examples will be selected
        prob = np.ones(num_examples)
        prob[-int(num_examples*0.2):] *= 4
        prob /= sum(prob)

        games = np.random.choice(self.buffer, size=self.batch_size,
                                 p=prob)
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

            # Augment Data to take advantage of the vertical symmetry of Connect4
            if np.random.random() < 0.5:
                image = np.fliplr(image)
                target_p = np.flip(target_p)

            images.append(image)
            target_vs.append(target_v)
            target_ps.append(target_p)
            # batch.append((image, target_v, target_p))
        batch = [np.array(images), np.array(target_vs), np.array(target_ps)]
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
            #to_play_matrix = -to_play_matrix
            state = state*-1
            state[state == -0] = 0
        curr_player_binary = np.array(state == to_play_matrix, dtype='float32')
        opp_player_binary = np.array(state == -to_play_matrix, dtype='float32')
        input_image = np.dstack([state, curr_player_binary, opp_player_binary])
        return input_image

    def make_target(self, state_index: int):
        discount_rate = 0.9
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
        diags = [np.delete(i, np.where(i == 0)) for i in diags]

        _all = rows + columns + diags
        try:
            winner = [i for i in _all if len(i) == 4 and (i == i[0]).all() and i[0] != 0][0][0]
        except:
            winner = 0
        return winner






    # @classmethod
    # def check_win_horiz(cls, board):
    #     winner = None
    #     win_con = 4
    #     for row in board:
    #         consecutive = []
    #
    #         for col in row:
    #             if len(consecutive) == 0:
    #                 consecutive = [col]
    #             elif consecutive[-1] == col:
    #                 consecutive.append(col)
    #             else:
    #                 consecutive = [col]
    #             if len(consecutive) == win_con and consecutive[0] != 0:
    #                 winner = consecutive[0]
    #     return winner
    #
    # @classmethod
    # def check_win_vert(cls, board):
    #     winner = None
    #     win_con = 4
    #     for row in board.T:
    #         consecutive = []
    #         for col in row:
    #             if len(consecutive) == 0:
    #                 consecutive = [col]
    #             elif consecutive[-1] == col:
    #                 consecutive.append(col)
    #             else:
    #                 consecutive = [col]
    #             if len(consecutive) == win_con and consecutive[0] != 0:
    #                 winner = consecutive[0]
    #     return winner
    #
    # @classmethod
    # def check_win_diag(cls, board):
    #     winner = None
    #     win_con = 4
    #     diags_lr = [board[::-1, :].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
    #     board_flip = np.fliplr(board)
    #     diags_rl = [board_flip[::-1, :].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
    #     diags = diags_lr + diags_rl
    #     for diag in diags:
    #         if len(diag) >= 4:
    #             consecutive = []
    #             for d in diag:
    #                 if len(consecutive) == 0:
    #                     consecutive = [d]
    #                 elif consecutive[-1] == d:
    #                     consecutive.append(d)
    #                 else:
    #                     consecutive = [d]
    #                 if len(consecutive) == win_con and consecutive[0] != 0:
    #                     winner = consecutive[0]
    #     return winner
