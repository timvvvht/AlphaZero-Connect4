from Agents import BaseAgent
from collections import deque
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.optimizers import Adam
import tensorflow as tf
from Connect4_env import *


MODEL_NAME = 'CONV128_64_64_LR15e-4'
DISCOUNT = 0.95

class DQNAgent2(BaseAgent):
    def __init__(self, player, name=None, epsilon=1, epsilon_decay=0.99975,
                 memory_size=50000, min_memory_size=1000, minibatch_size=128,
                 trained_model=None):
        super().__init__(player, name)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.min_memory_size = min_memory_size
        self.minibatch_size = minibatch_size
        self.trained_model = trained_model


        # Model
        self.model = self.create_model()

        self.current_game_states = []
        self.current_game_states_opp = []

        self.memory = deque(maxlen=self.memory_size)

        self.player_matrix = np.ones((Connect4().board.shape[0], Connect4().board.shape[1])) \
            if self.player == 1 else -np.ones((Connect4().board.shape[0], Connect4().board.shape[1]))

    def create_model(self):
        if self.trained_model is not None:
            model = tf.keras.models.load_model(self.trained_model)
        else:
            # inputs will be current state, mirrored current state and matrix of all 1s or -1s depending on player
            inputs = Input(shape=(Connect4().board.shape[0], Connect4().board.shape[1], 3))
            layer = Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)
            layer = Conv2D(128, (4, 4), activation='relu')(layer)
            layer = Flatten()(layer)
            layer = Dense(128, activation='relu')(layer)
            layer = Dense(64, activation='relu')(layer)
            layer = Dense(1)(layer)
            model = Model(inputs, layer)
        model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        return model

    def move(self, state):
        copy_state = copy.deepcopy(state)
        self.current_game_states_opp.append(copy_state)

        if np.random.random() < self.epsilon:
            move = random.choice(Connect4.available_actions(state))
        else:
            actions = Connect4.available_actions(state)
            assert type(actions) is list
            values = []
            for action in actions:
                copy_state = copy.deepcopy(state)
                copy_state[action] = self.player
                input_data = np.dstack((copy_state, np.fliplr(copy_state), self.player_matrix))
                value = self.model.predict(np.expand_dims(input_data, axis=0))
                values.append(value)
            idx = np.argmax(values)
            move = actions[idx]
        copy_state = copy.deepcopy(state)
        copy_state[move] = self.player
        self.current_game_states.append(copy_state)
        return move


    def update_memory(self, reward):
        # Updates memory if terminal state is reached
        for state_self, state_opp in zip(reversed(self.current_game_states), reversed(self.current_game_states_opp)):
            input_data_self = np.dstack((state_self, np.fliplr(state_self), self.player_matrix))
            self.memory.append((input_data_self, reward))

            input_data_opp = np.dstack((state_opp, np.fliplr(state_opp), -self.player_matrix))
            self.memory.append((input_data_opp, -reward))

            reward *= DISCOUNT
        self.current_game_states = []
        self.current_game_states_opp = []


    def train(self):
        if len(self.memory) < self.min_memory_size:
            # don't do anything here
            return

        # sample minibatch from replay memory, each a tuple of (current_state, move, reward, new_state, terminal_flag)
        minibatch = random.sample(self.memory, self.minibatch_size)
        X = []
        Y = []

        for x, y in minibatch:
            X.append(x)
            Y.append(y)

        self.model.fit(np.array(X), np.array(Y), batch_size=self.minibatch_size, verbose=0)




