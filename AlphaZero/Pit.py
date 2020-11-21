import copy
import random
from AlphaZero.AlphaZero_backend import Game
from AlphaZero.AlphaZeroMCTS import *


class Pit:
    def __init__(self, p1, p2, num_sims, verbose=False):
        self.p1 = p1
        self.p2 = p2
        self.num_sims = num_sims
        self.verbose = verbose
        self.stats = dict()
        self.stats[self.p1] = 0
        self.stats[self.p2] = 0
        self.simulate()

    def simulate(self):
        for i in range(self.num_sims):
            game = Game()
            if i % 2 == 0:
                player1, player2 = self.p1, self.p2
            else:
                player1, player2 = self.p2, self.p1

            while not game.terminal:
                # p1 move
                game = player1.move(game)
                if game.terminal:
                    if self.verbose:
                        print(game)
                    if len(game.legal_actions) != 0:
                        self.stats[player1] += 1
                    break
                # p2 move
                game = player2.move(game)
                if game.terminal:
                    if self.verbose:
                        print(game)
                    if len(game.legal_actions) != 0:
                        self.stats[player2] += 1
                    break

    @property
    def p1_winrate(self):
        return self.stats[self.p1] / self.num_sims * 100

    @property
    def p2_winrate(self):
        return self.stats[self.p2] / self.num_sims * 100

    def __repr__(self):
        return str(self.stats.items())

class SmartRandomAgent:
    ''' Makes random moves, unless it is one move away from victory, in which case, it will always choose that move
           If opponent's next optimal move is a winning move, it will play the move that blocks the win'''

    def __init__(self, dumb=False):
        self.game = None
        self.dumb = dumb

    @property
    def self_winning_move(self):
        token = 1 if self.game.to_play == 0 else -1
        move_col = None
        for i in self.game.legal_actions:
            copy_state = copy.deepcopy(self.game.state())
            move = self.game.lowest_position(copy_state, i)
            copy_state[move] = token
            if self.game.terminal_value(self.game.to_play, copy_state) != 0:
                move_col = i
        return move_col

    @property
    def opp_winning_move(self):
        token = 1 if self.game.to_play == 0 else -1
        move_col = None
        for i in self.game.legal_actions:
            copy_state = copy.deepcopy(self.game.state())
            move = self.game.lowest_position(copy_state, i)
            copy_state[move] = -token
            if self.game.terminal_value(-self.game.to_play, copy_state) != 0:
                move_col = i
        return move_col

    def move(self, game):
        if self.dumb:
            game.apply(random.choice(game.legal_actions))
            return game
        self.game = game.clone()
        opp_win_move = self.opp_winning_move
        if opp_win_move is not None:
            game.apply(opp_win_move)
            return game
        self_win_move = self.self_winning_move
        if self_win_move is not None:
            game.apply(self_win_move)
            return game
        game.apply(random.choice(self.game.legal_actions))
        return game


class AgentZeroCompetitive:
    def __init__(self, config, net=None, mcts=False):
        self.config = config
        self.net = net
        self.config.num_sampling_moves = 0
        self.config.root_exploration_fraction = 0
        self.mcts = mcts

    def move(self, game):
        if self.mcts:
            action, root = run_mcts(self.config, game, self.net)
            game.store_search_statistics(root)
        else:
            img = game.make_image()
            _, prob = self.net.inference(np.expand_dims(img, axis=0))
            prob = [p if idx in game.legal_actions else 0 for idx, p in enumerate(np.squeeze(prob))]
            prob /= sum(prob)

            action = np.argmax(prob)
        game.apply(action)
        return game