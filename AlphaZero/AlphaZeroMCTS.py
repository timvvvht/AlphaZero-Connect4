import numpy as np
import math
import tensorflow as tf

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    @property
    def is_expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __repr__(self):
        return f'Node Value: {self.value}\n' \
               f'Node Prior: {self.prior}\n' \
               f'Node Visits : {self.visit_count}'


def run_mcts(config, game, network, add_noise=True):
    root = Node(0)
    evaluate(root, game, network)
    if add_noise:
        add_exploration_noise(config, root)

    for i in range(config.num_simulations):
        # print(f'sim{i}')
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.is_expanded:
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)
        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play)
    return select_action(config, game, root), root


def select_action(config, game, root):
    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


def select_child(config, node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value
    return prior_score + value_score


def evaluate(node, game, network):
    value, policy_logits = network.inference(game.make_image())

    if game.terminal:
    # return actual value of state if terminal state is reached
        value = game.terminal_value(game.to_play)
    node.to_play = game.to_play
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


def backpropagate(search_path: list, value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1


def add_exploration_noise(config, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def softmax_sample(visit_counts):
    visits, actions = zip(*visit_counts)
    visits = np.array(visits).astype('float64')
    prob = tf.nn.softmax(visits)
    idx = np.random.choice(len(actions), p=prob)
    return None, actions[idx]