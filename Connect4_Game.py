import pygame
import sys
from AlphaZero.AlphaZero_backend import Game, AlphaZeroConfig, ResNet
from AlphaZero.Pit import AgentZeroCompetitive
from AlphaZero.AlphaZeroMCTS import *

game = Game()

black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
yellow = (255, 255, 0)
square_size = 100
width = game.columns * square_size
height = (game.rows+1) * square_size
size = (width, height)
radius = int(square_size/2)
screen = pygame.display.set_mode(size)


def draw_board(game):
    for c in range(game.columns):
        for r in range(game.rows):
            pygame.draw.rect(screen, blue, (c * square_size, r * square_size + square_size, square_size, square_size))
            pygame.draw.circle(screen, black, (
            int(c * square_size + square_size / 2), int(r * square_size + square_size + square_size / 2)), radius)

    for c in range(game.columns):
        for r in range(game.rows):
            if game.state()[r][c] == 1:
                pygame.draw.circle(screen, red, (
                int(c * square_size + square_size / 2),  int(r * square_size + square_size + square_size / 2)), radius)
            elif game.state()[r][c] == -1:
                pygame.draw.circle(screen, yellow, (
                int(c * square_size + square_size / 2), int(r * square_size + square_size + square_size / 2)), radius)
    pygame.display.update()


class HumanPlayer:
    def move(self, game):
        col = int(input('Columns 1-7'))
        game.apply(col-1)
        return game


game_over = False
weights1 = r'/Users/timwu/models/AlphaZeroResNet/920__loss_3.475420832633972.h5'
weights2 = r'/Users/timwu/models/AlphaZeroResNet/episode_700__winrate_100.0_loss_3.5536495447158813.h5'
player1 = AgentZeroCompetitive(config=AlphaZeroConfig(), net=ResNet(weights1), mcts=True)
player2 = AgentZeroCompetitive(config=AlphaZeroConfig(), net=ResNet(weights2), mcts=True)

pygame.init()
draw_board(game)
pygame.display.update()

while not game_over:

    if game.to_play == 0:

        game = player1.move(game)

        if game.terminal is True:
            print(f'Winner is Player {game.to_play+1}')
            draw_board(game)
            game_over = True

    elif game.to_play == 1:

        game = player2.move(game)
        if game.terminal is True:
            print(f'Winner is Player {game.to_play+1}')
            draw_board(game)
            game_over = True

    draw_board(game)
    print(f'{game_over=}')
    if game_over:
        pygame.time.wait(5000)