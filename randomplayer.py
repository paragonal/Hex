from player import Player
from random import random


class RandomPlayer(Player):
    def __init__(self, color, board_size):
        super().__init__(color)
        self.size = board_size

    def place(self, board):
        p = (int(random() * board.size), int(random() * board.size))
        while not board.can_place(p):
            p = (int(random() * board.size), int(random() * board.size))
        return p
