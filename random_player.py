from player import Player
from random import random


class Random_player(Player):
    def __init__(self, color):
        super().__init__(color)

    def place(self, board):
        p = [int(random() * board.size), int(random() * board.size)]
        while not board.can_place(p):
            p = [int(random() * board.size), int(random() * board.size)]
        return p
