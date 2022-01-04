from board import *
from machine_player import Machine_player
from renderer import Hex_Renderer
from random_player import Random_player
from time import sleep

#Training on a 5x5
def start():
    size = 4
    renderer = Hex_Renderer(height=size * 30 * 2, width = (2+size) * 30)

    white_wins = 0
    black_wins = 0
    p1 = Machine_player('black', board_size=size)
    p2 = Machine_player('white', board_size=size)
    for i in range(1000):
        white_win = False

        board = Board(size=size)
        renderer.update(board.tiles)
        running = True
        while running:
            board.place(p1.place(board), p1.color)
            renderer.update(board.tiles)
            if board.check_win(p1.color):
                white_win = True
                black_wins += 1
                print(p1.color + " wins!")
                running = False
            if running:
                board.place(p2.place(board), p2.color)
                renderer.update(board.tiles)
            if board.check_win(p2.color):
                print(p2.color + " wins!")
                running = False
                white_wins += 1
        #print(board)
        p1.train(0 if white_win else 1)
        p2.train(1 if white_win else 0)
        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))

        #print(board)
    while True:
        ...




if __name__ == '__main__':
    start()
