from board import *
from machineplayer import MachinePlayer
from renderer import Hex_Renderer
from randomplayer import RandomPlayer
from time import sleep


# Training on a 5x5
def start():
    size = 4
    renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)

    white_wins = 0
    black_wins = 0
    p1 = MachinePlayer('black', board_size=size)
    p2 = MachinePlayer('white', board_size=size)
    for i in range(5000):
        black_win = False

        board = Board(size=size)
        renderer.update_hexes(board.tiles)
        running = True
        while running:
            # sleep(.5)
            board.place(p1.get_move_MCTS(board), p1.color)
            renderer.update_hexes(board.tiles)
            if board.check_win(p1.color):
                black_win = True
                black_wins += 1
                print(p1.color + " wins!")
                running = False

            if running:
                board.place(p2.get_move_MCTS(board), p2.color)
                renderer.update_hexes(board.tiles)

            if board.check_win(p2.color):
                print(p2.color + " wins!")
                running = False
                white_wins += 1

        # if black_win:
        #     p1.train(1)
        # else:
        #     p2.train(1)
        print("Total positions evaluated: ", len(MachinePlayer.position_lookup_table), " Overlaps: ", p1.overlaps)
        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
        renderer.kill()

    text_file = open("solved_positions3.txt", "wt")
    text_file.write(str(MachinePlayer.solved_positions))
    text_file.close()


if __name__ == '__main__':
    start()
