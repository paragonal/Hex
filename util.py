import numpy as np
from tqdm import tqdm
from board import Board
from renderer import Hex_Renderer


def parse_solver_file(filename, visualize=False):
    size = 4
    if visualize:
        renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)
        renderer.kill()
    with open(filename, 'r') as f:
        out_map = {}
        data = f.readline()[1:-1]
        data = data.split(", '")
        # print(data[:5])
        print("Loading %d solved positions:" % len(data))
        count = 0
        for l in tqdm(data):
            count += 1
            board, score = parse_board(l)
            out_map[str(board.get_integer_representation())] = (score[0], score[1])

            # add flipped board to expand training data
            for b in board.generate_flips():
                out_map[str(b.get_integer_representation())] = (score[0], score[1])
            # add rotations

        if visualize:
            for l in data[-500:]:
                print("~")
                renderer.update_hexes(parse_board(l)[0].tiles)
                print(parse_board(l)[1])
                input()
                renderer.kill()
    f.close()

    return out_map


def parse_board(s):
    board, score = s.split(":")
    board, score = [[int(y) for y in
                     x[:-1].replace("[ ", "").replace("]", "").replace("  ", " ").replace("[", "").split(" ")]
                    for x in (board[1:-1].split("\\n "))], [float(x) for x in score[2:-1].split(", ")]
    out = Board(len(board))
    for i in range(len(board)):
        for j in range(len(board[0])):
            out.tiles[i][j].state = board[i][j]
    out.recalc_legal_moves()
    return out, score


if __name__ == '__main__':
    parse_solver_file("solved_positions2.txt", visualize=True)
