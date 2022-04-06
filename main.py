from collections import defaultdict

from LiteModel import LiteModel
from board import *
from machineplayer import MachinePlayer
from renderer import Hex_Renderer
from randomplayer import RandomPlayer
from time import sleep
from util import *
import random as random
import math
import tensorflow as tf
from keras.models import load_model


# Training on a 11x11
def generate_MCTS_table():
    size = 7
    renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)

    white_wins = 0
    black_wins = 0
    p1 = MachinePlayer('black', board_size=size)
    p2 = MachinePlayer('white', board_size=size)
    for i in range(5000):

        board = Board(size=size)
        renderer.update_hexes(board.tiles)
        running = True
        while running:
            board.place(p1.get_move_MCTS(board), p1.color)
            renderer.update_hexes(board.tiles)
            if board.check_win(p1.color):
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

        print("Total positions evaluated: ", len(MachinePlayer.position_lookup_table), " Overlaps: ", p1.overlaps)
        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
        renderer.kill()

    text_file = open("solved_positions3.txt", "wt")
    text_file.write(str(MachinePlayer.solved_positions))
    text_file.close()


def neural_training_loop(size, swap_file, solver_file, episodes=10, epochs=50):
    renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)

    white_wins = 0
    black_wins = 0

    # convert to LiteModel for faster evaluation #
    p1 = MachinePlayer('black', board_size=size)
    p1.model.save(swap_file)
    ##

    for epoch in range(epochs):
        # Reload models for the new epoch
        p1 = load_machine_player(swap_file, 'black', size)
        p2 = load_machine_player(swap_file, 'white', size)

        # Reset our training data
        position_lookup_table = defaultdict(lambda: [])

        for game in range(episodes):
            positions = []

            board = Board(size=size)
            renderer.update_hexes(board.tiles)
            running = True
            black_won = False
            while running:
                board.place(p1.get_move_network(board, max_depth=5, branching_factor=4), p1.color)
                print("Board eval: ", p1.model.predict_single(board.get_integer_representation().flatten()))
                positions.append(board.get_integer_representation())
                renderer.update_hexes(board.tiles)
                if board.check_win(p1.color):
                    black_won = True
                    black_wins += 1
                    print(p1.color + " wins!")
                    running = False

                if running:
                    board.place(p2.get_move_network(board, max_depth=5, branching_factor=4), p2.color)
                    positions.append(board.get_integer_representation())
                    renderer.update_hexes(board.tiles)

                if board.check_win(p2.color):
                    print(p2.color + " wins!")
                    running = False
                    white_wins += 1

            val = 1 if black_won else -1
            position_evals = val * np.array([p1.gamma ** (len(positions) - i) for i in range(len(positions))])
            for j in range(len(positions)):
                position_lookup_table[positions[j].tostring()] = position_evals[j]

            print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
            renderer.kill()
        solved_positions = {}
        for key in position_lookup_table.keys():
            solved_positions[key] = (np.average(position_lookup_table[key]), np.std(position_lookup_table[key]))
        text_file = open(solver_file, "wt")
        text_file.write(str(solved_positions))
        text_file.close()
        train_model(swap_file, solver_file)



def play_sample_games(p1, p2):
    size = 4
    renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)

    white_wins = 0
    black_wins = 0

    for i in range(100):
        board = Board(size=size)
        renderer.update_hexes(board.tiles)
        running = True
        while running:
            # board.place(p1.get_move_MCTS(board), p1.color)
            board.place(p1.get_move_network(board, max_depth=50, branching_factor=1), p1.color)
            renderer.update_hexes(board.tiles)
            print("Board eval: ", p1.model.predict_single(board.get_integer_representation().flatten()))
            if board.check_win(p1.color):
                black_wins += 1
                print(p1.color + " wins!")
                running = False

            if running:
                board.place(p2.get_move_MCTS(board), p2.color)
                # board.place(p2.get_move_network(board, max_depth=3, branching_factor=3), p2.color)
                # board.place((int(input()), int(input())), p2.color)
                renderer.kill()
                renderer.update_hexes(board.tiles)

            sleep(1)
            if board.check_win(p2.color):
                print(p2.color + " wins!")
                running = False
                white_wins += 1

        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
        renderer.kill()


def train_model(model_file, solver_file):
    model = MachinePlayer('white', board_size=4)
    boards, scores = parse_solver_file(solver_file)
    training_indices = random.sample([i for i in range(len(boards))], int(len(boards)))
    testing_indices = random.sample([i for i in range(len(boards))], int(len(boards)))

    training_inputs = np.array([boards[i].get_integer_representation().flatten() for i in training_indices])
    testing_inputs = np.array([boards[i].get_integer_representation().flatten() for i in testing_indices])

    training_outputs = np.array([scores[i][0] for i in training_indices])
    testing_outputs = np.array([scores[i][0] for i in testing_indices])

    bias = 0
    fixed_training_outputs = []
    fixed_training_inputs = []
    for i in range(len(training_inputs)):
        if bias == 0:
            fixed_training_inputs.append(training_inputs[i])
            fixed_training_outputs.append(training_outputs[i])
            bias += np.sign(training_outputs[i])
        elif bias + np.sign(training_outputs[i]) == 0:
            fixed_training_inputs.append(training_inputs[i])
            fixed_training_outputs.append(training_outputs[i])
            bias += np.sign(training_outputs[i])

    training_inputs = np.array(fixed_training_inputs)
    training_outputs = np.array(fixed_training_outputs)
    print("Training Data dims: ", np.shape(training_inputs))
    print("training data bias: ", np.average(training_outputs))

    model.model.fit(x=training_inputs, y=training_outputs, validation_split=0.1, epochs=150)

    predicted = model.model.predict(testing_inputs)
    print("done training")
    print(predicted[:10])
    print(testing_outputs[:10])
    model.model.save(model_file)


def load_machine_player(filename, side, size):
    player = MachinePlayer(side, board_size=size)
    # player.model = load_model(filename)
    player.model = LiteModel.from_keras_model(load_model(filename))
    load_model(filename).summary()
    return player


if __name__ == '__main__':
    # generate_MCTS_table()
    # train_model()
    neural_training_loop(4, "training_value_network", "training_solved_positions.txt")

    # p1 = load_machine_player("value_network", 'black')
    # p2 = load_machine_player("value_network", 'white')
    # play_sample_games(p1, p2)
