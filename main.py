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

# Training on a 5x5
def generate_MCTS_table():
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

        print("Total positions evaluated: ", len(MachinePlayer.position_lookup_table), " Overlaps: ", p1.overlaps)
        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
        renderer.kill()

    text_file = open("solved_positions3.txt", "wt")
    text_file.write(str(MachinePlayer.solved_positions))
    text_file.close()


def play_sample_games(p1, p2):

    size = 4
    renderer = Hex_Renderer(height=size * 30 * 2, width=(2 + size) * 30)

    white_wins = 0
    black_wins = 0

    for i in range(10):
        board = Board(size=size)
        renderer.update_hexes(board.tiles)
        running = True
        while running:
            # board.place(p1.get_move_MCTS(board), p1.color)
            board.place(p1.get_move_network(board, max_depth=3, branching_factor=3), p1.color)
            renderer.update_hexes(board.tiles)
            print("Board eval: ", p1.model.predict_single(board.get_integer_representation().flatten()))
            if board.check_win(p1.color):
                black_wins += 1
                print(p1.color + " wins!")
                running = False

            if running:
                board.place(p2.get_move_MCTS(board), p2.color)
                # board.place(p2.get_move_network(board, max_depth=5, branching_factor=1), p2.color)
                renderer.update_hexes(board.tiles)
            sleep(1)
            if board.check_win(p2.color):
                print(p2.color + " wins!")
                running = False
                white_wins += 1

        print("White wins: %d, Black Wins: %d" % (white_wins, black_wins))
        renderer.kill()


def train_model():
    model = MachinePlayer('white', board_size=4)
    boards, scores = parse_solver_file("solved_positions2.txt")
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

    model.model.fit(x=training_inputs, y=training_outputs, validation_split=0.1, epochs=100)

    predicted = model.model.predict(testing_inputs)
    print("done training")
    print(predicted[:10])
    print(testing_outputs[:10])
    model.model.save("value_network")

def load_machine_player(filename, side):
    player = MachinePlayer(side, board_size=4)
    # player.model = load_model(filename)
    player.model = LiteModel.from_keras_model(load_model(filename))
    return player

if __name__ == '__main__':
    #generate_MCTS_table()
    #train_model()

    p1 = load_machine_player("value_network", 'black')
    p2 = load_machine_player("value_network", 'white')
    play_sample_games(p1, p2)

