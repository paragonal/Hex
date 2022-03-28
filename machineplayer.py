from tensorflow.python.keras.optimizers import SGD

from player import Player
from keras.models import Sequential
from keras.layers import Dense, Softmax
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from random import random, sample
import os
import threading
from time import time
from collections import defaultdict


class MachinePlayer(Player):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # map an integer representation of a board to a list of evals
    position_lookup_table = defaultdict(lambda: [])
    solved_positions = {}

    def __init__(self, color, board_size=7):
        super().__init__(color)
        self.board_size = board_size
        self.thread_lock = threading.Lock()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = .99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = .0025

        self.model = self.make_model()

        self.memory_size = 100
        self.inputs = []
        self.outputs = []
        self.rewards = []
        self.targets = []

        self.branching_factor = 1
        self.tree_discount_factor = 0.95
        self.MCTS_iters = 10
        self.calls = 0
        self.overlaps = 0
        self.positions_evaled = 0
        # self.model.summary()

    def make_model(self):
        model = Sequential()
        model.add(Dense(self.board_size ** 2, input_dim=self.board_size ** 2))
        model.add(Dense(self.board_size ** 3))
        model.add(Dense(self.board_size ** 3, activation='tanh'))
        model.add(Dense(self.board_size ** 3, activation='tanh'))
        model.add(Dense(self.board_size ** 3))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss="mean_squared_error",
                      optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate))
        return model

    # evaluate a node with iterative deepening down to a certain position
    def eval_node_network(self, board, active_color, depth, max_depth, branching_factor):
        self.positions_evaled+=1
        # print(board)
        # print("evaled ", self.positions_evaled)
        if board.check_win('black'):
            return -1
        if board.check_win('white'):
            return 1

        if depth == max_depth:
            return self.model.predict(np.array([board.get_integer_representation().flatten()]))[0]

        evals = [i[0] for i in self.evaluate_moves(board, board.get_legal_moves(), active_color)]
        # get indices of branching_factor largest elements
        if branching_factor < len(evals):
            if active_color == 'black':
                candidate_indices = np.argpartition(evals, -branching_factor)[-branching_factor:]
            else:
                candidate_indices = np.argpartition(evals, branching_factor)[:branching_factor]
        else:
            candidate_indices = [i for i in range(branching_factor)]
        # search on each of these moves
        boards = []
        values = []
        next_color = 'white' if active_color == 'black' else 'black'

        if len(candidate_indices) > len(board.get_legal_moves()):
            moves = board.get_legal_moves()
        else:
            moves = [board.get_legal_moves()[i] for i in candidate_indices]

        for move in moves:
            temp = board.clone()
            temp.place(move, next_color)
            boards.append(temp)

        for board in boards:
            values.append(self.tree_discount_factor
                          * self.eval_node_network(board, next_color, depth+1, max_depth, branching_factor))

        if active_color == 'white':
            return max(values)
        else:
            return min(values)

    def get_move_network(self, board, max_depth, branching_factor):
        vals = []
        self.positions_evaled = 0
        for move in board.get_legal_moves():
            temp = board.clone()
            temp.place(move, self.color)
            vals.append(self.eval_node_network(temp, self.color, 0, max_depth, branching_factor))

        if self.color == 'black':
            return board.get_legal_moves()[np.argmin(vals)]
        else:
            return board.get_legal_moves()[np.argmax(vals)]

    def evaluate_moves(self, board, moves, active_color):
        to_eval = []
        for move in moves:
            temp = board.clone()
            temp.place(move, active_color)
            to_eval.append(temp.get_integer_representation().flatten())
        return -self.model.predict(np.array(to_eval)) #TODO take out this negative when signs fixed on lookup table

    # Use a Monte Carlo Search Tree to get down to a final position to evaluate a node's value
    # black win = eval -1
    # white win = eval 1
    def eval_node_MCST(self, board, active_color):
        self.calls += 1
        if board.check_win('black'):
            MachinePlayer.position_lookup_table[board.get_integer_representation().tostring()].append(-1)
            return -1
        if board.check_win('white'):
            MachinePlayer.position_lookup_table[board.get_integer_representation().tostring()].append(1)
            return 1

        v = MachinePlayer.position_lookup_table[board.get_integer_representation().tostring()]
        if len(v) > 10 and np.std(v) < .1:
            self.overlaps += 1
            MachinePlayer.solved_positions[str(board.get_integer_representation())] = (np.mean(v), np.std(v))
            return np.mean(v)

        moves = self.select_moves_MCTS(board, self.branching_factor)
        boards = []
        values = []
        next_color = 'white' if active_color == 'black' else 'black'
        for move in moves:
            temp = board.clone()
            temp.place(move, next_color)
            boards.append(temp)
        for board in boards:
            values.append(self.tree_discount_factor * self.eval_node_MCST(board, next_color))

        MachinePlayer.position_lookup_table[board.get_integer_representation().tostring()].append(np.mean(values))

        return np.mean(values)

    # select our moves for MCTS, this will eventually be based on the network
    def select_moves_MCTS(self, board, n):
        # randomly choose moves
        if n > len(board.get_legal_moves()):
            return board.get_legal_moves()
        return sample(board.get_legal_moves(), n)

    def get_move_MCTS(self, board):
        max_move = (None, -2)  # tuples with the move and eval
        min_move = (None, 2)

        for move in board.get_legal_moves():
            vals = []
            # do a bunch of fully random searches from root node to evaluate it
            for _ in range(self.MCTS_iters):
                temp = board.clone()
                temp.place(move, self.color)
                vals.append(self.eval_node_MCST(temp, self.color))

            val = np.mean(vals)

            if val > max_move[1]:
                max_move = (move, val)
            if val < min_move[1]:
                min_move = (move, val)

        if self.color == 'black':
            return min_move[0]
        else:
            return max_move[0]

    def get_move(self, board):
        side = 1 if self.color == 'black' else -1
        input = side * np.array(board.get_integer_representation())
        e = input.flatten().reshape(1, board.size ** 2)
        rewards_table = self.model.predict(e)[0].reshape(board.size, board.size)

        # find element that gives max probability
        prob_max = -1
        position = []
        empty_slots = []
        for i in range(len(rewards_table)):
            for j in range(len(rewards_table[i])):
                if board.can_place([i, j]):
                    empty_slots.append([i, j])
                    if rewards_table[i][j] > prob_max:
                        position = [i, j]
                        prob_max = rewards_table[i][j]

        # pick randomly if we are < epsilon
        if random() < self.epsilon:
            position = sample(empty_slots, 1)[0]

        print("Color %s, max_prob: %f" % (self.color, prob_max))
        if not board.can_place(position):
            print("FAIL")

        self.inputs.append(input)
        self.outputs.append(np.array(rewards_table))

        return position
