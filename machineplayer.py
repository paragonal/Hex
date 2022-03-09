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


class MachinePlayer(Player):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)



    def __init__(self, color, board_size=7):
        super().__init__(color)
        self.board_size = board_size
        self.thread_lock = threading.Lock()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = .99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 1

        self.model = self.make_model()

        self.memory_size = 100
        self.inputs = []
        self.outputs = []
        self.rewards = []
        self.targets = []

        self.branching_factor = 1
        self.tree_discount_factor = 0.95
        self.calls = 0

        self.model.summary()

    def make_model(self):
        model = Sequential()
        model.add(Dense(self.board_size ** 2, input_dim=self.board_size ** 2))
        model.add(Dense(self.board_size ** 2))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Softmax())
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def train(self, reward):
        print("Training for episode!")
        width = len(self.inputs)

        # every move in the game gets a reward based on how recent it was
        #self.rewards.extend(np.linspace(0, reward, width))
        self.rewards.extend([reward/width for _ in range(width)])

        X = np.array(self.inputs).reshape(width, self.board_size ** 2)
        outputs = np.array(self.outputs).reshape(width, self.board_size ** 2)

        target_qs = []

        for frame in range(len(X) - 1):
            # Q learning equation to determine new Q
            Q = outputs[frame] + self.learning_rate \
                * (self.gamma * outputs[frame + 1] + self.rewards[frame] - outputs[frame])
            Q = np.array(Q) / sum(Q)
            target_qs.append(Q)

        print("Difference: ", sum((Q-outputs[frame])**2))

        target_qs = np.array(target_qs).reshape(width - 1, self.board_size ** 2)
        # Try to fit model to new Q
        X = X[:-1]
        self.model.fit(x=X
                       , y=target_qs
                       , callbacks=[MachinePlayer.cp_callback])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print("New epsilon=%f" % self.epsilon)
        self.inputs = self.inputs[-self.memory_size:]
        self.outputs = self.outputs[-self.memory_size:]
        self.rewards = self.rewards[-self.memory_size:]

    # Use a Monte Carlo Search Tree to get down to a final position to evaluate a node's value
    # black win = eval -1
    # white win = eval 1
    def eval_node(self, board, active_color):

        if self.calls % 100 == 0: print("Calls: ", self.calls)
        self.calls += 1
        if board.check_win('black'):
            return 1
        if board.check_win('white'):
            return -1
        moves = self.select_moves(board, self.branching_factor)
        boards = []
        values = []
        next_color = 'white' if active_color == 'black' else 'black'
        for move in moves:
            temp = board.clone()
            temp.place(move, next_color)
            boards.append(temp)
        for board in boards:
            values.append(self.tree_discount_factor * self.eval_node(board, next_color))
        return sum(values)

    # select our moves for MCTS, this will eventually be based on the network
    def select_moves(self, board, n):
        # randomly choose moves
        if n > len(board.get_legal_moves()):
            return board.get_legal_moves()
        return sample(board.get_legal_moves(), n)

    def get_move_MCTS(self, board):
        max_move = (None, -2) # tuples with the move and eval
        min_move = (None, 2)

        for move in board.get_legal_moves():
            temp = board.clone()
            temp.place(move, self.color)
            val = self.eval_node(temp, self.color)
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
                if board.can_place([i,j]):
                    empty_slots.append([i,j])
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
