# 1. Initialize wegihts and threshold to small random numbers.
# 2. Present a vector x to the neruon inputs and calculate the output.
# 3. Update the weights according to: 
#       d is desired output
#       t is the iteration number
#       eta is the gain or step size, where 0.0 < n < 1.0
# 4. Repeat steps 2 and 3 until:
#       the iteration error is less than a user-specified error threshold or
#       a predetermined number of iterations have been completed.

# correct input : python3 faces.py images.txt keys.txt test.txt
#       images.txt = images to train on
#       keys.txt = correct classifications
#       test.txt = test classification on this

# correct output : When executing your solution as described above, 
#       the program should print text to standard out (normally the screen)
#       that are completely compliant with the key-file section of the 
#       ascii-based file format. This means that every line that is not 
#       part of the classification result should be marked as a comment.
#       This is important since your solution will be automatically 
#       executed and the result parsed as if it was a key file

# Faces
#   Happy : 1
#   Sad : 2
#   Mischievous : 3
#   Mad : 4

#   32 levels of grey where 0 is white and 31 black

# dict = parser.create_dictionary_for_images("corrupt_data.txt")
# print(dict["Image2"])

# dict = parser.create_dictionary_for_labels("correct_answers.txt")
# print(dict)

import sys
import file_parser as parser
import numpy as np
import random
import math
import copy


class Perceptron:
    input_data = []

    def __init__(self, learning_rate, size_of_input):
        self.learning_rate = learning_rate
        self.weights = self.generate_random_weights(size_of_input)
        self.bias = 1
        self.activation_limit = 0

    def calculate_sum(self):
        return np.dot(self.weights, self.input_data) + self.bias

    def generate_random_weights(self, size):
        weight_list = []
        for i in range(0, size):
            weight_list.append(random.uniform(-1, 1))
        return weight_list

    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def activation_function(self, input_data):
        self.input_data = input_data
        activation = self.calculate_sum()
        # return 1 if activation > self.activation_limit else -1
        return self.sigmoid(activation)

    def learn_from_result(self, output_result, actual_result):
        error = actual_result - output_result
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + self.input_data[i] * error * self.learning_rate

    def train(self, input_data, desired_output):

        guess = self.activation_function(input_data)
        self.learn_from_result(guess, desired_output)


def get_desired_val_for_happy(val):
    return 1 if val == 1 else 0


def get_desired_val_for_sad(val):
    return 1 if val == 2 else 0


def get_desired_val_for_mischievous(val):
    return 1 if val == 3 else 0


def get_desired_val_for_mad(val):
    return 1 if val == 4 else 0


def print_if_not_debug(debug, key, val):
    if not debug:
        print(str(key) + " " + str(val))


def sort_by_image_num(a):
    return int(a[5:])


def normalize_list(data):
    for key2 in data:
        for i in range(0, len(data[key2])):
            if data[key2][i] > 10:
                data[key2][i] = 1
            else:
                data[key2][i] = 0


def normalize_list_eyebrows(data):
    for key2 in data:
        for i in range(0, len(data[key2])):
            if data[key2][i] == 31:
                data[key2][i] = 1
            else:
                data[key2][i] = 0


def rotate_smiley(list):
    if not 31 in list[:9]:
        return 2

    rotate = True
    for row in list:
        if 31 in row[:8]:
            rotate = False

    if rotate:
        return 1

    rotate = True
    for row in list:
        if 31 in row[12:]:
            rotate = False

    if rotate:
        return 3

    return 0


if __name__ == "__main__":

    train_data = sys.argv[1]
    train_data_labels = sys.argv[2]
    test_data = sys.argv[3]

    debug = True

    train_data = parser.create_dictionary_for_images(train_data)
    training_answers = parser.create_dictionary_for_labels(train_data_labels)
    if debug:
        test_answers = parser.create_dictionary_for_labels("test_answers.txt")
    test_data = parser.create_dictionary_for_images(test_data)

    flip_data = parser.create_dictionary_for_images("flip_training.txt")
    flip_data_answers = parser.create_dictionary_for_labels("flip_answers.txt")

    normalize_list(train_data)
    normalize_list(test_data)

    learning_rate = 0.5
    size_of_data = 400

    size_of_data_hidden = 4

    normalize_list_eyebrows(flip_data)

    percept_rotate_zero = Perceptron(learning_rate, size_of_data)
    percept_rotate_one = Perceptron(learning_rate, size_of_data)
    percept_rotate_two = Perceptron(learning_rate, size_of_data)
    percept_rotate_three = Perceptron(learning_rate, size_of_data)

    train_data_copy = copy.copy(train_data)
    test_data_copy = copy.copy(test_data)

    normalize_list_eyebrows(train_data_copy)
    normalize_list_eyebrows(test_data_copy)

    for _ in range(0, 1):
        for key in flip_data:
            percept_rotate_zero.train(flip_data[key], get_desired_val_for_happy(flip_data_answers[key] + 1))
            percept_rotate_one.train(flip_data[key], get_desired_val_for_happy(flip_data_answers[key] + 1))
            percept_rotate_two.train(flip_data[key], get_desired_val_for_happy(flip_data_answers[key] + 1))
            percept_rotate_three.train(flip_data[key], get_desired_val_for_happy(flip_data_answers[key] + 1))

    for key1 in train_data:
        rot_zero = percept_rotate_zero.activation_function(train_data_copy[key1])
        rot_one = percept_rotate_one.activation_function(train_data_copy[key1])
        rot_two = percept_rotate_two.activation_function(train_data_copy[key1])
        rot_three = percept_rotate_three.activation_function(train_data_copy[key1])

        if rot_zero > max(rot_one, rot_two, rot_three):
            numrots = 0
        elif rot_one > max(rot_two, rot_three):
            numrots = 1
        elif rot_two > rot_three:
            numrots = 2
        else:
            numrots = 3

        train_data[key1] = np.reshape(train_data[key1], (20, 20))
        train_data[key1] = np.rot90(train_data[key1], numrots)
        train_data[key1] = train_data[key1].ravel()

        # print key1
        # print rot_zero
        # print rot_two
        # print

    for key1 in test_data:
        rot_zero = percept_rotate_zero.activation_function(test_data_copy[key1])
        rot_one = percept_rotate_one.activation_function(test_data_copy[key1])
        rot_two = percept_rotate_two.activation_function(test_data_copy[key1])
        rot_three = percept_rotate_three.activation_function(test_data_copy[key1])

        if rot_zero > max(rot_one, rot_two, rot_three):
            numrots = 0
        elif rot_one > max(rot_two, rot_three):
            numrots = 1
        elif rot_two > rot_three:
            numrots = 2
        else:
            numrots = 3


        test_data[key1] = np.reshape(test_data[key1], (20, 20))
        test_data[key1] = np.rot90(test_data[key1], numrots)
        test_data[key1] = test_data[key1].ravel()

    percept_happy = Perceptron(learning_rate, size_of_data)
    percept_sad = Perceptron(learning_rate, size_of_data)
    percept_mischievous = Perceptron(learning_rate, size_of_data)
    percept_mad = Perceptron(learning_rate, size_of_data)

    for _ in range(0, 5):
        for key in train_data:
            percept_happy.train(train_data[key], get_desired_val_for_happy(training_answers[key]))
            percept_sad.train(train_data[key], get_desired_val_for_sad(training_answers[key]))
            percept_mischievous.train(train_data[key], get_desired_val_for_mischievous(training_answers[key]))
            percept_mad.train(train_data[key], get_desired_val_for_mad(training_answers[key]))

    keys = test_data.keys()
    num_correct = 0
    correct_answer = 0
    for key in sorted(keys, key=sort_by_image_num):

        num_happy = 0
        num_sad = 0
        num_mis = 0
        num_mad = 0

        num_happy += percept_happy.activation_function(test_data[key])
        num_sad += percept_sad.activation_function(test_data[key])
        num_mis += percept_mischievous.activation_function(test_data[key])
        num_mad += percept_mad.activation_function(test_data[key])

        if num_happy > max(num_sad, num_mis, num_mad):
            correct_answer = 1
        elif num_sad > max(num_mis, num_mad):
            correct_answer = 2
        elif num_mis > num_mad:
            correct_answer = 3
        else:
            correct_answer = 4

        print_if_not_debug(debug, key, correct_answer)
        if debug:
            if correct_answer == test_answers[key]:
                num_correct += 1
    if debug:
        print("percept was ", num_correct, " out of", len(test_data), "correct after training.")
