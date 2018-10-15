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

    def calculate_output(self, input_data):
        self.input_data = input_data
        activation = self.calculate_sum()
        return 1 if activation > self.activation_limit else -1

    def learn_from_result(self, output_result, actual_result):
        error = actual_result - output_result
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + self.input_data[i] * error * self.learning_rate

    def train(self, input_data, desired_output):
        guess = self.calculate_output(input_data)
        self.learn_from_result(guess, desired_output)


def get_desired_val_for_happy(val):
    return 1 if val == 1 else -1


def get_desired_val_for_sad(val):
    return 1 if val == 2 else -1


def get_desired_val_for_mischievous(val):
    return 1 if val == 3 else -1


def get_desired_val_for_mad(val):
    return 1 if val == 4 else -1


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


def rotate_smiley(list):
    if not 31 in list[:9]:
        return 2

    rotate = True
    for row in list:
        if 31 in row[:8]:
            rotate = False

    if rotate:
        return 3

    rotate = True
    for row in list:
        if 31 in row[12:]:
            rotate = False

    if rotate:
        return 1

    return 0


if __name__ == "__main__":

    train_data = sys.argv[1]
    train_data_labels = sys.argv[2]
    test_data = sys.argv[3]

    debug = True

    train_data = parser.create_dictionary_for_images(train_data)
    answers = parser.create_dictionary_for_labels(train_data_labels)
    test_data = parser.create_dictionary_for_images(test_data)

    for key in train_data:
        train_data[key] = np.reshape(train_data[key], (20, 20))
        numrots = rotate_smiley(train_data[key])
        np.rot90(train_data[key], numrots)
        train_data[key] = train_data[key].ravel()

    for key in test_data:
        test_data[key] = np.reshape(test_data[key], (20, 20))
        numrots = rotate_smiley(test_data[key])
        np.rot90(test_data[key], numrots)
        test_data[key] = test_data[key].ravel()

    normalize_list(train_data)
    normalize_list(test_data)

    learning_rate = 0.2
    size_of_data = 400

    size_of_data_hidden = 4

    percept_happy = [Perceptron(learning_rate, size_of_data) for _ in range(0, 12)]
    percept_sad = [Perceptron(learning_rate, size_of_data) for _ in range(0, 12)]
    percept_mischievous = [Perceptron(learning_rate, size_of_data) for _ in range(0, 12)]
    percept_mad = [Perceptron(learning_rate, size_of_data) for _ in range(0, 12)]

    percept_happy_hidden = Perceptron(learning_rate, size_of_data_hidden)
    percept_sad_hidden = Perceptron(learning_rate, size_of_data_hidden)
    percept_mischievous_hidden = Perceptron(learning_rate, size_of_data_hidden)
    percept_mad_hidden = Perceptron(learning_rate, size_of_data_hidden)

    for key in train_data:
        for happy, sad, mis, mad in zip(percept_happy, percept_sad, percept_mischievous, percept_mad):
            happy.train(train_data[key], get_desired_val_for_happy(answers[key]))
            sad.train(train_data[key], get_desired_val_for_sad(answers[key]))
            mis.train(train_data[key], get_desired_val_for_mischievous(answers[key]))
            mad.train(train_data[key], get_desired_val_for_mad(answers[key]))

    keys = test_data.keys()
    num_correct = 0
    correct_answer = 0
    for key in sorted(keys, key=sort_by_image_num):

        num_happy = 0
        num_sad = 0
        num_mis = 0
        num_mad = 0

        for happy, sad, mis, mad in zip(percept_happy, percept_sad, percept_mischievous, percept_mad):
            num_happy += happy.calculate_output(test_data[key])
            num_sad += sad.calculate_output(test_data[key])
            num_mis += mis.calculate_output(test_data[key])
            num_mad += mad.calculate_output(test_data[key])

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
            if correct_answer == answers[key]:
                num_correct += 1
        num_happy = 0
        num_sad = 0
        num_mis = 0
        num_mad = 0

    if debug:
        print("percept was ", num_correct, " out of", len(answers), "correct after training.")
