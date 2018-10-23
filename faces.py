import sys
import file_parser as parser
import numpy as np
import random
import math


class Perceptron:
    def __init__(self, learn_rate, size_of_input):
        self.learning_rate = learn_rate
        self.weights = self.generate_random_weights(size_of_input)
        self.bias = 1
        self.activation_limit = 0

    def calculate_sum(self, input_data):
        return np.dot(self.weights, input_data) + self.bias

    def generate_random_weights(self, size):
        return [random.uniform(0, 1) for _ in range(size)]

    def sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def calculate_output(self, input_data):
        return self.sigmoid(self.calculate_sum(input_data))

    def learn_from_result(self, input_data, output_result, actual_result):
        error = actual_result - output_result
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + input_data[i] * error * self.learning_rate

    def train(self, input_data, desired_output):
        self.learn_from_result(input_data, self.calculate_output(input_data), desired_output)


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


def should_rotate(list_to_rotate):
    return True if 31 in list_to_rotate[15 * 20:] else False


if __name__ == "__main__":

    debug = False

    train_data = parser.create_dictionary_for_images(sys.argv[1])
    training_answers = parser.create_dictionary_for_labels(sys.argv[2])
    if debug:
        test_answers = parser.create_dictionary_for_labels("test_answers.txt")
    test_data = parser.create_dictionary_for_images(sys.argv[3])

    for key in train_data:
        while should_rotate(train_data[key]):
            train_data[key] = np.reshape(train_data[key], (20, 20))
            train_data[key] = np.rot90(train_data[key], 1)
            train_data[key] = train_data[key].ravel()

    for key in test_data:
        while should_rotate(test_data[key]):
            test_data[key] = np.reshape(test_data[key], (20, 20))
            test_data[key] = np.rot90(test_data[key], 1)
            test_data[key] = test_data[key].ravel()

    normalize_list(train_data)
    normalize_list(test_data)

    learning_rate = 0.1
    size_of_data = 400

    happy = Perceptron(learning_rate, size_of_data)
    sad = Perceptron(learning_rate, size_of_data)
    mischievous = Perceptron(learning_rate, size_of_data)
    mad = Perceptron(learning_rate, size_of_data)

    for _ in range(0, 50):
        for key in train_data:
            happy.train(train_data[key], get_desired_val_for_happy(training_answers[key]))
            sad.train(train_data[key], get_desired_val_for_sad(training_answers[key]))
            mischievous.train(train_data[key], get_desired_val_for_mischievous(training_answers[key]))
            mad.train(train_data[key], get_desired_val_for_mad(training_answers[key]))

    keys = test_data.keys()
    num_correct = 0
    correct_answer = 0
    for key in sorted(keys, key=sort_by_image_num):

        num_happy = happy.calculate_output(test_data[key])
        num_sad = sad.calculate_output(test_data[key])
        num_mis = mischievous.calculate_output(test_data[key])
        num_mad = mad.calculate_output(test_data[key])

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
