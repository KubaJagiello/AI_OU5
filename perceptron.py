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
    weights = []
    input_data = []
    bias = 0
    activation_limit = 0
    learning_rate = 0.3

    def __init__(self, learning_rate, size_of_input):
        self.learning_rate = learning_rate
        self.weights = self.generate_random_weights(size_of_input)

    def calculate_sum(self):
        sum_of_output = 0
        for i in range(0, len(self.input_data)):
            sum_of_output += self.weights[i] * self.input_data[i]
        return sum_of_output + self.bias

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


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Invalid number of arguments")
    #     exit()

    train_data = sys.argv[1]
    train_data_labels = sys.argv[2]
    test_data = sys.argv[3]

    image_data_dict = parser.create_dictionary_for_images("training.txt")
    answers = parser.create_dictionary_for_labels("correct_answers.txt")

    learning_rate = 100
    size_of_data = 400

    percept_happy = Perceptron(learning_rate, size_of_data)
    percept_sad = Perceptron(learning_rate, size_of_data)
    percept_mischievous = Perceptron(learning_rate, size_of_data)
    percept_mad = Perceptron(learning_rate, size_of_data)

    for key in image_data_dict:
        percept_happy.train(image_data_dict[key], get_desired_val_for_happy(answers[key]))
        percept_sad.train(image_data_dict[key], get_desired_val_for_sad(answers[key]))
        percept_mischievous.train(image_data_dict[key], get_desired_val_for_mischievous(answers[key]))
        percept_mad.train(image_data_dict[key], get_desired_val_for_mad(answers[key]))

    num_correct = 0

    correct_answer = 0
    for key in image_data_dict:
        if percept_happy.calculate_output(image_data_dict[key]) == 1:
            correct_answer = 1
        elif percept_sad.calculate_output(image_data_dict[key]) == 1:
            correct_answer = 2
        elif percept_mischievous.calculate_output(image_data_dict[key]) == 1:
            correct_answer = 3
        elif percept_mad.calculate_output(image_data_dict[key]) == 1:
            correct_answer = 4

        if correct_answer == answers[key]:
            num_correct = num_correct + 1

    print("percept was ", num_correct, " out of", len(answers), "correct after training.")
