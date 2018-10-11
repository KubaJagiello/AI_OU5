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

#dict = parser.create_dictionary_for_images("corrupt_data.txt")
#print(dict["Image2"])

#dict = parser.create_dictionary_for_labels("correct_answers.txt")
#print(dict)

import sys
import file_parser as parser
import numpy as np
import random

def get_perceptron_output(weights, input_array):
    sum = 0
    for i in range(0, 400):
        sum += weights[i] * input_array[i]
    return sum + 1

# 400 numbers between 1-1000
def generate_random_weights():
    return np.sign(np.array(random.sample(range(-1000, 1000), 400)))

def activation_function(value):
    return 

def sigMOD(plox):
    return np.sign(plox)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Invalid number of arguments")
        exit()

    train_data = sys.argv[1]
    train_data_labels = sys.argv[2]
    test_data = sys.argv[3]

    dict = parser.create_dictionary_for_images("corrupt_data.txt")

    weights = generate_random_weights()
    print("DICTIONARY for Image1",dict['Image1'])

    result = get_perceptron_output(weights, dict['Image1'])
    print(result)
    #perceptron_output = get_perceptron_output(weights, dict['Image1'])
    print(sigMOD(result))
   # print(perceptron_output)