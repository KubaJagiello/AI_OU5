import numpy as np

# dict = {'Image1' : []}
def create_dictionary_for_images(filename):
    image_dictionary = {}
    dict_index = 0
    dict_key = ""

    with open(filename, "r") as file:
        for line in file:

            if line.isspace() or line.find("#") != -1:
                continue

            line = line.replace("\n", "")
            line = line.split()
            if len(line) == 1:
                dict_key = line[0]
                image_dictionary[dict_key] = np.array([])
                dict_index += 1
                continue
            line = np.array(list(map(int, line)))
            image_dictionary[dict_key] = np.append(image_dictionary[dict_key], line)
    return image_dictionary


# dict = {'Image1' : '1}
def create_dictionary_for_labels(filename):
    label_dictionary = {}

    with open(filename) as file:
        for line in file:

            if line.isspace() or line.find("#") != -1:
                continue

            line = line.replace("\n", "")
            line = line.split()
            label_dictionary[line[0]] = int(line[1])

    return label_dictionary