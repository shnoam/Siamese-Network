import os
import pickle
import warnings
import tqdm
from PIL import Image
import numpy as np


WIDTH = 105
HEIGHT = 105
CELL = 1

def open_image(path, width, height):
    image = Image.open(path)
    image = image.resize((width, height))
    data = np.asarray(image)
    data = np.array(data, dtype='float64')
    return data


def image_to_np_array(name, image_number, data_path, width, height, cells, to_train=True):
    """
    Given a person, image number and datapath, returns a numpy array which represents the image.
    predict - whether this function is called during training or testing. If called when training, we must reshape
    the images since the given dataset is not in the correct dimensions.
    """
    image_number = '0000' + image_number
    image_number = image_number[-4:]
    image_path = os.path.join(f"data/lfw2/lfw2/{name}/{name}_{image_number}.jpg")

    image_array = open_image(image_path, width, height)
    if to_train:
        image_array = image_array.reshape(width, height, cells)
    return image_array


def get_mismatched_images(split_line, data_path):
    first_name, first_image_num, second_name, second_image_num = split_line
    image1 = image_to_np_array(first_name, first_image_num, data_path, width=WIDTH, height=HEIGHT, cells=CELL)
    image2 = image_to_np_array(second_name, second_image_num, data_path, width=WIDTH, height=HEIGHT, cells=CELL)
    return image1, image2


def get_matched_images(split_line, data_path):
    name, first_image_num, second_image_num = split_line
    image1 = image_to_np_array(name, first_image_num, data_path, width=WIDTH, height=HEIGHT, cells=CELL)
    image2 = image_to_np_array(name, second_image_num, data_path, width=WIDTH, height=HEIGHT, cells=CELL)
    return image1, image2


def load_images(set_name,data_path,output_path):
    """
    Writes into the given output_path the images from the data_path.
    dataset_type = train or test
    """
    file_path = '{}/{}.txt'.format(data_path, set_name)
    x_1, x_2, y_true = [], [], []
    names = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in tqdm.tqdm(lines):
        split_line = line.split()
        line_length = len(split_line)
        y_matching_value = 0
        pair_status = 'match' if line_length == 3 else ('mismatch' if line_length == 4 else 'Unknown')
        if pair_status == 'Unknown':
            print(f'there is an invalid line: {split_line}')
            continue
        elif pair_status == 'match':
            names.append(split_line)
            img1, img2 = get_matched_images(split_line, data_path)
            x_1.append(img1)
            x_2.append(img2)
            y_matching_value = 1
        else: # pair_status == 'mismatch':
            names.append(split_line)
            img1, img2 = get_mismatched_images(split_line, data_path)
            x_1.append(img1)
            x_2.append(img2)

        y_true.append(y_matching_value)

    with open(output_path, 'wb') as f:
        pickle.dump([x_1, x_2, y_true, names], f)











