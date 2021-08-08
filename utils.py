import os
import glob
import numpy as np
from PIL import Image

import numpy as np
from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense

train_directory_path = "dataset/train/"
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_training_dataset():
    all_images = None
    all_labels = None
    for i, folder in enumerate(os.listdir(train_directory_path)):
        filelist = glob.glob(train_directory_path + folder + "/*.jpg")
        current_class_index = class_names.index(folder)
        current_images = np.array([np.array(Image.open(file_name)) for file_name in filelist])
        if(i == 0):
            all_images = current_images
            all_labels = np.full(
                (len(os.listdir(train_directory_path + folder)), 1), current_class_index, dtype=np.dtype(np.int32))
        else:
            all_images = np.concatenate((all_images, current_images))
            all_labels = np.concatenate((all_labels, np.full(
                (len(os.listdir(train_directory_path + folder)), 1), current_class_index, dtype=np.dtype(np.int32))))
    return all_images, all_labels


def load_dataset(directory_path):
    all_images = None
    all_labels = None
    for i, folder in enumerate(os.listdir(directory_path)):
        filelist = glob.glob(directory_path + folder + "/*.jpg")
        current_class_index = class_names.index(folder)
        current_images = np.array([np.array(Image.open(file_name)) for file_name in filelist])
        if(i == 0):
            all_images = current_images
            all_labels = np.full(
                (len(os.listdir(directory_path + folder)), 1), current_class_index, dtype=np.dtype(np.int32))
        else:
            all_images = np.concatenate((all_images, current_images))
            all_labels = np.concatenate((all_labels, np.full(
                (len(os.listdir(directory_path + folder)), 1), current_class_index, dtype=np.dtype(np.int32))))
    return all_images, all_labels

def convert_to_one_hot(array, num_classes):
    result_array = np.eye(num_classes)[array.reshape(-1)].T
    return result_array