"""
@author: onion-nikolay
"""


import numpy as np
import os
import cv2 as cv
from scipy.stats import norm
from os.path import join as pjoin


IMAGE_FORMATS = ['jpg', 'png', 'tif', 'bmp']


def array2list(input_array):
    """\n    Obsolete. Replaced by tolist()"""
    return [element for element in input_array]


def chooseArgs(function_args, user_args):
    """\n    Choose all args from user_args which function_args includes."""
    return {x: user_args[x] for x in user_args if x in function_args}


def flattenImage(input_array):
    """\n    Reshape image with (size1, size2) into (size1*size2,)"""
    shp = np.size(input_array)
    return np.reshape(input_array, (shp,))


def flattenList(input_list):
    """\n    Turns list of lists into one list."""
    return [item for sublist in input_list for item in sublist]


def listLengths(input_list):
    """\n    Calculate lengths of internal lists."""
    return [len(item) for item in input_list]


def listExpend(input_list_1, input_list_2):
    """\n   Turns ([1, 0], [2, 3]) into [[1, 1], [0, 0, 0]]."""
    output_list = []
    for element_1, element_2 in zip(input_list_1, input_list_2):
        output_list += [element_1]*element_2
    return output_list


def norm_dist(numbers, x):
    """\n    Returns normal distribution, calculated by numbers in range of x.
    """
    m = np.mean(np.array(numbers))
    s = np.std(np.array(numbers))
    if s < 1e-5:
        s = 1e-5
    return np.array(norm.pdf(x, loc=m, scale=s))


def returnImages(input_data):
    """\n    Returns images from input paths."""
    if type(input_data) is list:
        return [returnImages(element) for element in input_data]
    else:
        return cv.imread(input_data, 0)


def returnFiles(input_data, fmt=IMAGE_FORMATS):
    """\n    Returns paths of image files from directory or list of
    directories.
    """
    if type(input_data) is list:
        return [returnFiles(element) for element in input_data]
    else:
        if not os.path.exists(input_data):
            raise OSError("invalid input path!")
        return [pjoin(input_data, name) for name in os.listdir(
                input_data) if os.path.splitext(name)[1][1:] in fmt]


def calculateWindow(image, coord, window_size):
    """\n    Calculate window around pixel with limits of boards."""
    image_shape = np.shape(image)
    sz = (window_size-1)//2
    x_list = np.arange(coord[0]-sz, coord[0]+sz+1, 1)
    y_list = np.arange(coord[1]-sz, coord[1]+sz+1, 1)
    xs = [x for x in x_list if (x >= 0) and (x < image_shape[0])]
    ys = [y for y in y_list if (y >= 0) and (y < image_shape[1])]
    x1 = np.min(xs)
    x2 = np.max(xs)
    y1 = np.min(ys)
    y2 = np.max(ys)
    return image[x1:x2+1, y1:y2+1]
