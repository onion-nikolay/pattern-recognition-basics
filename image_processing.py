"""
@author: onion-nikolay
"""
import numpy as np
import skimage.filters as skf
import cv2 as cv
from inspect import getargspec
from helpers import calculateWindow, chooseArgs

BIN_METHODS_GLOBAL = ['custom', 'mean', 'median', 'otsu', 'triangle', 'li',
                      'isodata', 'yen']
BIN_METHODS_ADAPTIVE = ['ad_mean', 'ad_gaussian', 'niblack', 'sauvola',
                        'bradley']
BIN_METHODS = BIN_METHODS_ADAPTIVE + BIN_METHODS_GLOBAL


def binarize(image, bin_method, **kwargs):
    """\n    Returns binarized image.

    Parameters
    ----------
    image : ndarray, image to be binarized
    bin_method : str, name of thresholding method. available methods:
    ------------
        Globals:
        custom : constant threshold (default threshold = 0.5)
               args: 'threshold'
        mean : threshold = np.mean(image)
               args: None
        median : threshold = np.median(image)
               args: None
        otsu : threshold = skimage.filters.threshold_otsu(image, **kwargs)
               args: 'nbins'
        triangle : threshold = skimage.filters.threshold_triangle(image,
                                                                  **kwargs)
               args: 'nbins'
        li : threshold = skimage.filters.threshold_li(image)
               args: None
        isodata : threshold = skimage.filters.threshold_isodata(image,
                                                                **kwargs)
               args: 'nbins', 'return_all'
        yen: threshold = skimage.filters.threshold_yen(image, **kwargs)
               args: 'nbins'
    -----------
        Locals:
        ad_mean : threshold for each pixel is mean of window. For more
        imformation see:
            https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
               args: 'window_size', 'c'
        ad_gaussian : in work...
        niblack : threshold = skimage.filters.threshold_niblack(image,
                                                                **kwargs)
               args: 'window_size', 'k'
        sauvola : threshold = skimage.filters.threshold_sauvola(image,
                                                                **kwargs)
               args: 'window_size', 'k', 'r'
        bradley : see https://github.com/rmtheis/bradley-adaptive-thresholding
        Will be removed by skimage.filters.threshold_bradley in future.
               args: 't'

    Returns
    -------
    binary_image : ndarray (dtype=np.uint8)
    """
# Should be improved! I've done it, because I heven't found another way.
# Current problem is thresholding works good only with uint8 images and
# float images with pixels, distributed from 0 to 1.
    def __threshold_custom(image, threshold=0.5):
        if type(image[0, 0]) == np.uint8:
            threshold *= 255
        return threshold

# Soon it will be removed by function from skf.
    def __threshold_bradley(image, t=0.15):
        [width, height] = np.shape(image)
        s1 = width//8
        s2 = s1//2
        integral_image = np.zeros((width*height,))
        threshold_mask = np.zeros([width, height])

        for i in range(width):
            _sum = 0
            for j in range(height):
                index = j * width + i
                _sum += image[i, j]
                integral_image[index] = integral_image[index-1] + _sum

        for i in range(width):
            for j in range(height):
                index = j * width + i
                x1 = i - s2
                x2 = i + s2
                y1 = j - s2
                y2 = j + s2
                x1 = 0 if x1 < 0 else x1
                x2 = width-1 if x2 >= width else x2
                y1 = 0 if y1 < 0 else y1
                y2 = height-1 if y2 >= height else y2
                count = (x2-x1)*(y2-y1)
                _sum = integral_image[y2*width+x2] - \
                    integral_image[y1*width+x2] - \
                    integral_image[y2*width+x1] + \
                    integral_image[y1*width+x1]
                threshold_mask[i, j] = _sum*(1.0-t)//count
        return threshold_mask

# There is an implementation in cv2, but it returns image, not mask, so
# the slower one is used now.
    def __threshold_admean(image, window_size=11, c=0):
        shp = np.shape(image)
        threshold_mask = np.zeros(shp, dtype=np.uint8)
        for x in range(shp[0]):
            for y in range(shp[1]):
                windowed_image = calculateWindow(image, [x, y], window_size)
                threshold_mask[x, y] = np.mean(windowed_image) - c
        return threshold_mask

    __BIN_METHODS_FUNC = {'mean': np.mean,
                          'median': np.median,
                          'otsu': skf.threshold_otsu,
                          'triangle': skf.threshold_triangle,
                          'li': skf.threshold_li,
                          'isodata': skf.threshold_isodata,
                          'yen': skf.threshold_yen,
                          'niblack': skf.threshold_niblack,
                          'sauvola': skf.threshold_sauvola,
                          'bradley': __threshold_bradley,
                          'ad_mean': __threshold_admean,
                          'ad_gaussian': __threshold_admean,
                          'custom': __threshold_custom}

    if not(bin_method in BIN_METHODS):
        raise ValueError("method '{}' is not found.".format(bin_method))
        return np.uint8(image)
    else:
        bin_func = __BIN_METHODS_FUNC[bin_method]
        try:
            bin_func_args = dict(zip(getargspec(bin_func)[0][1:],
                                     getargspec(bin_func)[3]))
            threshold = bin_func(image, **chooseArgs(bin_func_args, kwargs))
        except TypeError:
            threshold = bin_func(image)
        return np.uint8(image > threshold)


def quantize(input_image, color_depth=8):
    """\n    Returns image with constant number of color depth.

    Parameters
    ----------
    imput_image : ndarray (with positive only elements)
    color_depth : int, output color depth (from 2 to 8 bytes, default = 8)

    Returns
    -------
    output_image : ndarray (dtype=np.uint8)
    """
    input_image = input_image.astype(float)
    output_image = input_image/input_image.max()*(2**color_depth-1)
    return output_image.astype(np.uint8)


# Remove dummy_sizing
def cfPreprocessing(input_images, field_color=0, **kwargs):
    """\n    Returns image, preprocessed for synthes of correlation filter or
    for correlation pattern recognition. The main idea is to fimd max size of
    all images and CF and place all images on equal square fields with sizes
    2^n.

    Parameters
    ----------
    input_images : list of ndarray
    field_color : int or 'mean', default=0
        Default field color is black. It can be a number from black to white,
        or 'mean' of input_image.
    **kwargs
        We use it to pass CF to the function if preprocessing is used for CPR
        and size of CF can be bigger then max size of test images.

    Returns
    -------
    output_images : list of ndarray
    """
    dummy_sizing = False
    try:
        corr_filter = kwargs['corr_filter']
        max_size = np.max(np.shape(corr_filter))
    except KeyError:
        max_size = 0
    for set_of_images in input_images:
        for image in set_of_images:
            size = np.max(np.shape(image))
            if size > max_size:
                max_size = size
    sizes = [2**num for num in range(17)]
    max_size = min(num for num in sizes if num >= max_size)

# It used for some tests, don't remove it now, until it can be useful or
# some better way of testing will be used.
    if dummy_sizing:
        max_size *= 2

    output_images = input_images
    for index, set_of_images in enumerate(input_images):
        output_images[index] = square(set_of_images, max_size, field_color)

    return output_images


def square(input_images, field_size, field_color=0, centered=False):
    """\n    Returns images placed on square fields.

    Parameters
    ----------
    input_image : list of ndarray
    field_size : int
        input_image is placed on field (field_size, field_size).
        Should be 2**n.
    field_color : int or 'mean', default=0
        Default field color is black. It can be a number from black to white,
        or 'mean' of input_image.
    centered : bool, default=False
        If True, place image to the center of field.

    Returns
    -------
    output_images : list of ndarray
    """
    output_images = []
    for input_image in input_images:

        if field_color is 'mean':
            field_color = np.mean(input_image)
        output_image = np.ones((field_size, field_size))*field_color
        [size1, size2] = np.shape(input_image)
        if field_size < max([size1, size2]):
            raise ValueError("field size is less then input image size.")
        if centered:
            x1 = int(field_size//2-size1//2)
            x2 = int(field_size//2+size1//2) - (size1 % 2)
            y1 = int(field_size//2-size2//2)
            y2 = int(field_size//2+size2//2) - (size2 % 2)
            output_image[x1:x2, y1:y2] = input_image
        else:
            output_image[:size1, :size2] = input_image
        output_images.append(cv.equalizeHist(output_image.astype(np.uint8)))
    return output_images


def cfProcessing(raw_image, processing_method, **kwargs):
    """\n    This function turns raw_image of CF into image can be used for
    SLM output or simulation. Now it includes quantization, binarizartion and
    phase addition.
    In progress: phase output, phase-on-amplitude, noises.

    Parameters
    ----------
    raw_image : ndarray
    processing_method : int, str or list
        If int, should be range from 2 to 8. Use for basic amplitude images.
        If str, shoul be in BIN_METHODS.
        If list, first element should be int or str like above, second one
        should be in phase.PHASE_SHAPES.
    **kwargs
        Can be used for arguments of thresholding or phase surface.

    Returns
    -------
    processed_image : ndarray
    """
    from phase import phase_surface
    if type(processing_method) is int:
        processed_image = quantize(raw_image, processing_method)
    elif type(processing_method) is str:
        preproc_image = quantize(raw_image, 8)
        processed_image = binarize(preproc_image, processing_method, **kwargs)
    elif type(processing_method) is list:
        try:
            processing_method_0 = int(processing_method[0])
        except ValueError:
            processing_method_0 = processing_method[0]
        if type(processing_method_0) is int:
            processed_image = quantize(raw_image, processing_method_0)
        else:
            processed_image = quantize(raw_image, 8)
            processed_image = binarize(processed_image, processing_method_0,
                                       **kwargs)
        phase = phase_surface(np.shape(processed_image), processing_method[1])
        processed_image = processed_image * np.exp(1j*phase)
    else:
        print("Error! Processed method not recognised.")
        processed_image = raw_image
    return processed_image
