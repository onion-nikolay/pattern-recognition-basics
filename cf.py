"""
@author: onion-nikolay
"""


import numpy as np
import cv2 as cv
from fft import fft, ifft
from helpers import flattenList


TEST = {'mach'}
LINEAR_FILTERS = {'mach', 'minace'}


def corr(img, flt):
    """\n    Calculate correlation of input image and filter. Fourier image of
    filter is used. Be careful!
    Parameters
    ----------
    img : ndarray
        Input image
    flt : ndarray
        Input filter Fourier image.

    Returns
    -------
    corr : ndarray
    """
    return ifft(fft(img)*np.conj(flt))


def corr_output(image, corr_filter, out='value'):
    """\n    Calculate correlation and returns different parameters.
    Parameters
    ----------
    image : ndarray
    corr_filter : ndarray
        See cf.corr for more information
    out : str
        If 'value', returns correlation peak height. If 'coord', returns
        CP height and coordinates, If full, returns CP height, coordinates and
        CP image. Else raises error.

    Returns
    -------
    output : number or list
    """
    corr_output = np.abs(corr(image, corr_filter))
    peak = np.max(corr_output)
    coord = np.unravel_index(corr_output.argmax(), corr_output.shape)
    if out is 'value':
        return peak
    elif out is 'coord':
        return [peak, coord]
    elif out is 'full':
        return [peak, coord, corr_output]
    else:
        raise("key {} not found.".format(out))


def corr_output_holo(image, corr_filter, out='value'):
    """\n    Like cf.corr_output, calculates correlation and returns
    different parameters. But CP coordinates is limited by one angle.
    Parameters
    ----------
    image : ndarray
    corr_filter : ndarray
        See cf.corr for more information
    out : str
        If 'value', returns correlation peak height. If 'coord', returns
        CP height and coordinates, If full, returns CP height, coordinates and
        CP image. Else raises error.

    Returns
    -------
    output : number or list
    """
    corr_output_raw = np.abs(corr(image, corr_filter))
    corr_output = corr_output_raw
    _size = np.shape(corr_output)[0]/4
    corr_output[3*_size/2:5*_size/2, 3*_size/2:5*_size/2] = 0
    peak = np.max(corr_output[_size*2:, _size*2:])
    coord = np.unravel_index(corr_output.argmax(), corr_output.shape)
    if out is 'value':
        return peak
    elif out is 'coord':
        return [peak, coord]
    elif out is 'full':
        return [peak, coord, corr_output_raw]
    else:
        raise("key {} not found.".format(out))


def synthesize(train_objects, train_object_labels, **kwargs):
    """\n    Synthesize CF.
    Parameters
    ----------
    train_objects : list or lists of ndarray
    train_object_labels : list of int
    **kwargs
        Use for filter_type and other parameters of filters.

    Returns
    -------
    corr_filter : ndarray
    """
    try:
        filter_type = kwargs['filter_type']
    except KeyError:
        raise("filter type not found.")
    true_objects = []
    false_objects = []
    for obj, label in zip(train_objects, train_object_labels):
        if label == 1:
            true_objects.append(obj)
        else:
            false_objects.append(obj)
    true_objects = flattenList(true_objects)
    false_objects = flattenList(false_objects)
    try:
        corr_filter = globals()[filter_type](true_objects, false_objects,
                                             **kwargs)
        return corr_filter
    except ImportError:
        print("Error! Filter {} not found! Return None.".format(filter_type))
        return None


def predict(corr_filter, data_to_predict, threshold, return_class=True,
            is_holo=False):
    """\n    Calculate correlation peaks or classes for input images.
    Parameters:
    corr_filter : ndarray
    data_to_predict : list of ndarray
        Input images.
    threshold : float
    return_class : bool, default=True
        If True, returns class (1 or 0). Else returns CP heights.
    is_holo : bool, default=False
        If True, cf.corr_output_holo uses for calculations, else
        cf.corr_output.

    Returns
    -------
    peaks : list of float or int
    """
    peaks = []
    for image in data_to_predict:
        if is_holo:
            peak = corr_output_holo(image, corr_filter)
        else:
            peak = corr_output(image, corr_filter)
        peaks.append(peak)
    if return_class:
        return np.uint8(peaks > threshold)
    else:
        return peaks


def mach(true_objects, *false_objects, **kwargs):
    """\n    MACH: h = [alpha*D_y + (1-alpha^2)^(1/2)S^0_x]^(-1)*m
    UOTSDF: MACH(alpha=1)
    Default: alpha=0.5
    """

    try:
        alpha = kwargs['alpha']
    except KeyError:
        alpha = 0.5

    n = len(true_objects)
    size = np.shape(true_objects[0])[0]
    length = np.size(true_objects[0])
    x_fft = __x2fftx(true_objects)
    m = np.mean(x_fft, 1)
    s = np.zeros(length, dtype=complex)
    for i in range(n):
        s += (x_fft[:, i] - m) * np.conj(x_fft[:, i] - m)
    s = s / n
    d = np.mean(x_fft*np.conj(x_fft), 1)
    h_fft = np.zeros(length, dtype=complex)
    for i in range(length):
        h_fft[i] = ((alpha*s[i] + ((1-alpha**2)**0.5)*d[i])**(-1)) * m[i]
    h_fft = h_fft / (n*length)
    h_fft = np.reshape(h_fft, (size, size), order='C')
    return h_fft


def minace(true_objects, *false_objects, **kwargs):
    """\n    Doc in progress..."""
    from numpy.linalg import inv
    n = len(true_objects)
    size = np.shape(true_objects[0])[0]
    length = np.size(true_objects[0])
    x_fft = __x2fftx(true_objects)
    d = x_fft*np.conj(x_fft)

    try:
        alpha = kwargs['alpha']
    except KeyError:
        alpha = 0
    try:
        p = alpha*kwargs['noise_sp']
    except KeyError:
        p = alpha*np.ones((length))
    try:
        beta = kwargs['beta']
    except KeyError:
        beta = 1

    p = np.reshape(p, (length, 1))
    d = np.concatenate((beta*d, ((1-beta**2)**0.5)*p), axis=1)
    t = np.max(d, 1)**-1
    c = np.ones(n)
    h_fft = __dot(t, x_fft)
    h_fft = np.dot(np.conj(x_fft.T), h_fft)
    h_fft = np.dot(inv(h_fft), c)
    h_fft = np.dot(x_fft, h_fft)
    h_fft = t * h_fft / (n*length)
    h_fft = np.reshape(h_fft, (size, size), order='C')
    return h_fft


def __x2fftx(x):
    n = len(x)
    x_size = np.shape(x[0])[0]
    x_length = np.size(x[0])
    x_fft = np.zeros((x_size, x_size, n), dtype=complex)
    for count in range(n):
        x_fft[:, :, count] = fft(x[count])
    x_fft = np.reshape(x_fft, (x_length, n), order='C')
    return x_fft


def __dot(vec, arr):
    [length, n] = np.shape(arr)
    res = np.zeros((length, n), dtype=complex)
    for i in range(length):
        for j in range(n):
            res[i, j] = vec[i] * arr[i, j]
    return res


def synthesizeHolo(input_image):
    """\n    Synthesize Fourier hologram of input image.
    """
    image_shape = np.shape(input_image)
    holo_image = np.zeros((image_shape[0]*4, image_shape[1]*4),
                          dtype=complex)
    holo_image[image_shape[0]/2:3*image_shape[0]/2,
               image_shape[1]/2:3*image_shape[1]/2] = input_image
    holo_image = np.real(fft(holo_image))
    holo_image = holo_image - np.min(holo_image)
    holo_image = holo_image / np.max(holo_image)
    return holo_image


def restoreHolo(holo_image, show_image=True):
    """\n    Restore Fourier hologram of input image.
    Parameters
    ----------
    holo_image : ndarray
    show_image : bool, default=True
        If True, shows image in new figure.

    Returns
    -------
    restored_holo_image : ndarray
    """
    restored_holo_image = np.abs(ifft(holo_image))
    image_shape = np.shape(restored_holo_image)
    restored_holo_image[image_shape[0]/2, image_shape[1]/2] = 0
    restored_holo_image -= np.min(restored_holo_image)
    restored_holo_image /= np.max(restored_holo_image)
    if show_image:
        cv.imshow("Restored Holo", restored_holo_image)
    return restored_holo_image
