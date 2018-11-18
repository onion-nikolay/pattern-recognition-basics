"""
@author: onion-nikolay
"""


from numpy.fft import fft2, ifft2, fftshift, ifftshift


def fft(input_image):
    return fftshift(fft2(ifftshift(input_image)))


def ifft(input_image):
    return fftshift(ifft2(ifftshift(input_image)))
