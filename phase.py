"""
@author: onion-nikolay
"""

import numpy as np


PHASE_SHAPES = ['random', 'quadratic']


def phase_surface(image_size, phase_type, module=0.5):
    """\n    Returns surface of chosen type.

    Parameters
    ----------
    images_size : tuple of int
        Shape of surface array.
    phase_type : str
        Should be in PHASE_SHAPES.
    module : float, default=0.5
        Define maximum of surface.

    Returns
    -------
    phase_surface : ndarray
    """
    if phase_type == 'random':
        return module * 2*np.pi*np.random.rand(image_size[0], image_size[1])

    elif phase_type == 'quadratic':

        quadratic_phase = np.zeros((image_size[0], image_size[1]))
        for x in range(image_size[0]):
            for y in range(image_size[1]):
                r = ((x-image_size[0]/2)**2+(y-image_size[1]/2)**2)**0.5
                quadratic_phase[x, y] = r**2 * 32*module*np.pi / (
                        image_size[0]*image_size[1])
        return quadratic_phase

    else:
        return np.zeros(image_size)
