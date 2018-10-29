'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import numpy as np

class RandomOrientation90(object):
    def __call__(self, img):
        degrees = 90*np.random.randint(0,4)
        img.rotate(degrees)
        return img

