import numpy as np
import random

class MaskGenerator(object):
    '''
    This mask generator just returns everything
    '''
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        self.mask = np.ones(ratingMatrix.shape)

    def getMaskCopy(self):
        return self.mask.copy()
