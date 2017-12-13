import numpy as np
import random
from .maskGenerator import MaskGenerator

class RandomMaskGenerator(MaskGenerator):
    '''
    A mask generator to randomly split to train and test
    '''
    def __init__(self, ratingMatrix, trainSplit, fixedSize=True):
        super().__init__(ratingMatrix)
        self.trainMask = None
        if fixedSize:
            numOne = int(round(trainSplit * self.ratingMatrix.size))
            numZero = self.ratingMatrix.size - numOne
            self.trainMask = np.array([0] * numZero + [1] * numOne)
            # Note: Must random before reshape, otherwise, will just randomize the rows 
            np.random.shuffle(self.trainMask)
            self.trainMask = np.reshape(self.trainMask, self.ratingMatrix.shape)
        else:
            self.trainMask = np.random.binomial(1, trainSplit, ratingMatrix.shape)
        self.testMask = np.ones(self.ratingMatrix.shape) - self.trainMask

    def getMasksCopy(self):
        return self.trainMask.copy(), self.testMask.copy()
