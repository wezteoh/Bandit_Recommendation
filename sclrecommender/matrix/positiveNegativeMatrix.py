import numpy as np
from .recommenderMatrix import RecommenderMatrix

class PositiveNegativeMatrix(RecommenderMatrix):
    def __init__(self, ratingMatrix, positiveThreshold):
        '''
        Generates a matrix of 1, -1, 0
        where:
        0 => Don't know anything about this data
        1 => This is a postive item
        -1 => This is a negative item
        '''
        ratingMatrix = ratingMatrix.copy()
        # Note: Don't have to round the reconstruction matrix since you are setting to binary from the threshold below itself
        super().__init__(ratingMatrix)

        # Convert to positive negative
        self.positiveNegativeMatrix = np.ones(self.ratingMatrix.shape)
        self.positiveNegativeMatrix[np.where(self.ratingMatrix < positiveThreshold)] = -1.0
        self.positiveNegativeMatrix[np.where(self.ratingMatrix == 0)] = 0.0

    # Override
    def applyMask(self, mask):
        super().applyMask(mask) # Checks for mask shape
        self.positiveNegativeMatrix *= mask

    def getPositiveNegativeMatrix(self):
        return self.positiveNegativeMatrix
