import numpy as np
from .recommenderMatrix import RecommenderMatrix

class OneClassMatrix(RecommenderMatrix):
    def __init__(self, ratingMatrix, positiveThreshold):
        '''
        Generates a matrix of 1,0
        where:
        1 => This is a positive item
        0 => Don't know anything about this data, might be negative, might be unseen
        '''
        ratingMatrix = ratingMatrix.copy()
        # Note: Don't have to round the reconstruction matrix since you are setting to binary from the threshold below itself
        super().__init__(ratingMatrix)

        # Convert to one class
        self.oneClassMatrix= np.ones(self.ratingMatrix.shape)
        self.oneClassMatrix[np.where(self.ratingMatrix == 0)] = 0.0
        self.oneClassMatrix[np.where(self.ratingMatrix < positiveThreshold)] = 0.0

    # Override
    def applyMask(self, mask):
        super().applyMask(mask) # Checks for mask shape
        self.oneClassMatrix *= mask

    def getOneClassMatrix(self):
        return self.oneClassMatrix.copy()
