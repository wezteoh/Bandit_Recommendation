from .evaluator import Evaluator

import numpy as np

class ReconstructionEvaluator(Evaluator):
    def __init__(self, ratingMatrix, reconstructionMatrix):
        if ratingMatrix.shape != reconstructionMatrix.shape:
            raise ValueError("ratingMatrix and reconstructionMatrix are different shapes!")
        super().__init__(ratingMatrix)
        self.reconstructionMatrix = reconstructionMatrix

    def evaluate(self):
        '''
        Simply calculates the number of same ratings vs the total number of ratings
        '''
        totalNumRatings = float(self.ratingMatrix.shape[0] * self.ratingMatrix.shape[1])
        roundedMatrix = np.round(self.reconstructionMatrix.copy())
        numEqualRating = np.sum(roundedMatrix == self.ratingMatrix)
        accuracy = numEqualRating/totalNumRatings
        return accuracy


