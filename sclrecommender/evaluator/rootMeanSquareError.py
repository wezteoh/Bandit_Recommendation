from .reconstructionEvaluator import ReconstructionEvaluator

import numpy as np

# TODO: Only evaluate on the non-zeroes for rating matrix? 
class RootMeanSquareError(ReconstructionEvaluator):
    def __init__(self, ratingMatrix, reconstructionMatrix):
        super().__init__(ratingMatrix, reconstructionMatrix)

    def evaluate(self):
        return np.sqrt(np.mean((self.reconstructionMatrix - self.ratingMatrix)**2))

