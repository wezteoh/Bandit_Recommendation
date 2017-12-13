from .reconstructionEvaluator import ReconstructionEvaluator
from ..matrix.positiveNegativeMatrix import PositiveNegativeMatrix

import numpy as np

class PositiveNegativeEvaluator(ReconstructionEvaluator):
    def __init__(self, ratingMatrix, reconstructionMatrix, positiveThreshold=3):
        if ratingMatrix.shape != reconstructionMatrix.shape:
            raise ValueError("ratingMatrix and reconstructionMatrix are different shapes!")
        super().__init__(ratingMatrix, reconstructionMatrix)
        self.positiveThreshold = positiveThreshold
        # Get positiveNegative matrices
        self.truthPositiveNegative = PositiveNegativeMatrix(self.ratingMatrix, self.positiveThreshold).getPositiveNegativeMatrix()
        self.predictedPositiveNegative = PositiveNegativeMatrix(self.reconstructionMatrix, self.positiveThreshold).getPositiveNegativeMatrix()

    def evaluate(self):
        # Should be overridden
        raise NotImplementedError
