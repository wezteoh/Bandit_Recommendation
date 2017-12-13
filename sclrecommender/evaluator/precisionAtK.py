from .rankingEvaluator import RankingEvaluator
from .positiveNegativeEvaluator import PositiveNegativeEvaluator

import numpy as np

class PrecisionAtK(RankingEvaluator,PositiveNegativeEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, positiveThreshold, k):
        # Run both parent's constructor
        RankingEvaluator.__init__(self, ratingMatrix, rankingMatrix)
        PositiveNegativeEvaluator.__init__(self, ratingMatrix, rankingMatrix.copy(), positiveThreshold)
        self.k = k
        self.precisionAtKPerUser = None
        self.meanPrecisionAtK = 0.0 # meanPrecisionAtK for all users

    def setK(self, k):
        self.k = k

    def getPrecisionAtKPerUser(self):
        if self.precisionAtKPerUser is None:
            raise Exception("Must call evaluate() first")
        return self.precisionAtKPerUser

    def evaluate(self):
        # Note: The order of matrices in predictedPositiveNegative is matters!
        # Because it came from the ranking matrix
        # Whereas in the usual f1ScoreEvaluator it doesn't. 
        # Because it came from the reconstruction matrix

        # For the bandit ranking matrix, 
        # Everything you listed out IS your predictions of positives
        # The labels in them are the actual true positive and negatives

        # Extract out only for first k values
        # Get the number of cells equal to 1.0 for first K values for each user
        eraseNegatives = self.predictedPositiveNegative.copy()
        eraseNegatives[np.where(self.predictedPositiveNegative != 1.0)] = 0.0

        truePositivePerUser = np.sum(eraseNegatives[:, :self.k], axis = 1)

        modNegatives = self.predictedPositiveNegative.copy()
        modNegatives[np.where(self.predictedPositiveNegative == -1.0)] = 1.0

        predictionPositivePerUser = np.sum(modNegatives[:, :self.k], axis = 1)

        # Perform division where conditionPositivePerUser isn't zero
        self.precisionAtKPerUser = np.divide(truePositivePerUser, predictionPositivePerUser, out=np.zeros_like(truePositivePerUser), where=predictionPositivePerUser !=0.0)

        # Get the total number of non-zeros (predicted values) for first K values for each user
        # Out of all predictions (!= 0.0 values), how many were actually ones
        self.meanPrecisionAtK = np.mean(self.precisionAtKPerUser)

        return self.meanPrecisionAtK
