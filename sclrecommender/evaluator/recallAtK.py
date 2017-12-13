from .rankingEvaluator import RankingEvaluator
from .positiveNegativeEvaluator import PositiveNegativeEvaluator

import numpy as np

class RecallAtK(RankingEvaluator,PositiveNegativeEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, positiveThreshold, k):
        # Run both parent's constructor
        RankingEvaluator.__init__(self, ratingMatrix, rankingMatrix)
        PositiveNegativeEvaluator.__init__(self, ratingMatrix, rankingMatrix.copy(), positiveThreshold)
        self.k = k
        self.recallAtKPerUser = np.zeros(ratingMatrix.shape[0])
        self.meanRecallAtK = 0.0 # meanRecallAcrossAllUsers

    def getRecallAtKPerUser(self):
        if self.recallAtKPerUser is None:
            raise Exception("Must call evaluate() first")
        return self.recallAtKPerUser

    def evaluate(self):
        # Note: The order of matrices in predictedPositiveNegative is matters!
        # Because it came from the ranking matrix
        # Whereas in the usual f1ScoreEvaluator it doesn't. 
        # Because it came from the reconstruction matrix

        eraseNegatives = self.predictedPositiveNegative.copy()
        eraseNegatives[np.where(self.predictedPositiveNegative != 1.0)] = 0.0

        # True positive should use original matrix order
        truePositivePerUser = np.sum(eraseNegatives[:, :self.k], axis = 1)

        sortedPredictedPositiveNegativeMatrix = self.predictedPositiveNegative.copy()
        # Sort by users
        sortedPredictedPositiveNegativeMatrix.sort(axis=1)
        # Sort descending, from large to small
        sortedPredictedPositiveNegativeMatrix = sortedPredictedPositiveNegativeMatrix[:, ::-1]
        # Set all -1 to 0 since don't need it for calculation of conditionPositive
        sortedPredictedPositiveNegativeMatrix[np.where(sortedPredictedPositiveNegativeMatrix == -1)] = 0.0

        # Get the total number of non-zeros (predicted values) for first K values for each user
        # Out of all predictions (!= 0.0 values), how many were actually ones
        # Condition positive uses sorted matrix to maximize number of true values
        conditionPositivePerUser = np.sum(sortedPredictedPositiveNegativeMatrix[:, :self.k], axis = 1)

        self.recallAtKPerUser = np.divide(truePositivePerUser, conditionPositivePerUser, out=np.zeros_like(truePositivePerUser), where=conditionPositivePerUser!=0.0)
        self.meanRecallAtK = np.mean(self.recallAtKPerUser)
        return self.meanRecallAtK
