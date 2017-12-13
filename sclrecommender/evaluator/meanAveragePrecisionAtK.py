from .rankingEvaluator import RankingEvaluator
from .positiveNegativeEvaluator import PositiveNegativeEvaluator
from .precisionAtK import PrecisionAtK

import numpy as np

class MeanAveragePrecisionAtK(RankingEvaluator,PositiveNegativeEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, positiveThreshold, k):
        # Run both parent's constructor
        RankingEvaluator.__init__(self, ratingMatrix, rankingMatrix)
        PositiveNegativeEvaluator.__init__(self, ratingMatrix, rankingMatrix.copy(), positiveThreshold)
        self.k = k
        self.averagePrecisionAtKPerUser = np.zeros(self.ratingMatrix.shape[0])
        self.meanAveragePrecisionAtK = 0.0 # meanPrecisionAtK for all users

    def evaluate(self):
        prec = PrecisionAtK(self.ratingMatrix, self.rankingMatrix, self.positiveThreshold, 1)
        # Initialize to zeros for each user
        totalPrecisionToKPerUser = np.zeros(self.ratingMatrix.shape[0])
        predictedPositiveZero = self.predictedPositiveNegative.copy()
        predictedPositiveZero[np.where(self.predictedPositiveNegative == -1)] = 0.0
        for currK in range(self.k):
            prec.setK(currK + 1)
            prec.evaluate()
            precisionAtKPerUser = prec.getPrecisionAtKPerUser()
            totalPrecisionToKPerUser += precisionAtKPerUser * predictedPositiveZero[:, currK]
        numToDividePerUser = np.sum(predictedPositiveZero[:, :self.k], axis=1)
        self.averagePrecisionAtKPerUser = np.divide(totalPrecisionToKPerUser, numToDividePerUser, out=np.zeros_like(totalPrecisionToKPerUser), where= numToDividePerUser != 0.0)
        sumAveragePrecision = np.sum(self.averagePrecisionAtKPerUser)
        self.meanAveragePrecisionAtK = np.divide(sumAveragePrecision, self.ratingMatrix.shape[0], out=np.zeros_like(sumAveragePrecision), where= self.ratingMatrix.shape[0]!=0.0)
        return self.meanAveragePrecisionAtK
