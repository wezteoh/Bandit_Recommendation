from .banditEvaluator import BanditEvaluator

import numpy as np

class RegretOptimalEvaluator(BanditEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, discountFactor):
        BanditEvaluator.__init__(self, ratingMatrix, rankingMatrix, discountFactor)
        self.bestOptimalRegret = None
        self.optimalRegret = None

        self.powers = np.array([range(self.rankingMatrix.shape[1]) for _ in range(self.rankingMatrix.shape[0])])

    def calculateRegret(self, matrix):
        regretMatrix = np.ones(matrix.shape)
        # Make discount factor the initial values
        regretMatrix *= self.discountFactor
        discounted = np.power(regretMatrix, self.powers)
        optimalRegretMatrix = matrix * discounted
        result = np.sum(optimalRegretMatrix, axis=1)
        return result

    def evaluate(self):
        sortedMatrix = self.rankingMatrix.copy()
        sortedMatrix.sort(axis = 1)
        sortedMatrix = sortedMatrix[:, ::-1]
        self.bestOptimalRegret = self.calculateRegret(sortedMatrix)
        self.optimalRegret = self.calculateRegret(self.rankingMatrix)
        # Guaranteed to be >= 0 since discounting by 1.0
        difference = np.sum(self.bestOptimalRegret - self.optimalRegret)
        return difference
