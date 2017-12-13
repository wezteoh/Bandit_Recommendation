from .banditEvaluator import BanditEvaluator
from .rootMeanSquareError import RootMeanSquareError

import numpy as np

class RegretInstantaneousEvaluator(BanditEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, discountFactor, exploreMask, orderChoices):
        BanditEvaluator.__init__(self, ratingMatrix, rankingMatrix, discountFactor)
        self.orderChoices = orderChoices
        self.exploreMask = exploreMask
        self.instantaneousRegret = None
        self.cumulativeInstantaneousRegret = None # For plotting how the regret increases

    def bestScore(self, userIndex, bestRatingMatrix):
        if userIndex < 0 or userIndex >= bestRatingMatrix.shape[0]:
            raise ValueError("userIndex " + str(userIndex) + " given is not compatible with ratingMatrix shape of " + str(bestRatingMatrix.shape[0]))
        return np.max(bestRatingMatrix[userIndex])
    
    def getCumulativeInstantaneousRegret(self):
        if self.cumulativeInstantaneousRegret is None:
            raise Exception("Must run evaluate first")
        return self.cumulativeInstantaneousRegret

    def evaluate(self, userIndex=None):
        # Note: Don't really need discount factor since will already penalize if don't pick current max earlier
        self.instantaneousRegret = 0.0 
        bestRatingMatrix = self.ratingMatrix.copy()
        bestRatingMatrix[np.where(self.exploreMask != 1)] = 0.0 # Set to zeroes for non-legal choices
        self.cumulativeInstantaneousRegret = []
        for choices in self.orderChoices:
            chosenUserIndex= choices[0]
            chosenItemIndex = choices[1]
            if userIndex is not None:
                if chosenUserIndex != userIndex:
                    # Skip this user as it's not the wanted user
                    continue
            maxRating = self.bestScore(chosenUserIndex, bestRatingMatrix)
            resultantRating = self.ratingMatrix[chosenUserIndex][chosenItemIndex]
            # Update that spot to be taken
            bestRatingMatrix[chosenUserIndex][chosenItemIndex] = 0
            self.instantaneousRegret += (maxRating - resultantRating)
            self.cumulativeInstantaneousRegret.append(self.instantaneousRegret)
        return self.instantaneousRegret
