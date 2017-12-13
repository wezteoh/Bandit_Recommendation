import numpy as np

from .banditChoice import BanditChoice

class WorstChoice(BanditChoice):
    '''
    Picks the worst feasible choice
    '''
    def __init__(self, ratingMatrix=None):
        BanditChoice.__init__(self, ratingMatrix)

    def evaluate(self, posteriorMatrix, legalItemVectorForUser, ratingMatrixForUser):
        itemIndex = 0 # An integer between [0, numItems - 1]
        choices = ratingMatrixForUser * legalItemVectorForUser
        maxRating = np.max(choices)
        # Set the 0 index to more than the max
        choices[np.where(choices == 0.0)] = maxRating + 1.0
        minIndex = np.argmin(choices)
        if legalItemVectorForUser[minIndex] == 0.0:
            raise ValueError("No legal choice available")
        return minIndex
