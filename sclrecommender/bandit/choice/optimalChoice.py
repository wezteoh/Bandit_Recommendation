import numpy as np

from .banditChoice import BanditChoice

class OptimalChoice(BanditChoice):
    '''
    Picks the most feasible optimal choice
    '''
    def __init__(self, ratingMatrix=None):
        BanditChoice.__init__(self, ratingMatrix)

    def evaluate(self, posteriorMatrix, legalItemVectorForUser, ratingMatrixForUser):
        itemIndex = 0 # An integer between [0, numItems - 1]
        choices = ratingMatrixForUser * legalItemVectorForUser
        maxIndex = np.argmax(choices)
        if legalItemVectorForUser[maxIndex] == 0.0:
            raise ValueError("No legal choice available")
        return maxIndex
