import random
import math

from .banditChoice import BanditChoice

class RandomChoice(BanditChoice):
    '''
    Randomly picks a legal choice.
    '''
    def __init__(self, ratingMatrix=None):
        BanditChoice.__init__(self, ratingMatrix)

    def evaluate(self, posteriorMatrix, legalItemVectorForUser, ratingMatrixForUser=None):
        itemIndex = 0 # An integer between [0, numItems - 1]
        choices = []
        for item in legalItemVectorForUser:
            if item == 1.0:
                choices.append(itemIndex)
            itemIndex += 1
        if len(choices) <= 0:
            raise ValueError("No legal choice available")
        return choices[int(math.floor(random.random() * len(choices)))]
