import random
import math

class BanditChoice(object):
    def __init__(self, ratingMatrix=None):
        self.ratingMatrix = ratingMatrix

    def evaluate(self, posteriorMatrix, legalItemVectorForUser, ratingMatrixForUser=None):
        # posteriorMatrix.shape = (numSamples, numItems)
        # legalItemVectorForAUser = (numItems)
        itemIndex = 0 # An integer between [0, numItems - 1]
        choices = []
        for item in legalItemVectorForUser:
            if item == 1.0:
                choices.append(itemIndex)
            itemIndex += 1
        return choices[int(math.floor(random.random() * len(choices)))]
