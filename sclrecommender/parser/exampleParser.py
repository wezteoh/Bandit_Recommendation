from .recommenderParser import RecommenderParser

import numpy as np
import random

class ExampleParser(RecommenderParser):
    def __init__(self, dataDirectory=None):
        super().__init__(dataDirectory)
        self.ratingMatrix = None

    def getRatingMatrix(self, numUser, numItem):
        self.ratingMatrix = np.ones((numUser, numItem))
        for userIndex in range(numUser):
            for itemIndex in range(numItem):
                self.ratingMatrix[userIndex][itemIndex] = int(random.random() * 5)
        return self.ratingMatrix


