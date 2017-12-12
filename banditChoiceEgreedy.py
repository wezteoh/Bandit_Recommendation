import numpy as np
import random

class BanditChoiceEgreedy(object):
    def __init__(self):
        pass
        
    def evaluate(self, posteriorMatrix, legalItemVector, ratingMatrixForUser=None):

        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_egreedy(user_ratings, user_indices)

        return itemIndex

    def get_egreedy(self, user_ratings, user_indices, epsilon=0.1):
            # default to greedy if no e specified
            mean_ratings = np.mean(user_ratings,axis=0)
            if random.random() < 1-epsilon:
                idx = np.argmax(mean_ratings)
            else:
                idx = random.randint(0, len(mean_ratings)-1)
            
            return user_indices[idx]

