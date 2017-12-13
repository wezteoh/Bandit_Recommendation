import numpy as np
import random

class BanditChoiceUCBEmpirical(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector, ratingMatrixForUser=None):
        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]

        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_ucb_empirical(user_ratings, user_indices)

        return itemIndex

    def get_ucb_empirical(self, user_ratings, user_indices, alpha=0.5):
        '''
        Given the sampled user ratings and valid indices,
        return the selected item based on:
        mean + alpha*std
        default exploration parameter alpha = 0.5
        '''
        
        mean_ratings = np.mean(user_ratings,axis=0)
        std_ratings = np.std(user_ratings,axis=0)

        confidence_bounds = mean_ratings + alpha*std_ratings

        idx = np.argmax(confidence_bounds)
        selected_item = user_indices[idx]

        return selected_item
