import numpy as np

class BanditChoiceBoltzmann(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector, ratingMatrixForUser=None):
        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_boltzmann_exploration(user_ratings, user_indices)

        return itemIndex


    def get_boltzmann_exploration(self, user_ratings, user_indices, tau=0.1):
        '''
        Convert predicted means to probabilities for each item p_j
        select each item with probability p_j
        tau is the temperature scaling parameter
        default tau as 0.1
        '''
        temp_scaled = [x/tau for x in np.mean(user_ratings,axis=0)]
        denom = np.sum(np.exp(temp_scaled))
        boltzmann_prob = [np.exp(x)/denom for x in temp_scaled]

        indices = list(range(len(boltzmann_prob)))
        idx = np.random.choice(indices,p=boltzmann_prob)
        selected_item = user_indices[idx]

        return selected_item
