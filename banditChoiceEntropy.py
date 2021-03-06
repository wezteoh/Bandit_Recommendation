import numpy as np

class BanditChoiceEntropy(object):
    def __init__(self, lam=0.5, begin=0.0, end=6.0):
        self.lam = lam
        self.begin = begin
        self.end = end
        
    def evaluate(self, posteriorMatrix, legalItemVector, ratingMatrixForUser=None):

        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_sample(user_ratings, user_indices)

        return itemIndex

    def get_sample(self, user_ratings, user_indices):

        mean_ratings = np.mean(user_ratings,axis=0)
        entropy = self.empirical_entropy(user_ratings)

        teoh_score = mean_ratings + self.lam*entropy

        idx = np.argmax(teoh_score)
        selected_item = user_indices[idx]

        return selected_item

    def empirical_entropy(self, samples, num_bins=12):
        # histogram method
        # put variables into 6/(num_bins)-width bins from 0 to 5
        bounds = np.linspace(self.begin,self.end,num_bins+1)
        frequencies = np.zeros((num_bins, samples.shape[1]))
        num_samples = samples.shape[0]
        for i in range(num_bins):
            indicators = samples*(samples <= bounds[i+1]) > bounds[i]
            frequencies[i] = np.sum(indicators, axis=0)
    
        # to get rid of 0s when taking log    
        mock_frequencies = frequencies + (frequencies==0)
        return -np.sum((frequencies/num_samples)*(np.log(mock_frequencies/(num_samples*0.5))), axis=0)


