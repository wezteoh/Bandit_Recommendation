class UncertaintyModel(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        #self.posteriorMatrix = None

    def reset(self, seed=None):
        """
        Reset the weights as if no training was done.
        Reset seed.
        """
        pass

    def save(self, fname):
        return fname

    def load(self, fname):
        return True

    def train(self, legalTrainIndices, user, train_global):
        # Train the weights based on current legalTrainIndices matrix
        pass

    def sample_for_user(self, user_index, num_samples):
        # return (k, m) matrix of k samples for user i
        return None
        
    def save_uncertainty_progress(self, data_name, bandit_name, folder='BanditProgress'):
        location = "temp"
        return location
