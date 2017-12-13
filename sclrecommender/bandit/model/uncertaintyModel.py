import numpy as np

class UncertaintyModel(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix

    def reset(self, seed=None):
        # Reset the weights as if no training was done
        pass

    def save(self, fileName):
        # Save the model
        return fileName

    def load(self, fileName):
        # Load the model
        return True

    def train(self, legalTrainIndices):
        # Train the weights based on current legalTrainIndices
        pass 

    #def sampleForUser(self, userIndex, numSamples):
    def sample_for_user(self, userIndex, numSamples):
        # Return the samples for a given user of size
        # (numSamples, numItems)
        samples = np.ones((numSamples, self.ratingMatrix.shape[1]))
        return samples

