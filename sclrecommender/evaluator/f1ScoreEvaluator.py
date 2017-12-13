import numpy as np

from .positiveNegativeEvaluator import PositiveNegativeEvaluator

class F1ScoreEvaluator(PositiveNegativeEvaluator):
    '''
    F1Score is the harmonic mean between Recall and Precision. 
    If you decide to suggest everything, your recall would always be 1.0. 
    This means that the arithmetic mean will always be >= 0.5.
    This is not good as you want a nice balance between Recall and Precision. 
    harmonic mean is closer towards min(Recall, Precision), so it is sort of worst case. 
    If you increase harmonic mean, it is like trying to maxmin.
    Or trying to maximize your worst case, which is what we care about.
    '''
    def __init__(self, ratingMatrix, reconstructionMatrix, positiveThreshold=3):
        super().__init__(ratingMatrix, reconstructionMatrix, positiveThreshold)
        self.precision = 0.0
        self.recall = 0.0
        self.f1Score = 0.0

    def evaluate(self):
        '''
        Calculates f1-score for the ratings
        '''
        conditionPositive = np.where(self.truthPositiveNegative == 1.0)[0].size
        predictionPositive = np.where(self.predictedPositiveNegative == 1.0)[0].size
        truePositive = np.where(np.logical_and(self.truthPositiveNegative == 1.0, self.predictedPositiveNegative == 1.0) == True)[0].size
        if conditionPositive:
            self.precision = truePositive/float(conditionPositive)
        if predictionPositive:
            self.recall = truePositive/float(predictionPositive)
        if (self.precision + self.recall):
            self.f1Score =  (2.0*self.precision*self.recall)/float(self.precision+self.recall)
        return self.f1Score

    def getPrecision(self):
        self.evaluate()
        return self.precision

    def getRecall(self):
        self.evaluate()
        return self.recall

    def getF1Score(self):
        self.evaluate()
        return self.f1Score

