import numpy as np

class MatrixAnalyzer(object):

    '''
    This mask generator just returns everything
    '''
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        self.binaryMatrix= self.initBinaryMatrix()

    def initBinaryMatrix(self):
        binaryMatrix = self.ratingMatrix.copy()
        binaryMatrix[np.where(self.ratingMatrix == 0.0)] = 0.0
        binaryMatrix[np.where(self.ratingMatrix != 0.0)] = 1.0
        return binaryMatrix

    def getNumUser(self):
        return self.ratingMatrix.shape[0]

    def getNumItem(self):
        return self.ratingMatrix.shape[1]

    def getMatrixSize(self):
        return self.ratingMatrix.size

    def getNumMissing(self):
        '''
        Return number of zero entries
        '''
        return len(np.where(self.ratingMatrix == 0.0)[0])

    def getMinNumItemsPerUser(self):
        '''
        Return minimum number of non-zero entries for any user
        '''
        numItemsPerUser = np.sum(self.binaryMatrix, axis = 1)
        return np.min(numItemsPerUser)

    def getMaxNumItemsPerUser(self):
        numItemsPerUser = np.sum(self.binaryMatrix, axis = 1)
        return np.max(numItemsPerUser)

    def summarize(self):
        '''
        Prints all the summaries of it's analysis
        '''
        print("Number of users: ", self.getNumUser())
        print("Number of items: ", self.getNumItem())
        print("Size of matrix: ", self.getMatrixSize())
        print("Number of missing cells: ", self.getNumMissing())
        print("Minimum number of items per user: ", self.getMinNumItemsPerUser())
        print("Maximum number of items per user: ", self.getMaxNumItemsPerUser())

