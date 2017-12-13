import numpy as np

'''
This class handles transforming the ratingMatrix to arbitrary size
'''
class MatrixTransform(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix.copy()
        self.binaryMatrix= self.initBinaryMatrix()

    def initBinaryMatrix(self):
        binaryMatrix = self.ratingMatrix.copy()
        binaryMatrix[np.where(self.ratingMatrix == 0.0)] = 0.0
        binaryMatrix[np.where(self.ratingMatrix != 0.0)] = 1.0
        return binaryMatrix

    def sparseUsers(self, maxItem):
        '''
        Only keep users who rated at most maxItem
        '''
        numItemsPerUser = self.binaryMatrix.sum(axis=1)
        raise NotImplementedError

    def denseUsers(self, minItem):
        '''
        Only keep users who has rated at least minItem
        '''
        numItemsPerUser = self.binaryMatrix.sum(axis=1)
        raise NotImplementedError
        

    def hotUsers(self, numUsers=None): 
        '''
        Get the top numUsers with a lot of items
        '''
        if numUsers is None:
            numUsers = self.ratingMatrix.shape[0]
        userList = []
        for currUser in range(self.ratingMatrix.shape[0]):
            hotUser = SortedNumRatingUser(self.ratingMatrix[currUser])
            userList.append(hotUser)
        # Sort from most ratings to few ratings
        sortedList = sorted(userList, reverse=True)
        topNumUsers = sortedList[:numUsers]
        # Form rating matrix
        ratingMatrix = [user.ratingItems for user in topNumUsers]
        return np.array(ratingMatrix)


    def coldUsers(self, numUsers=None):
        '''
        Get the top numUsers of coldStartUsers
        '''
        if numUsers is None:
            numUsers = self.ratingMatrix.shape[0]
        userList = []
        for currUser in range(self.ratingMatrix.shape[0]):
            hotUser = SortedNumRatingUser(self.ratingMatrix[currUser])
            userList.append(hotUser)
        # Sort from few ratings to most ratings
        sortedList = sorted(userList)
        topNumUsers = sortedList[:numUsers]
        # Form rating matrix
        ratingMatrix = [user.ratingItems for user in topNumUsers]
        return np.array(ratingMatrix)

class SortedNumRatingUser(object):
    def __init__(self, ratingItems):
        self.ratingItems = ratingItems
        self.binaryRating = self.ratingItems.copy()
        self.binaryRating[np.where(self.ratingItems == 0.0)] = 0.0
        self.binaryRating[np.where(self.ratingItems != 0.0)] = 1.0

    def __lt__(self, other):
        return self.binaryRating.sum() < other.binaryRating.sum()

    def __repr__(self):
        return "TotalNumRating: {}, binaryRating: {}".format(self.binaryRating.sum(), self.binaryRating)
