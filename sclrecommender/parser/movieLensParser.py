from .recommenderParser import RecommenderParser

import os
import numpy as np

class MovieLensParser20m(RecommenderParser):
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)
        self.dataFile = os.path.join(self.dataDirectory, 'ratings.csv')

        self.movieIdToItemId= dict()
        self.ratingMatrix = self.parseRatingMatrix()

    def getRatingMatrixCopy(self):
        return self.ratingMatrix.copy()

    def parseRatingMatrix(self):
        # (numUser, numItem)
        ratingMatrix = np.zeros((138493, 27278))
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile) as dataFile:
            firstLine = True
            for currLine in dataFile:
                if firstLine:
                    firstLine = False
                    continue
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split(",")))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)

        ''' Convert arbitrary movieId to ordered movieIds '''
        # Start with 0
        uniqueItemId = 0
        for currRating in arr:
            if currRating.movieId in self.movieIdToItemId:
                ratingMatrix[currRating.userId-1][self.movieIdToItemId[currRating.movieId]] = currRating.rating
            else:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                uniqueItemId += 1
        return ratingMatrix

class MovieLensParser1m(RecommenderParser):
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)
        self.dataFile = os.path.join(self.dataDirectory, 'ratings.dat')

        self.movieIdToItemId= dict()
        self.ratingMatrix = self.parseRatingMatrix()

    def getRatingMatrixCopy(self):
        return self.ratingMatrix.copy()

    def parseRatingMatrix(self):
        # (numUser, numItem)
        ratingMatrix = np.zeros((6040, 3952)) # For 1m dataset according to readme
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile) as dataFile:
            firstLine = True
            for currLine in dataFile:
                if firstLine:
                    firstLine = False
                    continue
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split("::")))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)

        ''' Convert arbitrary movieId to ordered movieIds '''
        # Start with 0
        uniqueItemId = 0
        for currRating in arr:
            if currRating.movieId in self.movieIdToItemId:
                ratingMatrix[currRating.userId-1][self.movieIdToItemId[currRating.movieId]] = currRating.rating
            else:
                self.movieIdToItemId[currRating.movieId] = uniqueItemId
                uniqueItemId += 1
        print("Number of unique items: ", uniqueItemId)
        return ratingMatrix


class MovieLensParser100k(RecommenderParser):
    def __init__(self, dataDirectory):
        super().__init__(dataDirectory)
        self.dataFile = os.path.join(self.dataDirectory, 'u.data')
        self.genreFile = os.path.join(self.dataDirectory, 'u.genre')
        self.itemFile = os.path.join(self.dataDirectory, 'u.item')
        self.occupationFile = os.path.join(self.dataDirectory, 'u.occupation')
        self.userFile = os.path.join(self.dataDirectory, 'u.user')

        self.ratingMatrix = self.parseRatingMatrix()

    def getRatingMatrixCopy(self):
        return self.ratingMatrix.copy()

    def parseRatingMatrix(self):
        ratingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile) as dataFile:
            for currLine in dataFile:
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split()))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)
        for currRating in arr:
            ratingMatrix[currRating.userId-1][currRating.movieId-1] = currRating.rating
        return ratingMatrix


class MovieLensRating(object):
    """
    UserItemRating
    Represents a single row in user item matrix
    """

    def __init__(self, userId, movieId, rating, timeStamp):
        self.userId= int(userId)
        self.movieId = int(movieId)
        self.rating = float(rating)
        self.timeStamp = int(timeStamp)

    def __eq__(self, other):
        return (isinstance(other, MovieLensRating) and
                self.userId, self.movieId, self.rating, self.timeStamp ==
                other.userId, other.movieId, other.rating, other.timeStamp)

    # Use __lt__ for python3 compatibility.
    def __lt__(self, other):
        return self.timeStamp < other.timeStamp

    def __hash__(self):
        return hash((self.userId, self.movieId, self.rating, self.timeStamp))
