from .recommenderAlgorithm import RecommenderAlgorithm

class RecommenderMatrixAlgorithm(RecommenderAlgorithm):
    def __init__(self, ratingMatrix):
        super().__init__(ratingMatrix)
        self.recommenderMatrix = None
        self.rankingMatrix = None

    def getRecommenderMatrix(self):
        if self.recommenderMatrix is None:
            raise ValueError("Must execute self.executeRecommender first!")
        return self.recommenderMatrix

    def getRankingMatrix(self):
        return self.rankingMatrix

    def executeRanking(self, onlyUnknown):
        '''
        Assume already calculated reconstruction matrix
        Can generate ranking from it.
        onlyUnknown: True, Ignore things already purchased by setting to 0
        onlyUnknown: False, Include already rated/purchased items in ranking
        '''
        self.rankingMatrix = None
        # TODO:
        print("TODO IN THIS CLASS, NOT CHILD CLASS!") 
    #------------------------------------------------
    # Strategy Design Pattern
    #------------------------------------------------
    def executeRecommender(self):
        '''
        Sets the self.recommenderMatrix
        '''
        # Should be overriden by child classes
        raise NotImplementedError

