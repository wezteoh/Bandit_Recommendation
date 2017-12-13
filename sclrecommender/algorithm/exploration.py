from .recommenderAlgorithm import RecommenderAlgorithm

class Exploration(RecommenderAlgorithm):
    def __init__(self, ratingMatrix):
        super().__init__(ratingMatrix)

    def generateRanking(self):
        '''
        Generates the ranking of items matrix for a user
        e.g. 
        [ 2 4 0 0] # First item has truth rating 2, 2nd item has truth rating 4, no more picked
        [ 3 0 0 0] # Only picked first item with rating 3
        [ 5 4 5 0] # Picked 3 item, first and third rating 5, 2nd rating 4.
        '''
        # To be overridden by child classes
        raise NotImplementedError
