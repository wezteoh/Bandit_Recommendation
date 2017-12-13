
class RecommenderAlgorithm(object):
    def __init__(self, ratingMatrix):
        # 0 => No idea since not rated
        # 1->5 => Rating given by user
        self.ratingMatrix = ratingMatrix

    def generateRanking(self):
        # get the ranking matrix of the final predictions
        # the 2D matrix represent each user, 
        # where each cell contains a number which represents the rank in which
        # a user would purchase that item starting from highest number to 1
        # 0
        # e.g. 
        # Input: Purchase Matrix
        # [[ 0 0 1]
        #  [ 1 0 1]
        #  [ 0 0 0]
        # Output: Ranking Matrix
        # [[ 2 1 0]  # This user purchase first item, then second item
        #  [ 0 1 0]  # This user purchase second item only
        #  [ 2 1 3]] # This user purchase 3rd item, then first item, then 2nd item
        # This indicates the first user would buy first item and then second item
        raise NotImplementedError # Should be overridden by child
