from .recommenderMatrixAlgorithm import RecommenderMatrixAlgorithm

class NearestNeighbour(RecommenderMatrixAlgorithm):
    def __init__(self, ratingMatrix):
        super().__init__(ratingMatrix)

    def executeRecommender(self):
        # TODO:
        print("TODO!")

    # TODO: Maybe add a nearest neighbour algorithm? 
    # Child classes:
    # CollaborativeFiltering (both user and item in 1 class)
    # ContentBasedFiltering (both user and item in 1 class)
    # Hybrid (switches between the 2 depending on threshold)
