from .recommenderMatrix import RecommenderMatrix

class RatingMatrix(RecommenderMatrix):
    def __init__(self, ratingMatrix):
        super().__init__(ratingMatrix)
