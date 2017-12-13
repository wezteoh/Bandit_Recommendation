class RecommenderMatrix(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix

    def getRatingMatrix(self):
        return self.ratingMatrix

    def applyMask(self, mask):
        if mask.shape != self.ratingMatrix.shape:
            raise Exception("Mask shape not same as rating matrix!")
        self.ratingMatrix *= mask
