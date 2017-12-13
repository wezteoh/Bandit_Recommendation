from .rankingEvaluator import RankingEvaluator

class BanditEvaluator(RankingEvaluator):
    def __init__(self, ratingMatrix, rankingMatrix, discountFactor):
        RankingEvaluator.__init__(self, ratingMatrix, rankingMatrix)
        if discountFactor >= 1.0 or discountFactor < 0.0:
            raise ValueError("Discount factor is not within [0, 1)")
        self.discountFactor = discountFactor

    def evaluate(self):
        raise NotImplementedError
