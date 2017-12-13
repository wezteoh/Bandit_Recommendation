from .evaluator import Evaluator

class RankingEvaluator(Evaluator):
    def __init__(self, ratingMatrix, rankingMatrix):
        Evaluator.__init__(self, ratingMatrix)
        self.rankingMatrix = rankingMatrix

    def evaluate(self):
        raise NotImplementedError
