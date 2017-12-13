class Evaluator(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix

    def evaluate(self):
        # Should be overridden
        raise NotImplementedError
