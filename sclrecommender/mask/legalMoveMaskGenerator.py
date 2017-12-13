import numpy as np
from .maskGenerator import MaskGenerator

class LegalMoveMaskGenerator(MaskGenerator):
    def __init__(self, ratingMatrix):
        super().__init__(ratingMatrix)
        self.mask = np.ones(ratingMatrix.shape) 
        self.mask[np.where(ratingMatrix == 0)] = 0.0
