import numpy as np
from Protodeep.initializers.Initializer import Initializer


class RandomNormal(Initializer):
    """
        Random Normal initializer

    """

    def initialize(self, shape, mean=0.0, stddev=0.05, *args, **kwargs):
        return np.random.normal(loc=mean, scale=stddev, size=shape)
