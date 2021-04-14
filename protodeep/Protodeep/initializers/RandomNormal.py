import numpy as np


class RandomNormal():
    """
        Random Normal initializer

    """

    def __call__(self, shape, mean=0.0, stddev=0.05):
        return np.random.normal(loc=mean, scale=stddev, size=shape)
