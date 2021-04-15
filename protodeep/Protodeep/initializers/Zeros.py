import numpy as np
from Protodeep.initializers.Initializer import Initializer


class Zeros(Initializer):
    """
        Zeros initializer

        [0]
    """

    def initialize(self, shape, dtype=None, *args, **kwargs):
        return np.zeros(shape)
