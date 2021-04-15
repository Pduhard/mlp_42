import numpy as np
from Protodeep.initializers.Initializer import Initializer


class GlorotNormal(Initializer):
    """
        Glorot Normal initializer AKA Xavier initializer

        [0 - sqrt(2 / (nin + nout))]

    """

    def initialize(self, shape, dtype=None, *args, **kwargs):
        return np.random.randn(*shape) * np.sqrt(2. / (shape[1] + shape[0]))
