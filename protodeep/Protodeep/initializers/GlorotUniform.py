import numpy as np
from Protodeep.initializers.Initializer import Initializer


class GlorotUniform(Initializer):
    """
        Glorot Uniform initializer

        [0 - sqrt(6 / (nin + nout))]

    """

    def initialize(self, shape, dtype=None, *args, **kwargs):
        return np.random.randn(*shape) * np.sqrt(6. / (shape[1] + shape[0]))
