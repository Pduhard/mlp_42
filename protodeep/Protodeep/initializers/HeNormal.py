import numpy as np
from Protodeep.initializers.Initializer import Initializer


class HeNormal(Initializer):
    """
        He Normal initializer

        [0 - sqrt(2 / nout)]
    """

    def initialize(self, shape, dtype=None, *args, **kwargs):
        return np.random.randn(*shape) * np.sqrt(2 / shape[-1])
