from Protodeep.utils.env import Env
import numpy as np


class Initializer:

    def __call__(self, shape, dtype=None, *args, **kwargs):
        if Env.seed is not None:
            np.random.seed(Env.seed)
        return self.initialize(shape, dtype, args, kwargs)
