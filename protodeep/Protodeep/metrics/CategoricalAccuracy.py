import numpy as np
from Protodeep.utils.debug import class_timer


@class_timer
class CategoricalAccuracy:
    """
        Categorical Accurcy metric:
    """

    def __init__(self, name='categorical_accuracy'):
        self.name = name
        self.count = 0
        self.total = 0
        self.epsilon = 1e-8

    def reset_state(self):
        self.count = 0
        self.total = 0

    def result(self):
        if self.total == 0:
            return 0
        return self.count / self.total

    def update_state(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            self.total += prediction.shape[0]
            self.count += np.sum(
                np.argmax(prediction, axis=-1) == np.argmax(target, axis=-1)
            )
