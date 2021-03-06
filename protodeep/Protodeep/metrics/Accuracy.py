import numpy as np

from Protodeep.utils.debug import class_timer


@class_timer
class Accuracy:

    def __init__(self, name='accuracy'):
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
            self.total += prediction.size
            self.count += np.sum(np.abs(prediction - target) < self.epsilon)
