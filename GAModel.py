import Protodeep as ptd
from random import randrange


class GAModel():

    def __init__(self, constraints, input_shape,
                 metrics=['categorical_accuracy'],
                 loss='BinaryCrossentropy', optimizer='Adam'):

        self.constraints = constraints
        self.input_shape = input_shape
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.create_model()

    def create_model(self, summary=True):
        i = ptd.layers.Input(self.input_shape)()
        out = i
        print('esf')
        for c in self.constraints:
            units = c['unit_range'][0] if len(c['unit_range']) == 1 else randrange(c['unit_range'][0], c['unit_range'][1])
            out = ptd.layers.Dense(
                units=units,
                activation=c['fas'][randrange(0, len(c['fas']))],
                kernel_initializer=c['initializers'][randrange(0, len(c['initializers']))],
                kernel_regularizer=c['regularizers'][randrange(0, len(c['regularizers']))]
            )(out)
        self.model = ptd.model.Model(inputs=i, outputs=out)
        self.model.compile(self.input_shape, metrics=self.metrics, loss=self.loss,
                           optimizer=self.optimizer)
        if summary:
            self.model.summary()

    def fit(self):
        pass
