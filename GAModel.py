import Protodeep as ptd
from random import randrange


class GAModel():

    def __init__(self, constraints, input_shape,
                 metrics=['categorical_accuracy'],
                 loss='BinaryCrossentropy', optimizer='Adam'):
        i = ptd.layers.Input(input_shape)()
        out = i
        print('esf')
        for c in constraints:
            units = c['unit_range'][0] if len(c['unit_range']) == 1 else randrange(c['unit_range'][0], c['unit_range'][1])
            print({'units': units,
                'activation': c['fas'][randrange(0, len(c['fas']))],
                'kernel_initializer': c['initializers'][randrange(0, len(c['initializers']))],
                'kernel_regularizer': c['regularizers'][randrange(0, len(c['regularizers']))]})
            out = ptd.layers.Dense(
                units=units,
                activation=c['fas'][randrange(0, len(c['fas']))],
                kernel_initializer=c['initializers'][randrange(0, len(c['initializers']))],
                kernel_regularizer=c['regularizers'][randrange(0, len(c['regularizers']))]
            )(out)
        self.model = ptd.model.Model(inputs=i, outputs=out)
        self.model.compile(input_shape, metrics=metrics, loss=loss,
                           optimizer=optimizer)
        self.model.summary()
