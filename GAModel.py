import Protodeep as ptd
from random import randrange, choice, random
import numpy as np
# import tensorflow as tf

class GAModel():

    def __init__(self, constraints, input_shape, dataset,
                 metrics=['categorical_accuracy'],
                 loss='BinaryCrossentropy', optimizer='Adam',
                 model_attr=None):

        self.constraints = constraints
        self.input_shape = input_shape
        self.dataset = dataset
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.model_attr = model_attr
        self.create_model()

    def create_model(self, summary=False):

        self.models = []
        for m in range(1):
            inpt = ptd.layers.Input(self.input_shape)()
            out = inpt
            new_model_attr = self.model_attr is None
            if new_model_attr:
                self.model_attr = []
            for i, c in enumerate(self.constraints):
                if new_model_attr:
                    layer_attr = {
                        'unit': c['unit'][0] if len(c['unit']) == 1 else randrange(c['unit'][0], c['unit'][1]),
                        'fa': c['fa'][randrange(0, len(c['fa']))],
                        'initializer': c['initializer'][randrange(0, len(c['initializer']))],
                        'regularizer': c['regularizer'][randrange(0, len(c['regularizer']))]
                    }
                    self.model_attr.append(layer_attr)
                else:
                    layer_attr = self.model_attr[i]
                out = ptd.layers.Dense(
                    units=layer_attr['unit'],
                    activation=layer_attr['fa'],
                    kernel_initializer=layer_attr['initializer'],
                    kernel_regularizer=layer_attr['regularizer']
                )(out)
            model = ptd.model.Model(inputs=inpt, outputs=out)
            model.compile(self.input_shape, metrics=self.metrics, loss=self.loss,
                            optimizer=self.optimizer)
            if summary:
                model.summary()
            self.models.append(model)
    
    # def create_model(self, summary=False):

    #     self.models = []
    #     for m in range(1):
    #         inpt = tf.keras.Input(self.input_shape)
    #         out = inpt
    #         new_model_attr = self.model_attr is None
    #         if new_model_attr:
    #             self.model_attr = []
    #         for i, c in enumerate(self.constraints):
    #             if new_model_attr:
    #                 layer_attr = {
    #                     'unit': c['unit'][0] if len(c['unit']) == 1 else randrange(c['unit'][0], c['unit'][1]),
    #                     'fa': c['fa'][randrange(0, len(c['fa']))],
    #                     'initializer': c['initializer'][randrange(0, len(c['initializer']))],
    #                     'regularizer': c['regularizer'][randrange(0, len(c['regularizer']))]
    #                 }
    #                 self.model_attr.append(layer_attr)
    #             else:
    #                 layer_attr = self.model_attr[i]
    #             out = tf.keras.layers.Dense(
    #                 units=layer_attr['unit'],
    #                 activation=layer_attr['fa'],
    #                 kernel_initializer=layer_attr['initializer'],
    #                 kernel_regularizer=layer_attr['regularizer']
    #             )(out)
    #         model = tf.keras.Model(inputs=inpt, outputs=out)
    #         model.compile(metrics=self.metrics, loss=self.loss,
    #                       optimizer=self.optimizer)
    #         if summary:
    #             model.summary()
    #         self.models.append(model)
            
    def evaluate(self, x_train, y_train, x_test, y_test):

        losses = []
        for model in self.models:
            self.logs = model.fit(
                x_train, y_train, epochs=100, validation_data=(x_test, y_test),
                callbacks=[ptd.callbacks.EarlyStopping(restore_best_weights=True)],
                # callbacks=[ptd.callbacks.EarlyStopping(baseline=0.08, restore_best_weights=True)],
                verbose=False
            )
            # print(history.history.keys())
            # print(self.logs.history['val_loss'])
            losses.append(self.logs['val_loss'][-1])
        
        return sum(losses) / len(losses)

             
    # def evaluate(self, x_train, y_train, x_test, y_test):

    #     losses = []
    #     for model in self.models:
    #         self.logs = model.fit(
    #             x_train, y_train, epochs=100, validation_data=(x_test, y_test),
    #             callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True)],
    #             # callbacks=[ptd.callbacks.EarlyStopping(baseline=0.08, restore_best_weights=True)],
    #             verbose=False
    #         )
    #         # print(history.history.keys())
    #         # print(self.logs.history['val_loss'])
    #         losses.append(self.logs.history['val_loss'][-1])
        
    #     return sum(losses) / len(losses)


    def fit(self, x_train, y_train, x_test, y_test):
        
        loss = self.evaluate(x_train, y_train, x_test, y_test)

        self.fitness = 1 / (loss ** 2 * 2)
        if np.isnan(loss):
            self.fitness = 0

            print(self.model_attr)

        print(f"fitness: {self.fitness} -- loss: {loss}")
        return self.fitness

    def mutate_attr(self, l, key):
        if key == 'unit':
            return randrange(*self.constraints[l][key]) if len(self.constraints[l][key]) > 1 else self.constraints[l][key][0]
        else:
            return choice(self.constraints[l][key])

    def cross(self, b, mutation_rate):
        cross_model = []
        for l, (ma, mb) in enumerate(zip(self.model_attr, b.model_attr)):

            farand = [randrange(0, 2) for i in range(4)]

            cross_model.append({key: self.mutate_attr(l, key) if random() < mutation_rate else (ma[key] if farand[i] else mb[key]) for i, key in enumerate(mb)})
            # print(farand)
            # print(ma, mb, cross_model)

        return GAModel(self.constraints, self.input_shape, self.dataset,
                       model_attr=cross_model)
        # baby = GAModel(cross)
        # baby.create_model()
