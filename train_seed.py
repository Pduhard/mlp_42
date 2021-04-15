from dataset import Dataset
from Protodeep.model.model import Model
# from layers.Dense import Dense 
# from layers.Dense import Dense 
from Protodeep.layers.Dense import Dense
from Protodeep.layers.Conv2D import Conv2D
from Protodeep.layers.Flatten import Flatten
from Protodeep.layers.MaxPool2D import MaxPool2D
from Protodeep.regularizers.L1L2 import L1L2
from Protodeep.utils.env import Env
import sys
import Protodeep
# import numpy as np
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping

import numpy as np


def parse_option_value(opt, dflt):
    if opt in sys.argv:
        if sys.argv.index(opt) + 1 != len(sys.argv):
            return sys.argv[sys.argv.index(opt) + 1]
    return dflt


def usage():
    print("usage : blabla")
    quit()


def check_option(options):
    return True


def parse_options():
    options = {
        'optimizer': parse_option_value('-o', dflt=None),
        'epoch': parse_option_value('-e', dflt='100'),
        'csv_name': parse_option_value('-n', dflt='data.csv')
        }
    if check_option(options) is False:
        usage()
    return options


def get_model_regularizers():
    model = Model()
    # model.add(Protodeep.layers.Dense(64, activation=Protodeep.activations.Relu(), kernel_regularizer=Protodeep.regularizers.L1L2()))
    model.add(Protodeep.layers.Dense(64, activation=Protodeep.activations.Relu(), kernel_regularizer=''))
    model.add(Protodeep.layers.Dense(32, activation=Protodeep.activations.Relu(), kernel_regularizer=''))
    model.add(Protodeep.layers.Dense(2, activation=Protodeep.activations.Softmax()))
    model.compile(30, metrics=[Protodeep.metrics.CategoricalAccuracy(), Protodeep.metrics.Accuracy()], optimizer=Protodeep.optimizers.Adam())
    model.summary()
    return model


def get_basic_model():
    model = Model()
    model.add(Protodeep.layers.Dense(64, activation=Protodeep.activations.Relu()))
    model.add(Protodeep.layers.Dense(32, activation=Protodeep.activations.Relu()))
    model.add(Protodeep.layers.Dense(2, activation=Protodeep.activations.Softmax()))
    model.compile(30, metrics=[Protodeep.metrics.CategoricalAccuracy()], optimizer=Protodeep.optimizers.Adam())
    # model.summary()
    return model


def get_troll_model_for_bonuses():
    model = Model()
    model.add(Protodeep.layers.Conv2D(filters=30, kernel_size=(2, 2), activation=Protodeep.activations.Relu()))
    model.add(Protodeep.layers.Flatten())
    model.add(Protodeep.layers.Dense(32, activation=Protodeep.activations.Relu()))
    model.add(Protodeep.layers.Dense(2, activation=Protodeep.activations.Softmax()))
    model.compile((5, 6, 1), metrics=[Protodeep.metrics.CategoricalAccuracy(), Protodeep.metrics.Accuracy()], optimizer=Protodeep.optimizers.Adam())
    model.summary()
    return model


if __name__ == "__main__":
    Env.set_seed(35)
    options = parse_options()
    dataset = Dataset(options['csv_name'], 0.2)
    model = get_basic_model()

    history = model.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=500,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
    )
    model.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    )
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    # plt.plot(history['accuracy'])
    # plt.plot(history['val_accuracy'])
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend([
        'categorical_accuracy',
        'val_categorical_accuracy',
        'accuracy',
        'val_accuracy'], loc='lower right')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    best = 100
    best_seed = 0 
    best_a = 0
    best_seed_a = 0 
    for i in range(1000):
        Env.set_seed(i)
        model = get_basic_model()

        history = model.fit(
            features=dataset.features,
            targets=dataset.targets,
            epochs=500,
            batch_size=32,
            validation_data=(dataset.test_features, dataset.test_targets),
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=False
        )
        eval_log = model.evaluate(validation_data=(dataset.test_features, dataset.test_targets))
        if eval_log["loss"] < best:
            best = eval_log["loss"]
            best_seed = i
        if eval_log["categorical_accuracy"] > best_a:
            best_a = eval_log["categorical_accuracy"]
            best_seed_a = i
        print(f'seed: {i}, score: {eval_log["loss"]} best: {(best, best_seed), (best_a, best_seed_a)}')
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend([
        'categorical_accuracy',
        'val_categorical_accuracy',
        'accuracy',
        'val_accuracy'], loc='lower right')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
