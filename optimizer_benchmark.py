import sys
import matplotlib.pyplot as plt
import numpy as np

import Protodeep as P
from dataset import Dataset
from Preprocessing.Split import Split

# def parse_option_value(opt, dflt):
#     if opt in sys.argv and sys.argv.index(opt) + 1 != len(sys.argv):
#         return sys.argv[sys.argv.index(opt) + 1]
#     return dflt


# def usage():
#     print("usage : blabla")
#     quit()


# def check_option(options):
#     return True


# def parse_options():
#     options = {
#         'optimizer': parse_option_value('-o', dflt=None),
#         'epoch': parse_option_value('-e', dflt='100'),
#         'csv_name': parse_option_value('-n', dflt='data.csv')
#         }
#     if check_option(options) is False:
#         usage()
#     return options


def get_model_Adam():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['categorical_accuracy'], optimizer='Adam')
    model.summary()
    return model

def get_model_Adam_amsgrad():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['categorical_accuracy'], optimizer=P.optimizers.Adam(amsgrad=True))
    model.summary()
    return model


def get_model_Adagrad():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['categorical_accuracy'], optimizer='Adagrad')
    model.summary()
    return model


def get_model_SGD():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['categorical_accuracy'], optimizer='SGD')
    model.summary()
    return model

def get_model_SGD_momentum():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['categorical_accuracy'], optimizer=P.optimizers.SGD(momentum=0.9))
    model.summary()
    return model


def get_model_RMSProp():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['categorical_accuracy'], optimizer='RMSProp')
    model.summary()
    return model


def get_model_RMSProp_momentum():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    # print(opt)
    # quit()
    model.compile(30, metrics=['categorical_accuracy'], optimizer=P.optimizers.RMSProp(momentum=0.9))
    model.summary()
    return model


if __name__ == "__main__":
    dataset = Dataset('data.csv')
    # model = get_troll_model_for_bonuses()
    model_SGD = get_model_SGD()
    model_SGD_momentum = get_model_SGD_momentum()
    model_RMSProp = get_model_RMSProp()
    model_RMSProp_momentum = get_model_RMSProp_momentum()
    model_Adam = get_model_Adam()
    model_Adam_amsgrad = get_model_Adam_amsgrad()
    model_Adagrad = get_model_Adagrad()
    ((x, y), (tx, ty)) = Split.train_test_split(
        dataset.features, dataset.targets, seed=303)
    # x, y = dataset.features, dataset.targets
    # tx, ty = x, y
    # tests = [get_model_Adam() for i in range(100)]
    # hists = [t.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping(patience=10)]) for t in tests]

    # for h in hists:
    #     plt.plot(h['val_loss'])
    #     print(f'loss: {h["val_loss"][-1]}')

    plt.show()
    history_SGD = model_SGD.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_SGD_momentum = model_SGD_momentum.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_RMSProp = model_RMSProp.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_RMSProp_momentum = model_RMSProp_momentum.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_Adam = model_Adam.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_Adam_amsgrad = model_Adam_amsgrad.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])
    history_Adagrad = model_Adagrad.fit(x, y, 100, 32, validation_data=(tx, ty), verbose=False, callbacks=[P.callbacks.EarlyStopping()])

    print(f'model SGD: {model_SGD.evaluate((tx, ty))}')
    print(f'model SGD momentum: {model_SGD_momentum.evaluate((tx, ty))}')
    print(f'model RMSProp : {model_RMSProp.evaluate((tx, ty))}')
    print(f'model RMSProp momentum : {model_RMSProp_momentum.evaluate((tx, ty))}')
    print(f'model Adam: {model_Adam.evaluate((tx, ty))}')
    print(f'model Adam amsgrad: {model_Adam_amsgrad.evaluate((tx, ty))}')
    print(f'model Adagrad: {model_Adagrad.evaluate((tx, ty))}')
    
    plt.plot(history_RMSProp['categorical_accuracy'])
    plt.plot(history_RMSProp_momentum['categorical_accuracy'])
    plt.plot(history_SGD['categorical_accuracy'])
    plt.plot(history_SGD_momentum['categorical_accuracy'])
    plt.plot(history_Adam['categorical_accuracy'])
    plt.plot(history_Adam_amsgrad['categorical_accuracy'])
    plt.plot(history_Adagrad['categorical_accuracy'])

    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['RMSProp', 'RMSProp_momentum', 'SGD', 'SGD_momentum', 'Adam', 'Adam_amsgrad', 'Adagrad'], loc='lower right')
    plt.show()

    plt.plot(history_RMSProp['val_loss'])
    plt.plot(history_RMSProp_momentum['val_loss'])
    plt.plot(history_SGD['val_loss'])
    plt.plot(history_SGD_momentum['val_loss'])
    plt.plot(history_Adam['val_loss'])
    plt.plot(history_Adam_amsgrad['val_loss'])
    plt.plot(history_Adagrad['val_loss'])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['RMSProp', 'RMSProp_momentum', 'SGD', 'SGD_momentum', 'Adam', 'Adam_amsgrad', 'Adagrad'], loc='upper right')
    plt.show()
