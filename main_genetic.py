from dataset import Dataset
from scalers.StandardScaler import StandardScaler
from Preprocessing.Split import Split
from Genetic import Genetic
import Protodeep as pdt

if __name__ == "__main__":
    dataset = Dataset('data.csv', 0.2)

    scaler = StandardScaler().fit(dataset.features)
    dataset.features = scaler.transform(dataset.features)
    scaler.save()

    ((x_train, y_train), (x_test, y_test)) = Split.train_test_split(
        dataset.features, dataset.targets)

    print(pdt.activations.__all__[:-1])
    gen = Genetic(
        constraints=[
            {
                'unit': [20, 80],
                'fa': ['linear', 'relu', 'sigmoid', 'softmax', 'tanh'],
                'initializer': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal'],
                'regularizer': [None]
            },
            {
                'unit': [10, 30],
                'fa': ['linear', 'relu', 'sigmoid', 'softmax', 'tanh'],
                'initializer': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal'],
                'regularizer': [None]
            },
            {
                'unit': [2],
                'fa': ['softmax'],
                'initializer': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal'],
                'regularizer': [None]
            }
        ],
        dataset=dataset
    )

    model = gen.find_model()
