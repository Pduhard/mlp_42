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
                'unit_range': [20, 80],
                'fas': ['Linear', 'Relu', 'Sigmoid', 'Softmax', 'Tanh'],
                'initializers': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal', 'Zeros'],
                'regularizers': ['L1', 'L2', 'L1L2']
            },
            {
                'unit_range': [10, 30],
                'fas': ['Linear', 'Relu', 'Sigmoid', 'Softmax', 'Tanh'],
                'initializers': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal', 'Zeros'],
                'regularizers': ['L1', 'L2', 'L1L2']
            },
            {
                'unit_range': [2],
                'fas': ['Softmax'],
                'initializers': ['GlorotNormal', 'GlorotUniform', 'HeNormal', 'RandomNormal', 'Zeros'],
                'regularizers': [None]
            }
        ],
        dataset=dataset
    )

